/*
 * Fermat
 *
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vertex.h>
#include <bsdf.h>
#include <edf.h>
#include <lights.h>
#include <camera.h>
#include <renderer.h>
#include <mesh_utils.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>
#include <cugar/color/rgbe.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

///@addtogroup BPTLibCore
///@{

#define DEBUG_S				-1
#define DEBUG_T				-1
#define DEBUG_LENGTH		0

#define MIN_G_DENOM			1.0e-8f

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float mis_selector(const uint32 s, const uint32 t, const float w)
{
	return
		DEBUG_LENGTH ? (DEBUG_LENGTH == s + t - 1 ? w : 0.0f) :
		(DEBUG_S == -1 && DEBUG_T == -1) ? w :
		(DEBUG_S ==  s && DEBUG_T ==  t) ? 1.0f :
		(DEBUG_S ==  s && DEBUG_T == -1) ? 1.0f :
		(DEBUG_S == -1 && DEBUG_T ==  t) ? 1.0f :
		0.0f;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float mis_power(const float w) { return w/**w*/; }

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float bpt_mis(const float pGp, const float prev_pGp, const float next_pGp, const float pGp_sum)
{
	return pGp && prev_pGp && next_pGp ? mis_power(1 / pGp) / (mis_power(1 / pGp) + mis_power(1 / prev_pGp) + mis_power(1 / next_pGp) + pGp_sum) : 0.0f;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float bpt_mis(const float pGp, const float other_pGp, const float pGp_sum)
{
	return pGp && other_pGp ? mis_power(1 / pGp) / (mis_power(1 / pGp) + mis_power(1 / other_pGp) + pGp_sum) : 0.0f;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float pdf_product(const float p1, const float p2)
{
	return
		cugar::is_finite(p1) && 
		cugar::is_finite(p2) ? p1 * p2 : cugar::float_infinity();
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float pdf_product(const float p1, const float p2, const float p3)
{
	return 
		cugar::is_finite(p1) && 
		cugar::is_finite(p2) && 
		cugar::is_finite(p3) ? p1 * p2 * p3 : cugar::float_infinity();
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint32 channel_selector(const Bsdf::ComponentType comp)
{
	return (comp & Bsdf::kDiffuseMask) ? FBufferDesc::DIFFUSE_C : FBufferDesc::SPECULAR_C;
}

///\par
/// A small utility class to track the path weights needed for internal MIS calculations
///
struct PathWeights
{
	FERMAT_HOST_DEVICE PathWeights() {}
	FERMAT_HOST_DEVICE PathWeights(const float _pGp_sum, const float _pG) :
		pGp_sum(_pGp_sum), pG(_pG) {}
	FERMAT_HOST_DEVICE PathWeights(const float2 f) : vector_storage(f) {}

	FERMAT_HOST_DEVICE operator float2() const { return vector_storage; }

	union {
		float2 vector_storage;
		struct {
			float pGp_sum;
			float pG;
		};
	};
};

///\par
/// A small utility class to track temporary path weights needed for internal MIS calculations
///
struct TempPathWeights
{
	FERMAT_HOST_DEVICE TempPathWeights() {}
	FERMAT_HOST_DEVICE TempPathWeights(const PathWeights _weights, const float _out_p, const float _out_cos_theta) :
		pGp_sum(_weights.pGp_sum), pG(_weights.pG), out_p(_out_p), out_cos_theta(_out_cos_theta) {}
	FERMAT_HOST_DEVICE TempPathWeights(const float _pGp_sum, const float _pG, const float _out_p, const float _out_cos_theta) :
		pGp_sum(_pGp_sum), pG(_pG), out_p(_out_p), out_cos_theta(_out_cos_theta) {}
	FERMAT_HOST_DEVICE TempPathWeights(const float4 f) : vector_storage(f) {}

	template <typename EyeLightVertexType>
	FERMAT_HOST_DEVICE TempPathWeights(const EyeLightVertexType _v, const cugar::Vector3f _out, const float _out_p) :
		pGp_sum(_v.pGp_sum),										// 1 / [p(i-2)g(i-2)p(i-1)] + 1 / [p(i-3)g(i-3)p(i-2)] + ... + 1 / [p(-1)g(-1)p(0)]
		pG(_v.prev_pG),												// p(i-1)g(i-1)
		out_p(_out_p),												// p(i)
		out_cos_theta(fabsf(dot(_v.geom.normal_s, _out))) {}		// cos(theta_i)

	// return the temporary weights for the second camera vertex (i.s. t=1)
	FERMAT_HOST_DEVICE 
	static TempPathWeights eye_vertex_1(const float p_e, const float cos_theta_o, const float light_tracing_weight)
	{
		return TempPathWeights(
			0.0f,															// 1 / p(-2)g(-2)p(-1)
			1.0e8f,															// p(-1)g(-1)		= +inf : hitting the camera is impossible
			light_tracing_weight ? p_e / light_tracing_weight	: 1.0f,		// out_p = p(0)
			light_tracing_weight ? cos_theta_o					: 1.0e8f);	// out_cos_theta : so that the term f_0 g_0 f_1 (i.e. connection to the lens) gets the proper weight
																			//			+inf : so that the term f_0 g_0 f_1(i.e.connection to the lens) gets zero weight
	}

	// return the temporary weights for the second light vertex (i.e. s=1)
	FERMAT_HOST_DEVICE 
	static TempPathWeights light_vertex_1(const float p_A, const float p_proj, const float cos_theta_o)
	{
		return TempPathWeights(
			0.0f,								// p(-2)g(-2)p(-1)
			1.0f * p_A,							// p(-1)g(-1) = 1			: we want p(-1)g(-1)p(0) = p(0) - which will happen because we are setting p(0) = p_sigma(0) and g(-1) = p_A(0)
			p_proj,								// p(0)
			cos_theta_o);						// cos(theta_0)
	}

	FERMAT_HOST_DEVICE operator PathWeights() const { return PathWeights(pGp_sum, pG); }
	FERMAT_HOST_DEVICE operator float4() const { return vector_storage; }

	union {
		float4 vector_storage;
		struct
		{
			float pGp_sum;
			float pG;
			float out_p;
			float out_cos_theta;
		};
	};
};

///\par
/// Pack an EDF into a uint4
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_edf(const Edf& edf)
{
	return make_uint4(
		cugar::to_rgbe(edf.color),
		0,
		0,
		0);
}

///\par
/// Pack an EDF into a uint4
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_edf(const MeshMaterial& material)
{
	return make_uint4(
		cugar::to_rgbe(cugar::Vector4f(material.emissive).xyz()),
		0,
		0,
		0);
}

///\par
/// Pack an BSDF into a uint4
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_bsdf(const MeshMaterial& material)
{
	const uint32 roughness_i = (uint16)cugar::quantize( material.roughness,  65535u );
	const uint32 opacity_i   = (uint16)cugar::quantize( material.opacity, 255u );
	const uint32 ior_i       = (uint16)cugar::quantize( material.index_of_refraction / 3.0f, 255u );
	return make_uint4(
		cugar::to_rgbe(cugar::Vector4f(material.diffuse).xyz()),
		cugar::to_rgbe(cugar::Vector4f(material.specular).xyz()),
		roughness_i | (opacity_i << 16) | (ior_i << 24),
		cugar::to_rgbe(cugar::Vector4f(material.diffuse_trans).xyz()));
}

///\par
/// Unpack an EDF from a uint4
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Edf unpack_edf(const uint4 packed_info)
{
	return Edf(cugar::from_rgbe(packed_info.x));
}

///\par
/// Unpack an BSDF from a uint4
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Bsdf unpack_bsdf(const RenderingContextView& renderer, const uint4 packed_info, const TransportType transport = kParticleTransport)
{
	const float roughness = (packed_info.z & 65535u) / 65535.0f;
	const float opacity   = ((packed_info.z >> 16) & 255) / 255.0f;
	const float ior       = cugar::max( 3.0f * ((packed_info.z >> 24) / 255.0f), 0.00001f );

	return Bsdf(
		transport,
		renderer,
		cugar::from_rgbe(packed_info.x),
		cugar::from_rgbe(packed_info.y),
		roughness,
		cugar::from_rgbe(packed_info.w),
		opacity,
		ior);
}

///\par
/// Unpack the diffuse component of a uint4-packed BSDF
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f unpack_bsdf_diffuse(const uint4 packed_info) { return cugar::from_rgbe(packed_info.x); }

///\par
/// Unpack the specular component of a uint4-packed BSDF
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f unpack_bsdf_specular(const uint4 packed_info) { return cugar::from_rgbe(packed_info.y); }

///\par
/// Unpack the roughness component of a uint4-packed BSDF
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float unpack_bsdf_roughness(const uint4 packed_info) { return cugar::binary_cast<float>(packed_info.z); }

///\par
/// Unpack the diffuse transmission component of a uint4-packed BSDF
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f unpack_bsdf_diffuse_trans(const uint4 packed_info) { return cugar::from_rgbe(packed_info.w); }

///\par
/// Return the normal delta due to bump-mapping
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f bump_mapping(const VertexGeometryId& geom_id, const VertexGeometry& geom, const TextureReference& bump_map, const RenderingContextView& renderer)
{
	// 1. lookup dh_dst using the bump map
	const cugar::Vector2f dh_dst = diff_texture_lookup(geom.texture_coords, bump_map, renderer.textures, cugar::Vector2f(0.0f));

	if (dh_dst != cugar::Vector2f(0.0f))
	{
		// 2. compute dp_dst
		const cugar::Matrix3x2f dp_dst = prim_dp_dst<kTextureCoords0>( renderer.mesh, geom_id.prim_id );

		cugar::Vector3f dp_ds( dp_dst[0][0], dp_dst[1][0], dp_dst[2][0] );
		cugar::Vector3f dp_dt( dp_dst[0][1], dp_dst[1][1], dp_dst[2][1] );

		// 3. project dp_ds and dp_dt on the plane formed by the local interpolated normal
		dp_ds = dp_ds - geom.normal_s * dot(dp_ds, geom.normal_s);
		dp_dt = dp_dt - geom.normal_s * dot(dp_dt, geom.normal_s);

		// 4. recompute the new normal as: N + dh_dt * (dp_ds x N) + dh_ds * (dp_dt x N)
		return dh_dst.y * cugar::cross(dp_ds, geom.normal_s) + dh_dst.x * cugar::cross(dp_dt, geom.normal_s);
	}
	return cugar::Vector3f(0.0f);
}

///\par
/// A utility class to encapsulate all important information for a light subpath vertex
///
struct LightVertex
{
	FERMAT_HOST_DEVICE
	void setup(
		const cugar::Vector4f&			_pos,
		const uint2&					_packed_info1,
		const uint4&					_packed_info2,
		const PathWeights&				_weights,
		const uint32					_depth,
		const RenderingContextView&		renderer)
	{
		in		= unpack_direction(_packed_info1.x);
		alpha	= cugar::from_rgbe(_packed_info1.y);
		weights = _weights;
		depth	= _depth;

		geom.position = _pos.xyz();
		geom.normal_s = unpack_direction(cugar::binary_cast<uint32>(_pos.w));
		geom.normal_g = geom.normal_s;
		geom.tangent  = cugar::orthogonal(geom.normal_s);
		geom.binormal = cugar::cross(geom.normal_s, geom.tangent);

		if (depth == 0)
			edf = unpack_edf(_packed_info2);
		else
			bsdf = unpack_bsdf(renderer, _packed_info2);
	}

	FERMAT_HOST_DEVICE
	void setup(
		const Ray&					ray,
		const Hit&					hit,
		const cugar::Vector3f&		_alpha,
		const PathWeights&			_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		geom_id.prim_id = hit.triId;
		geom_id.uv		= cugar::Vector2f(hit.u, hit.v);

		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;
		//assert(depth);

		FERMAT_ASSERT(hit.triId < renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, hit.triId, hit.u, hit.v, &geom);

	#if !defined(BARYCENTRIC_HIT_POINT)
		// reset the position using the ray
		geom.position = cugar::Vector3f( ray.origin ) + hit.t * cugar::Vector3f( ray.dir );
	#endif

		// fetch the material
		const int material_id = renderer.mesh.material_indices[hit.triId];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= bilinear_texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= bilinear_texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

	  #if 0
		// perform bump-mapping
		geom.normal_s += 0.05f * bump_mapping( geom_id, geom, material.bump_map, renderer );
		geom.normal_s = cugar::normalize( geom.normal_s );
	  #endif

		in = -cugar::normalize(cugar::Vector3f(ray.dir));

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		bsdf = Bsdf(kParticleTransport, renderer, material);
	}

	FERMAT_HOST_DEVICE
	void setup(
		const Ray&					ray,
		const Hit&					hit,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		setup(ray, hit, _alpha, PathWeights(_weights), _depth, renderer);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / fmaxf(hit.t * hit.t, MIN_G_DENOM);
		
		prev_pG = pdf_product( _weights.out_p, _weights.out_cos_theta * prev_G_prime );
		pGp_sum = _weights.pGp_sum + mis_power(1 / pdf_product(_weights.pG, _weights.out_p));
	}

	FERMAT_HOST_DEVICE
	void setup(
		const cugar::Vector3f&		_in,
		const VertexGeometryId&		_v,
		const cugar::Vector3f&		_alpha,
		const PathWeights&			_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		geom_id = _v;

		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;
		//assert(depth);

		FERMAT_ASSERT(_v.prim_id < (uint32)renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, _v, &geom);

		// fetch the material
		const int material_id = renderer.mesh.material_indices[_v.prim_id];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_map, renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= bilinear_texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= bilinear_texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

	  #if 0
		// perform bump-mapping
		geom.normal_s += 0.05f * bump_mapping( geom_id, geom, material.bump_map, renderer );
		geom.normal_s = cugar::normalize( geom.normal_s );
	  #endif

		in = cugar::normalize( _in );

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		bsdf = Bsdf(kParticleTransport, renderer, material);
	}

	FERMAT_HOST_DEVICE
	void setup(
		const cugar::Vector3f&		_in,
		const VertexGeometryId&		_v,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		geom_id = _v;

		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;
		//assert(depth);

		FERMAT_ASSERT(_v.prim_id < (uint32)renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, _v, &geom);

		// fetch the material
		const int material_id = renderer.mesh.material_indices[_v.prim_id];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_map, renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= bilinear_texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= bilinear_texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

	  #if 0
		// perform bump-mapping
		geom.normal_s += 0.05f * bump_mapping( geom_id, geom, material.bump_map, renderer );
		geom.normal_s = cugar::normalize( geom.normal_s );
	  #endif

		in = cugar::normalize( _in );

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		bsdf = Bsdf(kParticleTransport, renderer, material);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / fmaxf(cugar::square_length(_in), MIN_G_DENOM);
		
		prev_pG = pdf_product( _weights.out_p, _weights.out_cos_theta * prev_G_prime );
		pGp_sum = _weights.pGp_sum + mis_power(1 / pdf_product(_weights.pG, _weights.out_p));
	}

	FERMAT_HOST_DEVICE
	LightVertex() {}

	FERMAT_HOST_DEVICE
	LightVertex(
		const cugar::Vector4f&			_pos,
		const uint2&					_packed_info1,
		const uint4&					_packed_info2,
		const PathWeights&				_weights,
		const uint32					_depth,
		const RenderingContextView&		renderer)
	{
		setup( _pos, _packed_info1, _packed_info2, _weights, _depth, renderer );
	}

	FERMAT_HOST_DEVICE
	LightVertex(
		const Ray&					_ray,
		const Hit&					_hit,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		setup( _ray, _hit, _alpha, _weights, _depth, renderer );
	}

	FERMAT_HOST_DEVICE
	LightVertex(
		const cugar::Vector3f&		_in,
		const VertexGeometryId&		_v,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		setup( _in, _v, _alpha, _weights, _depth, renderer );
	}

	/// sample an outgoing direction
	///
	/// \param z					the incoming direction
	/// \param out_comp				the output component
	/// \param out					the outgoing direction
	/// \param out_p				the output solid angle pdf
	/// \param out_p_proj			the output projected solid angle pdf
	/// \param out_g				the output sample value = f/p_proj
	/// \param RR					indicate whether to use Russian-Roulette or not
	/// \param evaluate_full_bsdf	ndicate whether to evaluate the full BSDF, or just an unbiased estimate
	/// \param components			the components to consider
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool sample(
		const float							z[3],
		Bsdf::ComponentType&				out_comp,
		cugar::Vector3f&					out,
		float&								out_p,
		float&								out_p_proj,
		cugar::Vector3f&					out_g,
		bool								RR					= true,
		bool								evaluate_full_bsdf	= false,
		const Bsdf::ComponentType			components			= Bsdf::kAllComponents) const
	{
		return bsdf.sample( geom, z, in, out_comp, out, out_p, out_p_proj, out_g, RR, evaluate_full_bsdf, components );
	}

	VertexGeometryId geom_id;
	VertexGeometry  geom;
	cugar::Vector3f in;
	cugar::Vector3f alpha;
	PathWeights     weights;
	uint32			depth;
	MeshMaterial	material;
	Edf				edf;
	Bsdf			bsdf;
	float			prev_G_prime;
	float			prev_pG;
	float			pGp_sum;
};

///\par
/// A utility class to encapsulate all important information for a eye subpath vertex
///
struct EyeVertex
{
	template <typename RayType>
	FERMAT_HOST_DEVICE
	void setup(
		const RayType&				ray,
		const Hit&					hit,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer,
		const float					min_roughness = 0.0f)
	{
		geom_id.prim_id = hit.triId;
		geom_id.uv		= cugar::Vector2f(hit.u, hit.v);

		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;

		FERMAT_ASSERT(hit.triId < renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, hit.triId, hit.u, hit.v, &geom);

	#if !defined(BARYCENTRIC_HIT_POINT)
		// reset the position using the ray
		geom.position = cugar::Vector3f( ray.origin ) + hit.t * cugar::Vector3f( ray.dir );
	#endif

		// fetch the material
		const int material_id = renderer.mesh.material_indices[hit.triId];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= bilinear_texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= bilinear_texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

	  #if 0
		// perform bump-mapping
		geom.normal_s += 0.05f * bump_mapping( geom_id, geom, material.bump_map, renderer );
		geom.normal_s = cugar::normalize( geom.normal_s );
	  #endif

		in = -cugar::normalize(cugar::Vector3f(ray.dir));

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		const float mollification_factor = 1.0f; // depth ? float(1u << (4 + depth)) : 1.0f; // at the first bounce, we increase the roughness by a factor 32, at the second 64, and so on...
		const float mollification_bias   = 0.0f; // depth ? 0.05f : 0.0f;
		bsdf = Bsdf(kRadianceTransport, renderer, material, mollification_factor, mollification_bias, min_roughness);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / (hit.t * hit.t); // the G' of the incoming edge

		prev_pG = pdf_product( weights.out_p, weights.out_cos_theta * prev_G_prime );
		pGp_sum = weights.pGp_sum + mis_power(1 / pdf_product(weights.pG, weights.out_p));
	}

	FERMAT_HOST_DEVICE
	void setup(
		const cugar::Vector3f&		_in,
		const VertexGeometryId&		_v,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		geom_id = _v;
		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;
		//assert(depth);

		FERMAT_ASSERT(_v.prim_id < uint32(renderer.mesh.num_triangles));
		setup_differential_geometry(renderer.mesh, _v, &geom);

		// fetch the material
		const int material_id = renderer.mesh.material_indices[_v.prim_id];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= bilinear_texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= bilinear_texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= bilinear_texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

	  #if 0
		// perform bump-mapping
		geom.normal_s += 0.05f * bump_mapping( geom_id, geom, material.bump_map, renderer );
		geom.normal_s = cugar::normalize( geom.normal_s );
	  #endif

		in = cugar::normalize( _in );

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		bsdf = Bsdf(kRadianceTransport, renderer, material);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / fmaxf(cugar::square_length(_in), MIN_G_DENOM);
		
		prev_pG = pdf_product( weights.out_p, weights.out_cos_theta * prev_G_prime );
		pGp_sum = weights.pGp_sum + mis_power(1 / pdf_product(weights.pG, weights.out_p));
	}

	FERMAT_HOST_DEVICE
	EyeVertex() {}

	FERMAT_HOST_DEVICE
	EyeVertex(
		const Ray&					_ray,
		const Hit&					_hit,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		setup( _ray, _hit, _alpha, _weights, _depth, renderer );
	}

	FERMAT_HOST_DEVICE
	EyeVertex(
		const cugar::Vector3f&		_in,
		const VertexGeometryId&		_v,
		const cugar::Vector3f&		_alpha,
		const TempPathWeights&		_weights,
		const uint32				_depth,
		const RenderingContextView&	renderer)
	{
		setup( _in, _v, _alpha, _weights, _depth, renderer );
	}

	/// sample an outgoing direction
	///
	/// \param z					the incoming direction
	/// \param out_comp				the output component
	/// \param out					the outgoing direction
	/// \param out_p				the output solid angle pdf
	/// \param out_p_proj			the output projected solid angle pdf
	/// \param out_g				the output sample value = f/p_proj
	/// \param RR					indicate whether to use Russian-Roulette or not
	/// \param evaluate_full_bsdf	ndicate whether to evaluate the full BSDF, or just an unbiased estimate
	/// \param components			the components to consider
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool sample(
		const float							z[3],
		Bsdf::ComponentType&				out_comp,
		cugar::Vector3f&					out,
		float&								out_p,
		float&								out_p_proj,
		cugar::Vector3f&					out_g,
		bool								RR					= true,
		bool								evaluate_full_bsdf	= false,
		const Bsdf::ComponentType			components			= Bsdf::kAllComponents) const
	{
		return bsdf.sample( geom, z, in, out_comp, out, out_p, out_p_proj, out_g, RR, evaluate_full_bsdf, components );
	}
		
	VertexGeometryId	geom_id;
	VertexGeometry		geom;
	cugar::Vector3f		in;
	cugar::Vector3f		alpha;
	TempPathWeights		weights;
	uint32				depth;
	MeshMaterial		material;
	Bsdf				bsdf;
	float				prev_G_prime;
	float				prev_pG;
	float				pGp_sum;
};

///\par
/// Evaluate all the separate terms needed for a bidirectional connection, including
///\n
/// - the edge vector joining the eye and light vertex
/// - the product of the BSDFs at the two vertices
/// - the geometric throughput G
/// - the distance between the vertices
///
FERMAT_HOST_DEVICE inline
void eval_connection_terms(const EyeVertex ev, const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& f_conn, float& G, float& d)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(MIN_G_DENOM, cugar::square_length(lv.geom.position - ev.geom.position));

	d = sqrtf(d2);

	// join the light sample with the current vertex
	out = (lv.geom.position - ev.geom.position) / d;

	// evaluate the geometric term
	G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, lv.geom.normal_s)) / d2;

	// evaluate the surface BSDF
	const cugar::Vector3f f_s = ev.bsdf.f(ev.geom, ev.in, out);

	if (lv.depth == 0) // this is a primary VPL / light vertex
	{
		// build the local BSDF (EDF)
		const Edf& light_bsdf = lv.edf;

		// evaluate the light's EDF and the surface BSDF
		const cugar::Vector3f f_L = light_bsdf.f(lv.geom, lv.geom.position, -out);

		f_conn = f_L * f_s;
	}
	else
	{
		// build the local BSDF
		const Bsdf& light_bsdf = lv.bsdf;

		// evaluate the light's EDF and the surface BSDF
		const cugar::Vector3f f_L = light_bsdf.f(lv.geom, lv.in, -out);

		f_conn = f_L * f_s;
	}
}

///\par
/// Evaluate all the separate terms needed for a bidirectional connection, including
///\n
/// - the edge vector joining the eye and light vertex
/// - the product of the BSDFs at the two vertices
/// - the geometric throughput G
/// - the distance between the vertices
/// - the MIS weight
///
FERMAT_HOST_DEVICE inline
void eval_connection_terms(const EyeVertex ev, const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& f_conn, float& G, float& d, float& mis_w,
	bool RR						= true,
	bool direct_lighting_nee	= true,
	bool direct_lighting_bsdf	= true)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(MIN_G_DENOM, cugar::square_length(lv.geom.position - ev.geom.position));

	d = sqrtf(d2);

	// join the light sample with the current vertex
	out = (lv.geom.position - ev.geom.position) / d;

	// evaluate the geometric term
	G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, lv.geom.normal_s)) / d2;

	// evaluate the surface BSDF
	cugar::Vector3f f_s;
	float			p_s;
	//f_s = ev.bsdf.f(ev.geom, ev.in, out);
	//p_s = ev.bsdf.p(ev.geom, ev.in, out, cugar::kProjectedSolidAngle, RR);
	ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s, p_s, cugar::kProjectedSolidAngle, RR);

	const float prev_pGp = pdf_product( ev.prev_pG, p_s );

	if (lv.depth == 0) // this is a primary VPL / light vertex
	{
		// build the local BSDF (EDF)
		const Edf& light_bsdf = lv.edf;

		// evaluate the light's EDF and the surface BSDF
		const cugar::Vector3f f_L = light_bsdf.f(lv.geom, lv.geom.position, -out);
		const float			  p_L = light_bsdf.p(lv.geom, lv.geom.position, -out, cugar::kProjectedSolidAngle);

		const float pGp      = pdf_product( p_s, G, p_L ); // infinity checks are not really needed here... but just in case
		const float next_pGp = pdf_product( p_L, lv.weights.pG );
		mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				(ev.depth == 0 && direct_lighting_bsdf == false) ? 1.0f :
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));

		f_conn = f_L * f_s;
	}
	else
	{
		// build the local BSDF
		const Bsdf& light_bsdf = lv.bsdf;

		// evaluate the light's EDF and the surface BSDF
		cugar::Vector3f f_L;
		float			p_L;
		//f_L = light_bsdf.f(lv.geom, lv.in, -out);
		//p_L = light_bsdf.p(lv.geom, lv.in, -out, RR, cugar::kProjectedSolidAngle);
		light_bsdf.f_and_p(lv.geom, lv.in, -out, f_L, p_L, cugar::kProjectedSolidAngle, RR);

		const float pGp = pdf_product( p_s, G, p_L ); // infinity checks are not really needed here... but just in case
		const float next_pGp = pdf_product( p_L, lv.weights.pG );
		mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));

		f_conn = f_L * f_s;
	}
}

///\par
/// Evaluate all the separate terms needed for a bidirectional connection, including
///\n
/// - the edge vector joining the eye and light vertex
/// - the product of the BSDFs at the two vertices
/// - the geometric throughput G
/// - the distance between the vertices
/// - the MIS weight
///
FERMAT_HOST_DEVICE inline
void eval_connection_terms(const EyeVertex ev, const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& f_s, cugar::Vector3f& f_L, float& G, float& d, float& mis_w,
	bool RR						= true,
	bool direct_lighting_nee	= true,
	bool direct_lighting_bsdf	= true)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(MIN_G_DENOM, cugar::square_length(lv.geom.position - ev.geom.position));

	d = sqrtf(d2);

	// join the light sample with the current vertex
	out = (lv.geom.position - ev.geom.position) / d;

	// evaluate the geometric term
	G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, lv.geom.normal_s)) / d2;

	// evaluate the surface BSDF
	float p_s;
	//f_s = ev.bsdf.f(ev.geom, ev.in, out);
	//p_s = ev.bsdf.p(ev.geom, ev.in, out, cugar::kProjectedSolidAngle, RR);
	ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s, p_s, cugar::kProjectedSolidAngle, RR);

	const float prev_pGp = pdf_product( ev.prev_pG, p_s );

	if (lv.depth == 0) // this is a primary VPL / light vertex
	{
		// build the local BSDF (EDF)
		const Edf& light_bsdf = lv.edf;

		// evaluate the light's EDF and the surface BSDF
					f_L = light_bsdf.f(lv.geom, lv.geom.position, -out);
		const float p_L = light_bsdf.p(lv.geom, lv.geom.position, -out, cugar::kProjectedSolidAngle);

		const float pGp      = pdf_product( p_s, G, p_L );
		const float next_pGp = pdf_product( p_L, lv.weights.pG );
		mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				(ev.depth == 0 && direct_lighting_bsdf == false) ? 1.0f :
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));
	}
	else
	{
		// build the local BSDF
		const Bsdf& light_bsdf = lv.bsdf;

		// evaluate the light's EDF and the surface BSDF
		float p_L;
		//f_L = light_bsdf.f(lv.geom, lv.in, -out);
		//p_L = light_bsdf.p(lv.geom, lv.in, -out, RR, cugar::kProjectedSolidAngle);
		light_bsdf.f_and_p(lv.geom, lv.in, -out, f_L, p_L, cugar::kProjectedSolidAngle, RR);

		const float pGp      = pdf_product( p_s, G, p_L );
		const float next_pGp = pdf_product( p_L, lv.weights.pG );
		mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));
	}
}

FERMAT_HOST_DEVICE inline
void eval_connection(
	const EyeVertex ev,	const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& out_w, float& d,
	bool RR						= true,
	bool direct_lighting_nee	= true,
	bool direct_lighting_bsdf	= true)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(MIN_G_DENOM, cugar::square_length(lv.geom.position - ev.geom.position));

	d = sqrtf(d2);

	// join the light sample with the current vertex
	out = (lv.geom.position - ev.geom.position) / d;

	// evaluate the geometric term
	const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, lv.geom.normal_s)) / d2;

	// evaluate the surface BSDF
	cugar::Vector3f f_s;
	float			p_s;
	//f_s = ev.bsdf.f(ev.geom, ev.in, out);
	//p_s = ev.bsdf.p(ev.geom, ev.in, out, cugar::kProjectedSolidAngle, RR);
	ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s, p_s, cugar::kProjectedSolidAngle, RR);

	const float prev_pGp = pdf_product( ev.prev_pG, p_s );

	if (lv.depth == 0) // this is a primary VPL / light vertex
	{
		if (direct_lighting_nee == false)
			out_w = cugar::Vector3f(0.0f);
		else
		{
			// build the local BSDF (EDF)
			const Edf& light_bsdf = lv.edf;

			// evaluate the light's EDF and the surface BSDF
			const cugar::Vector3f f_L = light_bsdf.f(lv.geom, lv.geom.position, -out);
			const float			  p_L = light_bsdf.p(lv.geom, lv.geom.position, -out, cugar::kProjectedSolidAngle);

			const float pGp      = pdf_product( p_s, G, p_L );
			const float next_pGp = pdf_product( p_L, lv.weights.pG );
			const float mis_w =
				mis_selector(
					lv.depth + 1, ev.depth + 2,
					(ev.depth == 0 && direct_lighting_bsdf == false) ? 1.0f :
					bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));

			// calculate the cumulative sample weight, equal to f_L * f_s * G / p
			out_w = ev.alpha * lv.alpha * f_L * f_s * G * mis_w;
		}
	}
	else
	{
		// build the local BSDF
		const Bsdf& light_bsdf = lv.bsdf;

		// evaluate the light's EDF and the surface BSDF
		cugar::Vector3f f_L;
		float			p_L;
		//f_L = light_bsdf.f(lv.geom, lv.in, -out);
		//p_L = light_bsdf.p(lv.geom, lv.in, -out, RR, cugar::kProjectedSolidAngle);
		light_bsdf.f_and_p(lv.geom, lv.in, -out, f_L, p_L, cugar::kProjectedSolidAngle, RR);

		const float pGp      = pdf_product( p_s, G, p_L );
		const float next_pGp = pdf_product( p_L, lv.weights.pG );
		const float mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));

		// calculate the cumulative sample weight, equal to f_L * f_s * G / p
		out_w = ev.alpha * lv.alpha * f_L * f_s * G * mis_w;
	}
}

///\par
/// Evaluate the surface emission in the incoming direction at a eye vertex
///
FERMAT_HOST_DEVICE inline
cugar::Vector3f eval_incoming_emission(
	const EyeVertex& ev, const RenderingContextView& renderer,
	bool direct_lighting_nee,
	bool indirect_lighting_nee,
	bool use_vpls)
{
	VertexGeometry	light_vertex_geom = ev.geom;
	float			light_pdf;
	Edf				light_edf;

	if (use_vpls)
		renderer.mesh_vpls.map(ev.geom_id.prim_id, ev.geom_id.uv, light_vertex_geom, &light_pdf, &light_edf);
	else
		renderer.mesh_light.map(ev.geom_id.prim_id, ev.geom_id.uv, light_vertex_geom, &light_pdf, &light_edf);

	// evaluate the edf's output along the incoming direction
	const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, ev.in);
	const float		      p_L = light_edf.p(light_vertex_geom, light_vertex_geom.position, ev.in, cugar::kProjectedSolidAngle);

	const float pGp      = pdf_product( p_L, light_pdf );
	const float prev_pGp = pdf_product( ev.prev_pG, p_L );
	const float mis_w =
		mis_selector(
			0, ev.depth + 2,
			(ev.depth == 0 || pGp == 0.0f || (ev.depth == 1 && direct_lighting_nee == false) || (ev.depth > 1 && indirect_lighting_nee == false)) ? 1.0f :
			bpt_mis(pGp, prev_pGp, ev.pGp_sum));

	// and accumulate the weighted contribution
	return ev.alpha * f_L * mis_w;
}

///\par
/// sample a scattering/absorption event at a given vertex
///
template <typename VertexType>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool scatter(
	const VertexType&		v,
	const float				z[3],
	Bsdf::ComponentType&	out_component,
	cugar::Vector3f&		out,
	float&					out_p,
	float&					out_p_proj,
	cugar::Vector3f&		out_w,
	bool					RR					= true,
	bool					output_alpha		= true,
	bool					evaluate_full_bsdf	= false,
	Bsdf::ComponentType		components			= Bsdf::kAllComponents)
{
	bool scattered = v.bsdf.sample(
		v.geom,
		z,
		v.in,
		out_component,
		out,
		out_p,
		out_p_proj,
		out_w,
		RR,
		evaluate_full_bsdf,
		components);

	if (output_alpha)
		out_w *= v.alpha;

	return scattered;
}

///@} BPTLibCore
///@} BPTLib
///@} Fermat
