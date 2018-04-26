/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#define DEBUG_S				-1
#define DEBUG_T				-1
#define DEBUG_LENGTH		0

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
uint32 channel_selector(const Bsdf::ComponentType comp)
{
	return (comp & Bsdf::kDiffuseMask) ? FBufferDesc::DIFFUSE_C : FBufferDesc::SPECULAR_C;
}

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
struct TempPathWeights
{
	FERMAT_HOST_DEVICE TempPathWeights() {}
	FERMAT_HOST_DEVICE TempPathWeights(const float _pGp_sum, const float _pG, const float _out_p, const float _out_cos_theta) :
		pGp_sum(_pGp_sum), pG(_pG), out_p(_out_p), out_cos_theta(_out_cos_theta) {}
	FERMAT_HOST_DEVICE TempPathWeights(const float4 f) : vector_storage(f) {}

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

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_edf(const Edf& edf)
{
	return make_uint4(
		cugar::to_rgbe(edf.color),
		0,
		0,
		0);
}
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_edf(const MeshMaterial& material)
{
	return make_uint4(
		cugar::to_rgbe(cugar::Vector4f(material.emissive).xyz()),
		0,
		0,
		0);
}
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint4 pack_bsdf(const MeshMaterial& material)
{
	return make_uint4(
		cugar::to_rgbe(cugar::Vector4f(material.diffuse).xyz()),
		cugar::to_rgbe(cugar::Vector4f(material.specular).xyz()),
		cugar::binary_cast<uint32>(material.roughness),
		cugar::to_rgbe(cugar::Vector4f(material.diffuse_trans).xyz()));
}
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Edf unpack_edf(const uint4 packed_info)
{
	return Edf(cugar::from_rgbe(packed_info.x));
}
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Bsdf unpack_bsdf(const RendererView& renderer, const uint4 packed_info, const TransportType transport = kParticleTransport)
{
	return Bsdf(
		transport,
		renderer,
		cugar::from_rgbe(packed_info.x),
		cugar::from_rgbe(packed_info.y),
		cugar::binary_cast<float>(packed_info.z),
		cugar::from_rgbe(packed_info.w));
}

struct LightVertex
{
	FERMAT_HOST_DEVICE
	void setup(
		const cugar::Vector4f&	_pos,
		const uint2&			_packed_info1,
		const uint4&			_packed_info2,
		const PathWeights&		_weights,
		const uint32			_depth,
		const RendererView&		renderer)
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
		const Ray&				ray,
		const Hit&				hit,
		const cugar::Vector3f&	_alpha,
		const PathWeights&		_weights,
		const uint32			_depth,
		const RendererView&		renderer)
	{
		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;
		//assert(depth);

		FERMAT_ASSERT(hit.triId < renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, hit.triId, hit.u, hit.v, &geom);

		// fetch the material
		const int material_id = renderer.mesh.material_indices[hit.triId];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map, renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

		in = -cugar::normalize(cugar::Vector3f(ray.dir));

		if (dot(geom.normal_s, in) < 0.0f)
			geom.normal_s = -geom.normal_s;

		bsdf = Bsdf(kParticleTransport, renderer, material);
	}

	FERMAT_HOST_DEVICE
	void setup(
		const Ray&				ray,
		const Hit&				hit,
		const cugar::Vector3f&	_alpha,
		const TempPathWeights&	_weights,
		const uint32			_depth,
		const RendererView&		renderer)
	{
		setup(ray, hit, _alpha, PathWeights(_weights), _depth, renderer);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / fmaxf(hit.t * hit.t, 1.0e-8f);
		
		prev_pG = _weights.out_p * _weights.out_cos_theta * prev_G_prime;
		pGp_sum = _weights.pGp_sum + mis_power(1 / (_weights.pG * _weights.out_p));
	}

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

struct EyeVertex
{
	FERMAT_HOST_DEVICE
	void setup(
		const Ray&				ray,
		const Hit&				hit,
		const cugar::Vector3f&	_alpha,
		const TempPathWeights&	_weights,
		const uint32			_depth,
		const RendererView&		renderer)
	{
		geom_id.prim_id = hit.triId;
		geom_id.uv		= cugar::Vector2f(hit.u, hit.v);

		alpha	= _alpha;
		weights = _weights;
		depth	= _depth;

		FERMAT_ASSERT(hit.triId < renderer.mesh.num_triangles);
		setup_differential_geometry(renderer.mesh, hit.triId, hit.u, hit.v, &geom);

		// fetch the material
		const int material_id = renderer.mesh.material_indices[hit.triId];

		material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= texture_lookup(geom.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map, renderer.textures, cugar::Vector4f(1.0f));

		in = -cugar::normalize(cugar::Vector3f(ray.dir));

		//if (dot(geom.normal_s, in) < 0.0f)
		//	geom.normal_s = -geom.normal_s;

		const float mollification_factor = 1.0f; // depth ? float(1u << (4 + depth)) : 1.0f; // at the first bounce, we increase the roughness by a factor 32, at the second 64, and so on...
		const float mollification_bias   = 0.0f; // depth ? 0.05f : 0.0f;
		bsdf = Bsdf(kRadianceTransport, renderer, material, mollification_factor, mollification_bias);

		// compute the MIS terms for the next iteration
		prev_G_prime = fabsf(dot(in, geom.normal_s)) / (hit.t * hit.t); // the G' of the incoming edge

		prev_pG = weights.out_p * weights.out_cos_theta * prev_G_prime;
		pGp_sum = weights.pGp_sum + mis_power(1 / (weights.pG * weights.out_p));
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

FERMAT_HOST_DEVICE inline
void eval_connection_terms(const EyeVertex ev, const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& f_conn, float& G, float& d)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(1.0e-8f, cugar::square_length(lv.geom.position - ev.geom.position));

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

FERMAT_HOST_DEVICE inline
void eval_connection(
	const EyeVertex ev,	const LightVertex& lv, cugar::Vector3f& out, cugar::Vector3f& out_w, float& d,
	bool RR						= true,
	bool direct_lighting_nee	= true,
	bool direct_lighting_bsdf	= true)
{
	// start evaluating the geometric term
	const float d2 = fmaxf(1.0e-8f, cugar::square_length(lv.geom.position - ev.geom.position));

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

	const float prev_pGp = ev.prev_pG * p_s;

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

			const float pGp = p_s * G * p_L;
			const float next_pGp = p_L * lv.weights.pG;
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

		const float pGp = p_s * G * p_L;
		const float next_pGp = p_L * lv.weights.pG;
		const float mis_w =
			mis_selector(
				lv.depth + 1, ev.depth + 2,
				bpt_mis(pGp, prev_pGp, next_pGp, ev.pGp_sum + lv.weights.pGp_sum));

		// calculate the cumulative sample weight, equal to f_L * f_s * G / p
		out_w = ev.alpha * lv.alpha * f_L * f_s * G * mis_w;
	}
}

FERMAT_HOST_DEVICE inline
cugar::Vector3f eval_incoming_emission(
	const EyeVertex& ev, const RendererView& renderer,
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

	const float pGp = p_L * light_pdf;
	const float prev_pGp = ev.prev_pG * p_L;
	const float mis_w =
		mis_selector(
			0, ev.depth + 2,
			(ev.depth == 0 || pGp == 0.0f || (ev.depth == 1 && direct_lighting_nee == false) || (ev.depth > 1 && indirect_lighting_nee == false)) ? 1.0f :
			bpt_mis(pGp, prev_pGp, ev.pGp_sum));

	// and accumulate the weighted contribution
	return ev.alpha * f_L * mis_w;
}

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
	bool					evaluate_full_bsdf	= false)
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
		evaluate_full_bsdf);

	if (output_alpha)
		out_w *= v.alpha;

	return scattered;
}

///@} BPTLib
///@} Fermat
