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

// ------------------------------------------------------------------------- //
//
// Declaration of classes used to represent and manipulate paths.
//
// ------------------------------------------------------------------------- //

#include <path.h>
#include <bsdf.h>
#include <camera.h>
#include <lights.h>
#include <renderer.h>
#include <mesh_utils.h>

///@addtogroup Fermat
///@{

///@addtogroup PathModule
///@{

#define MAXIMUM_INVERSION_ERROR 1.0e-6f

#define TRANSFORM_PDF(_p, _w, _w_norm) (_p *= _w_norm / _w)
//#define TRANSFORM_PDF(_p, _w, _w_norm) (_p *= _w / _w_norm)
//#define TRANSFORM_PDF(_p, _w, _w_norm) (_p *= _w)
//#define TRANSFORM_PDF(_p, _w, _w_norm) (_p /= _w)

template <typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_diffuse(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	TRandomGenerator&		random,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// invert the diffuse component
	if (bsdf.diffuse().invert(
		v_prev,
		in,
		out,
		random,
		z,
		p,
		p_proj) == false)
		return false;

	if (1)
	{
		cugar::Vector3f out_;
		cugar::Vector3f g_;
		float			p_, p_proj_;

		bsdf.diffuse().sample(
			z,
			v_prev,
			in,
			out_,
			g_,
			p_,
			p_proj_);

		//const float err = cugar::max3(
		//	fabsf(out_.x - out.x),
		//	fabsf(out_.y - out.y),
		//	fabsf(out_.z - out.z));
		const float err = 1.0f - dot(out_, out);
		if (err > MAXIMUM_INVERSION_ERROR)
		{
			/*printf("diffuse err %f\n  N: %f %f %f\n  I: %f %f %f\n, O: %f %f %f\n  O: %f %f %f\n",
			err,
			v_prev.normal_s.x, v_prev.normal_s.y, v_prev.normal_s.z,
			in.x, in.y, in.z,
			out.x, out.y, out.z,
			out_.x, out_.y, out_.z);*/
			return false;
		}
	}
	return true;
}

template <typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_diffuse_trans(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	TRandomGenerator&		random,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// invert the diffuse component
	if (bsdf.diffuse_trans().invert(
		v_prev,
		in,
		out,
		random,
		z,
		p,
		p_proj) == false)
		return false;

	if (1)
	{
		cugar::Vector3f out_;
		cugar::Vector3f g_;
		float			p_, p_proj_;

		bsdf.diffuse_trans().sample(
			z,
			v_prev,
			in,
			out_,
			g_,
			p_,
			p_proj_);

		//const float err = cugar::max3(
		//	fabsf(out_.x - out.x),
		//	fabsf(out_.y - out.y),
		//	fabsf(out_.z - out.z));
		const float err = 1.0f - dot(out_, out);
		if (err > MAXIMUM_INVERSION_ERROR)
		{
			/*printf("diffuse err %f\n  N: %f %f %f\n  I: %f %f %f\n, O: %f %f %f\n  O: %f %f %f\n",
			err,
			v_prev.normal_s.x, v_prev.normal_s.y, v_prev.normal_s.z,
			in.x, in.y, in.z,
			out.x, out.y, out.z,
			out_.x, out_.y, out_.z);*/
			return false;
		}
	}
	return true;
}

template <typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_glossy(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	TRandomGenerator&		random,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// invert the diffuse component
	if (bsdf.glossy().invert(
		v_prev,
		in,
		out,
		random,
		z,
		p,
		p_proj) == false)
		return false;

	if (1)
	{
		cugar::Vector3f out_;
		cugar::Vector3f g_;
		float			p_, p_proj_;

		bsdf.glossy().sample(
			z,
			v_prev,
			in,
			out_,
			g_,
			p_,
			p_proj_);

		//const float err = cugar::max3(
		//	fabsf(out_.x - out.x),
		//	fabsf(out_.y - out.y),
		//	fabsf(out_.z - out.z));
		const float err = 1.0f - dot(out_, out);
		if (err > MAXIMUM_INVERSION_ERROR)
		{
			/*printf("glossy err %f\n  N: %f %f %f\n  I: %f %f %f\n, O: %f %f %f\n  O: %f %f %f\n",
			err,
			v_prev.normal_s.x, v_prev.normal_s.y, v_prev.normal_s.z,
			in.x, in.y, in.z,
			out.x, out.y, out.z,
			out_.x, out_.y, out_.z);*/
			return false;
		}
	}
	return true;
}

template <typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_glossy_trans(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	TRandomGenerator&		random,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// invert the diffuse component
	if (bsdf.glossy_trans().invert(
		v_prev,
		in,
		out,
		random,
		z,
		p,
		p_proj) == false)
		return false;

	if (1)
	{
		cugar::Vector3f out_;
		cugar::Vector3f g_;
		float			p_, p_proj_;

		bsdf.glossy_trans().sample(
			z,
			v_prev,
			in,
			out_,
			g_,
			p_,
			p_proj_);

		//const float err = cugar::max3(
		//	fabsf(out_.x - out.x),
		//	fabsf(out_.y - out.y),
		//	fabsf(out_.z - out.z));
		const float err = 1.0f - dot(out_, out);
		if (err > MAXIMUM_INVERSION_ERROR)
		{
			/*printf("glossy err %f\n  N: %f %f %f\n  I: %f %f %f\n, O: %f %f %f\n  O: %f %f %f\n",
			err,
			v_prev.normal_s.x, v_prev.normal_s.y, v_prev.normal_s.z,
			in.x, in.y, in.z,
			out.x, out.y, out.z,
			out_.x, out_.y, out_.z);*/
			return false;
		}
	}
	return true;
}

template <typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_bsdf(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	TRandomGenerator&		random,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// choose the BSDF component utilized for inversion
	float diffuse_refl_prob;
	float diffuse_trans_prob;
	float glossy_refl_prob;
	float glossy_trans_prob;

	bsdf.sampling_weights(v_prev, in, diffuse_refl_prob, diffuse_trans_prob, glossy_refl_prob, glossy_trans_prob);

	const float w1 = diffuse_refl_prob;
	const float w2 = glossy_refl_prob;
	const float w3 = diffuse_trans_prob;
	const float w4 = glossy_trans_prob;
	const float w_sum = w1 + w2 + w3 + w4;

	const float w1_norm = w1 / w_sum;
	const float w2_norm = w2 / w_sum;
	const float w3_norm = w3 / w_sum;
	const float w4_norm = w4 / w_sum;

	const float v = random.next();
	if (v < w1_norm)
	{
		// invert the diffuse component
		if (invert_diffuse(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// transform the z component into the desired range [0,w1)
		z.z = z.z * w1;

		// compute the change of density due to squeezing the interval [0,w1/w_sum) into the interval [0,w1)
		TRANSFORM_PDF(p_proj, w1, w1_norm);
		TRANSFORM_PDF(p,      w1, w1_norm);
	}
	else if (v < w1_norm + w2_norm)
	{
		// invert the glossy component
		if (invert_glossy(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// transform the z component into the desired range [w1,w2)
		z.z = z.z * w2 + w1;

		// compute the change of density due to squeezing the interval [w1/w_sum,w2/w_sum) into the interval [w1,w2)
		TRANSFORM_PDF(p_proj, w2, w2_norm);
		TRANSFORM_PDF(p,      w2, w2_norm);
	}
	else if (v < w1_norm + w2_norm + w3_norm)
	{
		// invert the diffuse transmission component
		if (invert_diffuse_trans(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// transform the z component into the desired range [w1,w2)
		z.z = z.z * w3 + w1 + w2;

		// compute the change of density due to squeezing the interval [w2/w_sum,w3/w_sum) into the interval [w2,w3)
		TRANSFORM_PDF(p_proj, w3, w3_norm);
		TRANSFORM_PDF(p,      w3, w3_norm);
	}
	else
	{
		// invert the glossy transmission component
		if (invert_glossy_trans(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// transform the z component into the desired range [w1,w2)
		z.z = z.z * w4 + w1 + w2 + w3;

		// compute the change of density due to squeezing the interval [w3/w_sum,w4/w_sum) into the interval [w3,w4)
		TRANSFORM_PDF(p_proj, w4, w4_norm);
		TRANSFORM_PDF(p,      w4, w4_norm);
	}

	float p1, p1_proj;
	float p2, p2_proj;
	float p3, p3_proj;
	float p4, p4_proj;

	// invert the diffuse component
	p1      = bsdf.diffuse().p(v_prev, in, out, cugar::kSolidAngle);
	p1_proj = bsdf.diffuse().p(v_prev,	in,	out, cugar::kProjectedSolidAngle);

	p1		*= w1;
	p1_proj *= w1;

	// invert the glossy component
	p2      = bsdf.glossy().p(v_prev, in, out, cugar::kSolidAngle);
	p2_proj = bsdf.glossy().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p2      *= w2;
	p2_proj *= w2;

	// invert the diffuse transmission component
	p3		= bsdf.diffuse_trans().p(v_prev, in, out, cugar::kSolidAngle);
	p3_proj = bsdf.diffuse_trans().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p3		*= w3;
	p3_proj *= w3;

	// invert the glossy transmission component
	p4		= bsdf.glossy_trans().p(v_prev, in, out, cugar::kSolidAngle);
	p4_proj = bsdf.glossy_trans().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p4		*= w4;
	p4_proj *= w4;

	p		= 1.0f / (p1 + p2 + p3 + p4);
	p_proj  = 1.0f / (p1_proj + p2_proj + p3_proj + p4_proj);

	return true;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
void inversion_pdf(
	const Bsdf&				bsdf,
	const VertexGeometry&	v_prev,
	const cugar::Vector3f&	in,
	const cugar::Vector3f&	out,
	cugar::Vector3f&		z,
	float&					p,
	float&					p_proj)
{
	// choose the BSDF component utilized for inversion
	float diffuse_refl_prob;
	float diffuse_trans_prob;
	float glossy_refl_prob;
	float glossy_trans_prob;

	bsdf.sampling_weights(v_prev, in, diffuse_refl_prob, diffuse_trans_prob, glossy_refl_prob, glossy_trans_prob);

	float p1, p1_proj;
	float p2, p2_proj;
	float p3, p3_proj;
	float p4, p4_proj;

	// invert the diffuse component
	p1      = bsdf.diffuse().p(v_prev, in, out, cugar::kSolidAngle);
	p1_proj = bsdf.diffuse().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p1      *= diffuse_refl_prob;
	p1_proj *= diffuse_refl_prob;

	// invert the glossy component
	p2      = bsdf.glossy().p(v_prev, in, out, cugar::kSolidAngle);
	p2_proj = bsdf.glossy().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p2      *= glossy_refl_prob;
	p2_proj *= glossy_refl_prob;

	// invert the diffuse transmission component
	p3		= bsdf.diffuse_trans().p(v_prev, in, out, cugar::kSolidAngle);
	p3_proj = bsdf.diffuse_trans().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p3		*= diffuse_trans_prob;
	p3_proj *= diffuse_trans_prob;

	// invert the glossy transmission component
	p4		= bsdf.glossy_trans().p(v_prev, in, out, cugar::kSolidAngle);
	p4_proj = bsdf.glossy_trans().p(v_prev, in, out, cugar::kProjectedSolidAngle);

	p4		*= glossy_trans_prob;
	p4_proj *= glossy_trans_prob;

	p		= 1.0f / (p1 + p2 + p3 + p4);
	p_proj	= 1.0f / (p1_proj + p2_proj + p3_proj + p4_proj);
}

template <typename PathType, typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_eye_subpath(const PathType& path, const uint32 t, const uint32 t_ext, float* out_Z, float* out_pdf, const RendererView& renderer, TRandomGenerator& random)
{
	VertexGeometry  v_prev;
	cugar::Vector3f in;

	*out_pdf = 1.0f;

	if (t == 0)
	{
		// special case: this is a light path which hit the camera, and we need to invert the lens sampling
		return false;
	}
	else
	{
		// setup vertex t - 1
		setup_differential_geometry(renderer.mesh, path.v_E(t - 1), &v_prev);

		// setup the incoming edge at t - 1
		if (t == 1)
			in = cugar::Vector3f(0.0f);
		else if (t == 2)
		{
			// vertex zero is represented by uv's on the lens (currently a pinhole)
			in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - v_prev.position);
		}
		else
		{
			// fetch vertex t - 2
			const cugar::Vector3f p_prev2 = interpolate_position(renderer.mesh, path.v_E(t - 2));

			in = cugar::normalize(p_prev2 - v_prev.position);
		}
	}

	// NOTE: here we can basically assume t is at least 2, otherwise, if t == 1 we would have to consider the camera sampler instead of the BSDF at i = 1
	// (corresponding to pure light tracing)
	for (uint32 i = t; i < t_ext; ++i)
	{
		VertexGeometryId v_id = path.v_E(i);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the BSDF at v_prev

		// fetch the material at v_prev
		const int material_id = renderer.mesh.material_indices[path.v_E(i-1).prim_id];
		MeshMaterial material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse  *= texture_lookup(v_prev.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular *= texture_lookup(v_prev.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive *= texture_lookup(v_prev.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));

		Bsdf bsdf(kRadianceTransport, renderer, material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max( cugar::square_length(out), 1.0e-8f );

		out /= sqrtf(d2); // normalize it

		// 3. invert the bsdf
		cugar::Vector3f z;
		float			p, p_proj;

		if (invert_bsdf(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// store the recovered primary space coordinates
		out_Z[(i - t) * 3 + 0] = z.x;
		out_Z[(i - t) * 3 + 1] = z.y;
		out_Z[(i - t) * 3 + 2] = z.z;

		// multiply by the inverse of the geometric term G(v_prev,v)
		*out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//*out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in     = -out;
	}
	return true;
}

template <typename PathType>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float eye_subpath_inversion_pdf(const PathType& path, const uint32 t, const uint32 t_ext, const float* out_Z, const RendererView& renderer)
{
	FERMAT_ASSERT(t >= 2);
	FERMAT_ASSERT(t_ext <= path.n_vertices);

	VertexGeometry  v_prev;
	cugar::Vector3f in;

	float out_pdf = 1.0f;

	if (t == 0)
	{
		// special case: this is a light path which hit the camera, and we need to invert the lens sampling
		return 0.0f;
	}
	else
	{
		// setup vertex t - 1
		setup_differential_geometry(renderer.mesh, path.v_E(t - 1), &v_prev);

		// setup the incoming edge at t - 1
		if (t == 1)
			in = cugar::Vector3f(0.0f);
		else if (t == 2)
		{
			// vertex zero is represented by uv's on the lens (currently a pinhole)
			in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - v_prev.position);
		}
		else
		{
			// fetch vertex t - 2
			const cugar::Vector3f p_prev2 = interpolate_position(renderer.mesh, path.v_E(t - 2));

			in = cugar::normalize(p_prev2 - v_prev.position);
		}
	}

	// NOTE: here we can basically assume t is at least 2, otherwise, if t == 1 we would have to consider the camera sampler instead of the BSDF at i = 1
	// (corresponding to pure light tracing)
	for (uint32 i = t; i < t_ext; ++i)
	{
		VertexGeometryId v_id = path.v_E(i);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the BSDF at v_prev

		// fetch the material at v_prev
		const int material_id = renderer.mesh.material_indices[path.v_E(i - 1).prim_id];
		MeshMaterial material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse  *= texture_lookup(v_prev.texture_coords, material.diffuse_map, renderer.textures, cugar::Vector4f(1.0f));
		material.specular *= texture_lookup(v_prev.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive *= texture_lookup(v_prev.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));

		Bsdf bsdf(kRadianceTransport, renderer, material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max(cugar::square_length(out), 1.0e-8f);

		out /= sqrtf(d2); // normalize it

		// 3. compute the inversion pdf
		cugar::Vector3f z;
		float			p, p_proj;

		// fetch the recovered primary space coordinates
		z.x = out_Z[(i - t) * 3 + 0];
		z.y = out_Z[(i - t) * 3 + 1];
		z.z = out_Z[(i - t) * 3 + 2];

		inversion_pdf(
			bsdf,
			v_prev,
			in,
			out,
			z,
			p,
			p_proj);

		// multiply by the inverse of the geometric term G(v_prev,v)
		out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in = -out;
	}
	return out_pdf;
}

template <typename PathType, typename TRandomGenerator>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
bool invert_light_subpath(const PathType& path, const uint32 s, const uint32 s_ext, float* out_Z, float* out_pdf, const RendererView& renderer, TRandomGenerator& random)
{
	FERMAT_ASSERT(s_ext <= path.n_vertices);

	VertexGeometry  v_prev;
	cugar::Vector3f in;

	*out_pdf = 1.0f;

	if (s == 0)
	{
		float z[3] = { random.next(), random.next(), random.next() };

		// special case: this is a eye path which hit the light, and we need to invert the light source sampling
		if (renderer.mesh_light.invert_impl(path.v_L(0).prim_id, path.v_L(0).uv, z, out_Z, out_pdf) == false)
			return false;

		setup_differential_geometry(renderer.mesh, path.v_L(0), &v_prev);

		in = cugar::Vector3f(0.0f);
	}
	else
	{
		// setup vertex s - 1
		setup_differential_geometry(renderer.mesh, path.v_L(s - 1), &v_prev);

		// setup the incoming edge at s - 1
		if (s == 1)
			in = cugar::Vector3f(0.0f);
		else
		{
			// fetch vertex s - 2
			const cugar::Vector3f p_prev2 = interpolate_position( renderer.mesh, path.v_L(s - 2) );

			in = cugar::normalize(p_prev2 - v_prev.position);
		}
	}

	if (s <= 1 && s_ext >= 2)
	{
		// invert the EDF at the light source
		VertexGeometryId v_id = path.v_L(1);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the EDF at v_prev

		// fetch the material at v_prev
		const int material_id = renderer.mesh.material_indices[path.v_L(0).prim_id];
		MeshMaterial material = renderer.mesh.materials[material_id];

		// NOTE: we don't interpolate textures here because, for the currently used diffuse EDF, the probability is not affected by the actual material parameters
		Edf edf(material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max(cugar::square_length(out), 1.0e-8f);

		out /= sqrtf(d2); // normalize it

	    // 3. invert the EDF
		cugar::Vector3f z;
		float			p, p_proj;

		edf.invert(
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj);

		// store the recovered primary space coordinates
		out_Z[(1 - s) * 3 + 0] = z.x;
		out_Z[(1 - s) * 3 + 1] = z.y;
		out_Z[(1 - s) * 3 + 2] = z.z;

		// multiply by the inverse of the geometric term G(v_prev,v)
		*out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//*out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in = -out;
	}

	for (uint32 i = cugar::max( s, 2u ); i < s_ext; ++i)
	{
		FERMAT_ASSERT(path.v_L(i).prim_id < renderer.mesh.num_triangles);
		VertexGeometryId v_id = path.v_L(i);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the BSDF at v_prev

		// fetch the material at v_prev
		FERMAT_ASSERT(path.v_L(i - 1).prim_id < renderer.mesh.num_triangles);
		const int material_id = renderer.mesh.material_indices[path.v_L(i - 1).prim_id];
		FERMAT_ASSERT(material_id >= 0 && material_id < renderer.mesh.num_materials);
		MeshMaterial material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse  *= texture_lookup(v_prev.texture_coords, material.diffuse_map,  renderer.textures, cugar::Vector4f(1.0f));
		material.specular *= texture_lookup(v_prev.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive *= texture_lookup(v_prev.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));

		Bsdf bsdf(kParticleTransport, renderer, material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max(cugar::square_length(out), 1.0e-8f);

		out /= sqrtf(d2); // normalize it

		// 3. invert the bsdf
		cugar::Vector3f z;
		float			p, p_proj;

		if (invert_bsdf(
			bsdf,
			v_prev,
			in,
			out,
			random,
			z,
			p,
			p_proj) == false)
			return false;

		// store the recovered primary space coordinates
		out_Z[(i - s) * 3 + 0] = z.x;
		out_Z[(i - s) * 3 + 1] = z.y;
		out_Z[(i - s) * 3 + 2] = z.z;

		// multiply by the inverse of the geometric term G(v_prev,v)
		*out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//*out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in = -out;
	}
	return true;
}

template <typename PathType>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float light_subpath_inversion_pdf(const PathType& path, const uint32 s, const uint32 s_ext, const float* out_Z, const RendererView& renderer)
{
	FERMAT_ASSERT(s_ext <= path.n_vertices);

	VertexGeometry  v_prev;
	cugar::Vector3f in;

	float out_pdf = 1.0f;

	if (s == 0)
	{
		// special case: this is a eye path which hit the light, and we need to invert the light source sampling
		out_pdf = renderer.mesh_light.inverse_pdf_impl(path.v_L(0).prim_id, path.v_L(0).uv, out_Z);

		setup_differential_geometry(renderer.mesh, path.v_L(0), &v_prev);

		in = cugar::Vector3f(0.0f);
	}
	else
	{
		// setup vertex s - 1
		setup_differential_geometry(renderer.mesh, path.v_L(s - 1), &v_prev);

		// setup the incoming edge at s - 1
		if (s == 1)
			in = cugar::Vector3f(0.0f);
		else
		{
			// fetch vertex s - 2
			const cugar::Vector3f p_prev2 = interpolate_position(renderer.mesh, path.v_L(s - 2));

			in = cugar::normalize(p_prev2 - v_prev.position);
		}
	}

	if (s <= 1 && s_ext >= 2)
	{
		// invert the EDF at the light source
		VertexGeometryId v_id = path.v_L(1);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the EDF at v_prev

		// fetch the material at v_prev
		const int material_id = renderer.mesh.material_indices[path.v_L(0).prim_id];
		MeshMaterial material = renderer.mesh.materials[material_id];

		// NOTE: we don't interpolate textures here because, for the currently used diffuse EDF, the probability is not affected by the material parameters
		Edf edf(material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max(cugar::square_length(out), 1.0e-8f);

		out /= sqrtf(d2); // normalize it

		// 3. invert the EDF
		cugar::Vector3f z;
		float			p, p_proj;

		// fetch the recovered primary space coordinates
		z.x = out_Z[(1 - s) * 3 + 0];
		z.y = out_Z[(1 - s) * 3 + 1];
		z.z = out_Z[(1 - s) * 3 + 2];

		edf.inverse_pdf(
			v_prev,
			in,
			out,
			z,
			p,
			p_proj);

		// multiply by the inverse of the geometric term G(v_prev,v)
		out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in = -out;
	}

	for (uint32 i = cugar::max(s, 2u); i < s_ext; ++i)
	{
		VertexGeometryId v_id = path.v_L(i);
		VertexGeometry   v;

		setup_differential_geometry(renderer.mesh, v_id, &v);

		// invert the edge (v_prev -> v)
		//

		// 1. build the BSDF at v_prev

		// fetch the material at v_prev
		const int material_id = renderer.mesh.material_indices[path.v_L(i - 1).prim_id];
		MeshMaterial material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse  *= texture_lookup(v_prev.texture_coords, material.diffuse_map, renderer.textures, cugar::Vector4f(1.0f));
		material.specular *= texture_lookup(v_prev.texture_coords, material.specular_map, renderer.textures, cugar::Vector4f(1.0f));
		material.emissive *= texture_lookup(v_prev.texture_coords, material.emissive_map, renderer.textures, cugar::Vector4f(1.0f));

		Bsdf bsdf(kParticleTransport, renderer, material);

		// 2. setup the outgoing edge
		cugar::Vector3f out = v.position - v_prev.position;

		const float d2 = cugar::max(cugar::square_length(out), 1.0e-8f);

		out /= sqrtf(d2); // normalize it

	  // 3. compute the inversion pdf
		cugar::Vector3f z;
		float			p, p_proj;

		// fetch the recovered primary space coordinates
		z.x = out_Z[(i - s) * 3 + 0];
		z.y = out_Z[(i - s) * 3 + 1];
		z.z = out_Z[(i - s) * 3 + 2];

		inversion_pdf(
			bsdf,
			v_prev,
			in,
			out,
			z,
			p,
			p_proj);

		// multiply by the inverse of the geometric term G(v_prev,v)
		out_pdf *= p_proj * d2 / cugar::max(fabsf(dot(v_prev.normal_s, out)*dot(v.normal_s, out)), 1.0e-8f);
		//out_pdf *= p * d2 / cugar::max( fabsf( dot(v.normal_s,out) ), 1.0e-8f);

		// curry the previous vertex and incoming edge information to the next iteration
		v_prev = v;
		in = -out;
	}
	return out_pdf;
}

///@} PathModule
///@} Fermat
