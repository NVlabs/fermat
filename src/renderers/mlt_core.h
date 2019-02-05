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

#define DISABLE_SPHERICAL_PERTURBATIONS	0

#define SPHERICAL_SCREEN_PERTURBATIONS  0
#define DISK_SCREEN_SPACE_PERTURBATION  1

#include <mlt.h>
#include <mlt_perturbations.h>
#include <renderer.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/sampling/distributions.h>
#include <cugar/color/rgbe.h>
#include <cugar/basic/cuda/warp_atomics.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_utils.h>
#include <bpt_context.h>
#include <bpt_control.h>
#include <bpt_samplers.h>
#include <tiled_sampling.h>
#include <path_inversion.h>


namespace {

	FERMAT_HOST_DEVICE
	bool valid_sample(const float3 f)
	{
		return
			cugar::is_finite(f.x) && cugar::is_finite(f.y) && cugar::is_finite(f.z) &&
			!cugar::is_nan(f.x) && !cugar::is_nan(f.y) && !cugar::is_nan(f.z);
	}

	FERMAT_HOST_DEVICE
	bool valid_non_zero_sample(const float3 f)
	{
		return
			cugar::max_comp( f ) > 0.0f &&
			valid_sample( f );
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f reconstruct_input_edge(const VertexGeometryId v_prev, const VertexGeometryId v, const RenderingContextView& renderer)
	{
		return 
			interpolate_position(renderer.mesh, v_prev) - 
			interpolate_position(renderer.mesh, v);
	}

	template <typename PathVertexType>
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	PathVertexType reconstruct_vertex(const VertexGeometryId v_prev, const VertexGeometryId v, const uint32 in_bounce, const RenderingContextView& renderer)
	{
		const cugar::Vector3f old_in = reconstruct_input_edge( v_prev, v, renderer );
		return PathVertexType( old_in, v, cugar::Vector3f(0.0f), TempPathWeights(), in_bounce, renderer );
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float cos_theta(const cugar::Vector3f& N, const cugar::Vector3f dir) { return fabsf( dot(N, dir) ); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float cos_theta(const VertexGeometry& geom, const cugar::Vector3f dir) { return fabsf( dot(geom.normal_s, dir) ); }

} // anonymous namespace


struct MLTContext : BPTContextBase<MLTOptions>
{
	TiledSequenceView	sequence;
	uint32				n_light_paths;		// number of light paths
	uint32				n_eye_paths;		// number of eye paths
	float4*				connections_value;
	uint4*				connections_index;
	uint32*				connections_counter;
	float*				st_norms;
	float*				st_norms_cdf;
	uint32*				st_counters;
	uint32*				seeds;
	char4*				st;
	VertexGeometryId*	bpt_light_vertices;		// BPT light vertices
	VertexGeometryId*	bpt_eye_vertices;		// BPT eye vertices
	VertexGeometryId*	mut_vertices;
	VertexGeometryId*	vertices;
	float4*				path_value;
	float*				path_pdf;
	uint32*				rejections;
	uint32				n_chains;
	uint32				chain_length;
	uint32				chain_step;
	float				pdf_norm;
	uint32				enable_mutations;
	float*				acceptance_rate_sum;
	float*				checksum;

	float4*				mut_f_vertices;
	float4*				f_vertices;
	cugar::Vector4f*	Q_old;
	cugar::Vector4f*	Q_new;

	MeshLight			mesh_light;

	MLTContext() {}

	MLTContext(
		MLT&							_mlt,
		const RenderingContextView&		_renderer) :
		BPTContextBase(
			_renderer,
			_mlt.m_light_vertices.view(),
			_mlt.m_queues.view(_mlt.m_n_init_paths, _mlt.m_n_init_paths),
			_mlt.m_options ),
		sequence(_mlt.m_sequence.view()),
		mesh_light( options.use_vpls ? _renderer.mesh_vpls : _renderer.mesh_light )
	{}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float u_m(const uint32 chain_id, const uint32 dim) const { return cugar::randfloat( dim, chain_id + n_chains * chain_step ); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void discard(const uint32 chain_id)
	{
		Q_old[chain_id] = 1.0f;
		Q_new[chain_id] = 0.0f;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool discard_invalid(const uint32 chain_id)
	{
		if (!valid_sample( Q_old[chain_id].xyz() ) ||
			!valid_sample( Q_new[chain_id].xyz() ))
		{
			discard( chain_id );
			return true;
		}
		return false;
	}
};

namespace { // anonymous namespace

	// A config for the BPT presampling/seeding pass, used to store all generated path vertices
	//
	struct MLTPresamplingBPTConfig : BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		MLTPresamplingBPTConfig(MLTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kAll,
				VertexOrdering::kRandomOrdering,
				VertexSampling::kAll,
				true) {}

		// store the given light vertex
		//
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		void visit_light_vertex(
			const uint32				light_path_id,
			const uint32				depth,
			const VertexGeometryId		v_id,
			MLTContext&					context,
			const RenderingContextView&	renderer) const
		{
			if (context.bpt_light_vertices)
				context.bpt_light_vertices[light_path_id + depth * context.n_light_paths] = v_id;
		}

		// store the given eye vertex
		//
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		void visit_eye_vertex(
			const uint32				eye_path_id,
			const uint32				depth,
			const VertexGeometryId		v_id,
			const EyeVertex&			v,
			MLTContext&					context,
			const RenderingContextView&	renderer) const
		{
			if (context.bpt_eye_vertices)
				context.bpt_eye_vertices[eye_path_id + depth * context.n_eye_paths] = v_id;
		}
	};

	//
	// The \ref SampleSinkAnchor "Sample Sink" used by the BPT presampling/seeding pass
	//
	struct ConnectionsSink : SampleSinkBase
	{
		FERMAT_HOST_DEVICE
		ConnectionsSink() {}

		FERMAT_HOST_DEVICE
		void sink(
			const uint32			channel,
			const cugar::Vector4f	value,
			const uint32			light_path_id,
			const uint32			eye_path_id,
			const uint32			s,
			const uint32			t,
			MLTContext&				context,
			RenderingContextView&	renderer)
		{
			if (cugar::max_comp(value.xyz()) > 0.0f)
			{
				const uint32 slot = cugar::atomic_add(context.connections_counter, 1);
				context.connections_value[slot] = value;
				context.connections_index[slot] = make_uint4(light_path_id, eye_path_id, s, t);

				cugar::atomic_add(context.st_norms + s + t * (context.options.max_path_length + 2), cugar::max_comp(value.xyz()));
			}
		}
	};

	struct DecorrelatedRandoms
	{
		FERMAT_HOST_DEVICE
		DecorrelatedRandoms(const uint32 _dim, const uint32 _seed) : dim(_dim), seed(_seed) {}

		FERMAT_HOST_DEVICE
		float next()
		{
			return cugar::randfloat( dim++, seed % 65535u );
		}

		uint32 dim;
		uint32 seed;
	};

	FERMAT_DEVICE FERMAT_FORCEINLINE
	void accept_reject_accumulate(const uint32 chain_id, const uint32 s, const uint32 t, const cugar::Vector4f w, MLTContext& context, RenderingContextView& renderer)
	{
		// perform the MH acceptance-rejection test
		const float new_pdf = cugar::max_comp(w.xyz());
		const float old_pdf = context.path_pdf[chain_id];

		FERMAT_ASSERT( cugar::is_finite(old_pdf) && !cugar::is_nan(old_pdf) );
		FERMAT_ASSERT( cugar::is_finite(new_pdf) && !cugar::is_nan(new_pdf) );
		FERMAT_ASSERT( old_pdf >= 0.0f );
		FERMAT_ASSERT( new_pdf >= 0.0f );

		const Path old_path(s + t, context.vertices + chain_id, context.n_chains);
		const Path new_path(s + t, context.mut_vertices + chain_id, context.n_chains);

		const cugar::Vector2f old_uv = old_path.v_E(0).uv;
		const cugar::Vector2f new_uv = new_path.v_E(0).uv;

		const uint32 old_pixel_x = cugar::quantize(old_uv.x, renderer.res_x);
		const uint32 old_pixel_y = cugar::quantize(old_uv.y, renderer.res_y);
		const uint32 old_pixel   = old_pixel_x + old_pixel_y*renderer.res_x;

		const uint32 new_pixel_x = cugar::quantize(new_uv.x, renderer.res_x);
		const uint32 new_pixel_y = cugar::quantize(new_uv.y, renderer.res_y);
		const uint32 new_pixel   = new_pixel_x + new_pixel_y*renderer.res_x;

		//const float T_ratio = w.w;
		//const float ar = old_pdf ? fminf(1.0f, T_ratio * (new_pdf / old_pdf)) : (new_pdf ? 1.0f : 0.0f);
		//FERMAT_ASSERT( cugar::is_finite(T_ratio) && !cugar::is_nan(T_ratio) );
		//FERMAT_ASSERT( T_ratio >= 0.0f );
		const float Q_old = cugar::max_comp( context.Q_old[chain_id].xyz() );
		const float Q_new = cugar::max_comp( context.Q_new[chain_id].xyz() );
		const float ar = Q_old ? cugar::min( 1.0f, Q_new / Q_old ) : (Q_new ? 1.0f : 0.0f);
		FERMAT_ASSERT( cugar::is_finite(Q_old) && !cugar::is_nan(Q_old) );
		FERMAT_ASSERT( cugar::is_finite(Q_new) && !cugar::is_nan(Q_new) );
		FERMAT_ASSERT( cugar::is_finite(ar) && !cugar::is_nan(ar) );
		FERMAT_ASSERT( ar >= 0.0f );

		if (old_pdf > 0)
		{
			const float4 old_value = context.path_value[chain_id];

			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).x, context.pdf_norm * (1.0f - ar) * (old_value.x / old_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).y, context.pdf_norm * (1.0f - ar) * (old_value.y / old_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).z, context.pdf_norm * (1.0f - ar) * (old_value.z / old_pdf) / float(renderer.instance + 1));
		}
		if (new_pdf > 0)
		{
			const float4 new_value = w;

			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).x, context.pdf_norm * ar * (new_value.x / new_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).y, context.pdf_norm * ar * (new_value.y / new_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).z, context.pdf_norm * ar * (new_value.z / new_pdf) / float(renderer.instance + 1));
		}

		if (context.u_m(chain_id, (context.options.max_path_length + 1) * 3) < ar)
		{
			context.path_value[chain_id] = w;
			context.path_pdf[chain_id] = new_pdf;

			// copy the compact vertex information
			for (uint32 i = 0; i < s + t; ++i)
			{
				context.vertices[chain_id + i * context.n_chains] = context.mut_vertices[chain_id + i * context.n_chains];
				context.f_vertices[chain_id + i * context.n_chains] = context.mut_f_vertices[chain_id + i * context.n_chains];
			}

			context.rejections[chain_id] = 0;
		}
		else
		{
			context.rejections[chain_id]++;

			//if ((context.rejections[chain_id] % 5) == 0)
			//	printf("chain[%u,%u:%u] stuck for %u iterations\n", s, t, chain_id, context.rejections[chain_id]);
		}

		// keep some stats
		cugar::atomic_add( context.acceptance_rate_sum, ar );

		// keep track of a checksum
		cugar::atomic_add( context.checksum, context.path_pdf[chain_id] );
	}

	//------------------------------------------------------------------------------

	FERMAT_DEVICE
	bool perturb_primary_light_vertex(MLTContext& context, RenderingContextView& renderer, RayQueue& scatter_queue, const uint32 chain_id)
	{
		// fetch the original (s,t) pair
		const uint32 s_old = context.st[chain_id].x;
		const uint32 t_old = context.st[chain_id].y;

		// and generate a new pair
		const uint32 t = (context.enable_mutations == false) || (context.options.st_perturbations == false) || (s_old == 0) ?
			t_old :
			2 + cugar::quantize(context.u_m(chain_id, 0), s_old + t_old - 2); // t for now is at least 2
		const uint32 s = s_old + t_old - t;
		FERMAT_ASSERT(s + t == s_old + t_old);

		// store the new (s,t) pair
		context.st[chain_id] = make_char4(s_old, t_old, s, t);

		// set up the acceptance ratios
		context.Q_old[chain_id] = cugar::Vector4f(1.0f);
		context.Q_new[chain_id] = cugar::Vector4f(1.0f);

		// check if the light subpath stops here
		if (s == 0)
		{
			context.light_vertices.vertex_path_id[chain_id] = uint32(-1);
			context.light_vertices.vertex_weights[chain_id]	= PathWeights(1.0f, 0.0f); // track the acceptance rate factors
			return true;
		}

		VertexGeometryId	photon;
		VertexGeometry		geom;
		float				pdf;
		Edf					edf;

		// for now we never perturb the position on the light source, so we just fetch the old one
		//   TODO: if we ever decided to perturb the position we'd also have to track the geometric
		//   acceptance ratio T_ratio = s >= 1 ? (pdf_old / pdf_new) : 0
		photon = context.vertices[chain_id];

		// recover the pdf and EDF information
		context.mesh_light.map(photon.prim_id, photon.uv, &geom, &pdf, &edf);

		// store the compact vertex information
		Path path(s + t, context.mut_vertices + chain_id, context.n_chains);
		path.v_L(0) = photon;

		// track the bsdf's at each vertex
		PathCache<float4> f_path(s + t, context.mut_f_vertices + chain_id, context.n_chains);

		const uint32 packed_normal = pack_direction(geom.normal_s);

		// update the acceptance ratios
		context.Q_old[chain_id] *= 1.0f / pdf;
		context.Q_new[chain_id] *= 1.0f / pdf;

		// store the photon if and only if s == 1
		if (s == 1)
		{
			context.light_vertices.vertex[chain_id]			= VPL(photon.prim_id, photon.uv, 1.0f); // track the acceptance rate factors in the .w component
			context.light_vertices.vertex_path_id[chain_id]	= chain_id;
			context.light_vertices.vertex_gbuffer[chain_id]	= make_uint4(cugar::to_rgbe(edf.color), 0, 0, __float_as_uint(pdf));
			context.light_vertices.vertex_pos[chain_id]		= cugar::Vector4f(geom.position, __uint_as_float(packed_normal));
			context.light_vertices.vertex_input[chain_id]	= make_uint2(0, cugar::to_rgbe(cugar::Vector3f(1.0f)/* / pdf*/)); // the material emission factor
			context.light_vertices.vertex_weights[chain_id]	=  PathWeights(
				0.0f,			// p(-2)g(-2)p(-1)
				1.0f * pdf);	// p(-1)g(-1) = pdf								: we want p(-1)g(-1)p(0) = p(0)*pdf - which will happen because we are setting p(0) = p(0) and g(-1) = pdf
			return true; // the light subpath stops here
		}
		else
		{
			// temporarily generate an invalid photon - used in case the light subpath gets terminated early
			context.light_vertices.vertex_path_id[chain_id] = uint32(-1);
		}

		if (s > 1)
		{
			// setup the old path
			const Path old_path(s + t, context.vertices + chain_id, context.n_chains);
			const PathCache<float4> old_f_path(s + t, context.f_vertices + chain_id, context.n_chains);

			const VertexGeometryId old_v	  = old_path.v_L(0);
			const VertexGeometryId old_v_next = old_path.v_L(1);

			// compute the old outgoing direction
			const cugar::Vector3f out_old = cugar::normalize(
				interpolate_position(renderer.mesh, old_v_next) -
				interpolate_position(renderer.mesh, old_v) );

			// fetch the randoms we need
			cugar::Vector2f samples;
			for (uint32 i = 0; i < 2; ++i)
				samples[i] = context.u_m(chain_id, 3 + i);

			// apply a spherical perturbation
			const cugar::Vector3f out = context.enable_mutations ? spherical_perturbation(geom, out_old, samples, context.options.perturbation_radius) : out_old;

			// evaluate the EDF
			const cugar::Vector3f f = edf.f( geom, geom.position, out );

			// cache the first vertex bsdf
			f_path.v_L(0) = cugar::Vector4f(f, 0.0f);

			if (cugar::max_comp(f))
			{
				FERMAT_ASSERT( cugar::max_comp(f) > 0.0f );
				Ray out_ray;
				out_ray.origin	= geom.position;
				out_ray.dir		= out;
				out_ray.tmin	= 1.0e-4f;
				out_ray.tmax	= 1.0e8f;

				context.Q_old[chain_id] *= old_f_path.v_L(0) * cos_theta( interpolate_normal(renderer.mesh, old_v), out_old ); 
				context.Q_new[chain_id] *=     f_path.v_L(0) * cos_theta( geom.normal_s, out );
				FERMAT_ASSERT(valid_sample(context.Q_old[chain_id].xyz()));

				// we use the fourth component to track parts of the acceptance rate
				const cugar::Vector4f out_w = cugar::Vector4f(f, 0.0f);

				const float p_proj = edf.p(geom, geom.position, out, cugar::kProjectedSolidAngle);
				const TempPathWeights out_path_weights = TempPathWeights::light_vertex_1( pdf, p_proj, fabsf(dot(geom.normal_s, out)) );

				scatter_queue.warp_append(
					PixelInfo(chain_id, FBufferDesc::DIFFUSE_C),
					out_ray,
					out_w,
					0u,												// we need to track the bounce number
					out_path_weights);								// only needed for MIS
			}
			else
			{
				// kill the new sample
				context.discard( chain_id );
			}
		}
		return false;
	}

	FERMAT_DEVICE
	bool perturb_secondary_light_vertex(MLTContext& context, RenderingContextView& renderer, RayQueue& in_queue, RayQueue& scatter_queue, const uint32 task_id, uint32* out_chain_id = NULL)
	{
		const PixelInfo		  pixel_info	= in_queue.pixels[task_id];
		const Ray			  ray			= in_queue.rays[task_id];
		const Hit			  hit			= in_queue.hits[task_id];
		const cugar::Vector4f w				= in_queue.weights[task_id];
		const TempPathWeights path_weights	= in_queue.path_weights[task_id]; // only needed for MIS
		const uint32		  in_bounce		= context.in_bounce;

		const uint32 chain_id = pixel_info.pixel;
		if (out_chain_id)
			*out_chain_id = chain_id;

		const uint32 s = context.st[chain_id].z;
		const uint32 t = context.st[chain_id].w;

		if (hit.t > 0.0f && hit.triId >= 0)
		{
			// store the compact vertex information
			Path path(s + t, context.mut_vertices + chain_id, context.n_chains);
			path.v_L(in_bounce + 1) = VertexGeometryId(hit.triId, hit.u, hit.v);

			// track the bsdf's at each vertex
			PathCache<float4> f_path(s + t, context.mut_f_vertices + chain_id, context.n_chains);

			// setup the light vertex
			LightVertex lv;
			lv.setup(ray, hit, w.xyz(), path_weights, in_bounce + 1, renderer);

			// store the photon / VPL
			if (in_bounce + 2 == s)
			{
				const uint32 slot = chain_id;

				const uint32 packed_normal = pack_direction(lv.geom.normal_s);
				const uint32 packed_direction = pack_direction(lv.in);

				context.light_vertices.vertex[slot]			= VPL(hit.triId, cugar::Vector2f(hit.u, hit.v), w.w); // track the acceptance rate factors in the .w component
				context.light_vertices.vertex_path_id[slot]	= pixel_info.pixel;
				context.light_vertices.vertex_gbuffer[slot]	= make_uint4(
					cugar::to_rgbe(cugar::Vector4f(lv.material.diffuse).xyz()),
					cugar::to_rgbe(cugar::Vector4f(lv.material.specular).xyz()),
					cugar::binary_cast<uint32>(lv.material.roughness),
					cugar::to_rgbe(cugar::Vector4f(lv.material.diffuse_trans).xyz()));
				context.light_vertices.vertex_pos[slot]		= cugar::Vector4f(lv.geom.position, __uint_as_float(packed_normal));
				context.light_vertices.vertex_input[slot]	= make_uint2(packed_direction, cugar::to_rgbe(w.xyz()));
				context.light_vertices.vertex_weights[slot]	= PathWeights(
					lv.pGp_sum,					// f(i-2)g(i-2)f(i-1)
					lv.prev_pG);				// f(i-1)g(i-1)
			}

			// trace a bounce ray
			if (in_bounce + 2 < s)
			{
				// fetch the randoms we need
				cugar::Vector3f samples;
				for (uint32 i = 0; i < 3; ++i)
					samples[i] = context.u_m(chain_id, (in_bounce + 2) * 3 + i); // remember we need 6 coordinates just to sample the second vertex (3 for the light position + 3 for the direction)

				// setup the old path
				const Path old_path(s + t, context.vertices + chain_id, context.n_chains);
				const PathCache<float4> old_f_path(s + t, context.f_vertices + chain_id, context.n_chains);
					
				const VertexGeometryId old_v	  = old_path.v_L(in_bounce + 1);
				const VertexGeometryId old_v_next = old_path.v_L(in_bounce + 2);

				// compute the outgoing direction in the old path
				const cugar::Vector3f out_old = cugar::normalize(
					interpolate_position(renderer.mesh, old_v_next) -
					interpolate_position(renderer.mesh, old_v));

				cugar::Vector3f out = out_old;

				const float u_p = context.enable_mutations ?
					context.u_m(chain_id, (in_bounce + 2) * 3 + 2) :
					2.0f; // a value higher than 1

				if (u_p < context.options.exp_perturbations)
				{
					// apply a spherical perturbation
					out = spherical_perturbation(lv.geom, out_old, samples.xy(), context.options.perturbation_radius);
				}
				else if (u_p < context.options.H_perturbations
							 + context.options.exp_perturbations)
				{
					// reconstruct the old vertex
					const LightVertex old_lv = reconstruct_vertex<LightVertex>( old_path.v_L(in_bounce), old_v, in_bounce, renderer );

					// apply a H-perturbation
					out = H_perturbation(old_lv, out_old, lv, samples.xy(), context.options.perturbation_radius);
				}
				// evaluate the bsdf
				cugar::Vector3f f = lv.bsdf.f(lv.geom, lv.in, out);

				// cache this vertex' bsdf
				f_path.v_L(in_bounce + 1) = cugar::Vector4f(f, 0.0f);

				// update the acceptance rate factors
				if (u_p < context.options.exp_perturbations)
				{
					context.Q_old[chain_id] *= old_f_path.v_L(in_bounce + 1) * cos_theta( interpolate_normal(renderer.mesh, old_v), out_old );
					context.Q_new[chain_id] *=     f_path.v_L(in_bounce + 1) * cos_theta( lv.geom.normal_s,                         out );
					FERMAT_ASSERT(valid_sample(context.Q_old[chain_id].xyz()));
				}
				else if (u_p < context.options.H_perturbations
							 + context.options.exp_perturbations)
				{
					// compute the geometric portion of the acceptance ratio
					const LightVertex old_lv = reconstruct_vertex<LightVertex>( old_path.v_L(in_bounce), old_v, in_bounce, renderer );

					context.Q_old[chain_id] *= old_f_path.v_L(in_bounce + 1) * H_perturbation_geometric_density( old_lv, out_old ) * cos_theta( old_lv.geom.normal_s, out_old );
					context.Q_new[chain_id] *=     f_path.v_L(in_bounce + 1) * H_perturbation_geometric_density( lv,     out     ) * cos_theta( lv.geom.normal_s,	  out );
					FERMAT_ASSERT(valid_sample(context.Q_old[chain_id].xyz()));
				}
				else
				{
					context.Q_old[chain_id] *= old_f_path.v_L(in_bounce + 1) * cos_theta( interpolate_normal(renderer.mesh, old_v), out_old );
					context.Q_new[chain_id] *=     f_path.v_L(in_bounce + 1) * cos_theta( lv.geom.normal_s,                         out );
					FERMAT_ASSERT(valid_sample(context.Q_old[chain_id].xyz()));
				}

				// we use the fourth component to track parts of the acceptance rate
				const cugar::Vector4f out_w = cugar::Vector4f(f * w.xyz(), w.w);

				if (valid_non_zero_sample( out_w.xyz() ))
				{
					// enqueue the output ray
					Ray out_ray;
					out_ray.origin	= lv.geom.position;
					out_ray.dir		= out;
					out_ray.tmin	= 1.0e-4f;
					out_ray.tmax	= 1.0e8f;

					const PixelInfo out_pixel = pixel_info;

					const float p_proj = lv.bsdf.p(lv.geom, lv.in, out, cugar::kProjectedSolidAngle);
					const TempPathWeights out_path_weights( lv, out, p_proj );
						
					scatter_queue.warp_append(
						out_pixel,
						out_ray,
						out_w,
						in_bounce + 1,
						out_path_weights );	// only needed for MIS
				}
				else
				{
					// kill the new sample
					context.discard( chain_id );
				}
			}
		}
		else
		{
			// hit the environment - nothing to do
			//

			// kill the new sample
			context.discard( chain_id );
			return true; // the light subpath stops here
		}

		return (in_bounce + 2 == s);
	}

	FERMAT_DEVICE
	void perturb_primary_eye_vertex(MLTContext& context, RenderingContextView& renderer, RayQueue& out_queue, const uint32 chain_id, const uint32 out_slot)
	{
		const uint32 s = context.st[chain_id].z;
		const uint32 t = context.st[chain_id].w;

		// setup the old path
		Path old_path(s + t, context.vertices + chain_id, context.n_chains);

	#if 0
		// fetch the old primary ray direction
		cugar::Vector3f out_old = cugar::normalize(
			interpolate_position(renderer.mesh, old_path.v_E(1)) -
			cugar::Vector3f(renderer.camera.eye));

		// invert the screen sampling
		cugar::Vector2f uv = renderer.camera_sampler.invert(out_old);
	#else
		// fetch the conveniently pre-packaged screen uv coordinates stored in the zero-th vertex
		cugar::Vector2f uv = old_path.v_E(0).uv;
	#endif

		cugar::Vector3f old_ray_direction = renderer.camera_sampler.sample_direction(uv);

		const float u_p = context.u_m(chain_id, (s + t - 2)*3 + 2);

		if (context.enable_mutations && u_p < context.options.screen_perturbations)
		{
			// fetch the randoms we need
			cugar::Vector2f samples;
			for (uint32 i = 0; i < 2; ++i)
				samples[i] = context.u_m(chain_id, (s + t - 2)*3 + i);

		  #if SPHERICAL_SCREEN_PERTURBATIONS
			// and remap it to a new direction
			const cugar::Vector3f out = exponential_spherical_perturbation(cugar::Vector3f(old_ray_direction), samples, context.options.perturbation_radius);

			// map the new direction back to screen-space uv's
			uv = renderer.camera_sampler.invert(out);
		  #elif DISK_SCREEN_SPACE_PERTURBATION
			// map the samples to a point on the disk
			const cugar::Bounded_exponential dist(0.0001f, context.options.perturbation_radius);
			//const cugar::Cauchy_distribution dist(context.options.perturbation_radius);

			// compute polar coordinates
			const float phi   = samples.x*2.0f*M_PIf;
			const float r     = dist.map(samples.y);

			const cugar::Vector2f d = cugar::Vector2f( cosf(phi) * r, sinf(phi) * r );

			// map them to a point and sum all up
			uv.x = cugar::mod( uv.x + d.x, 1.0f );
			uv.y = cugar::mod( uv.y + d.y, 1.0f );
		  #endif
		}

		// store uv's in vertex 0 (even if one day these coordinates should be dedicated to lens uv's...)
		Path path(s + t, context.mut_vertices + chain_id, context.n_chains);
		path.v_E(0) = VertexGeometryId(0, uv);

		cugar::Vector3f ray_origin	 = renderer.camera.eye;
		cugar::Vector3f ray_direction = renderer.camera_sampler.sample_direction(uv);

		((float4*)out_queue.rays)[2 * out_slot + 0] = make_float4(ray_origin.x, ray_origin.y, ray_origin.z, 0.0f); // origin, tmin
		((float4*)out_queue.rays)[2 * out_slot + 1] = make_float4(ray_direction.x, ray_direction.y, ray_direction.z, 1e34f); // dir, tmax

		// write the pixel index
		out_queue.pixels[out_slot] = chain_id;

		// compute the acceptance ratio
		const cugar::Vector3f old_out = cugar::normalize( old_ray_direction );
		const cugar::Vector3f new_out = cugar::normalize( ray_direction );

		// compute the camera response
		const float W_e = renderer.camera_sampler.W_e(new_out);

		// cache the first vertex
		PathCache<float4> f_path(s + t, context.mut_f_vertices + chain_id, context.n_chains);
		f_path.v_E(0) = cugar::Vector4f(W_e, W_e, W_e, 0.0f);

		// fetch the old vertex bsdfs
		PathCache<float4> old_f_path(s + t, context.f_vertices + chain_id, context.n_chains);

	#if SPHERICAL_SCREEN_PERTURBATIONS
		const float old_cos_theta_o = (dot( old_out, renderer.camera_sampler.W ) / renderer.camera_sampler.W_len);
		const float new_cos_theta_o = (dot( new_out, renderer.camera_sampler.W ) / renderer.camera_sampler.W_len);

		context.Q_old[chain_id] *= old_f_path.v_E(0) * old_cos_theta_o;
		context.Q_new[chain_id] *=     f_path.v_E(0) * new_cos_theta_o;
	#else
		// no real need to track the acceptance rates as they are unity
		const float new_pdf = renderer.camera_sampler.pdf(new_out, true);
		const float old_pdf = renderer.camera_sampler.pdf(old_out, true);

		context.Q_old[chain_id] *= old_pdf ? old_f_path.v_E(0) / old_pdf : cugar::Vector4f(1.0f);
		context.Q_new[chain_id] *= new_pdf ?     f_path.v_E(0) / new_pdf : cugar::Vector4f(0.0f);
	#endif

		// write the filter weight
		out_queue.weights[out_slot] = cugar::Vector4f(W_e, W_e, W_e, 1.0f);

		// only needed for MIS
		const float p_e = W_e; // FIXME: only true for the pinhole camera model!
		const float cos_theta = (dot( new_out, renderer.camera_sampler.W ) / renderer.camera_sampler.W_len);
		out_queue.path_weights[out_slot] = TempPathWeights::eye_vertex_1( p_e, cos_theta, context.options.light_tracing );
	}

	FERMAT_DEVICE
	void perturb_secondary_eye_vertex(MLTContext& context, RenderingContextView& renderer, RayQueue& in_queue, RayQueue& scatter_queue, const uint32 task_id)
	{
		const PixelInfo		  pixel_info	= in_queue.pixels[task_id];
		const Ray			  ray			= in_queue.rays[task_id];
		const Hit			  hit			= in_queue.hits[task_id];
		const cugar::Vector4f w				= in_queue.weights[task_id];
		const TempPathWeights path_weights	= in_queue.path_weights[task_id]; // only needed for MIS
		const uint32		  in_bounce		= context.in_bounce;

		const uint32 chain_id = pixel_info.pixel;

		const uint32 s = context.st[chain_id].z;
		const uint32 t = context.st[chain_id].w;

		bool			invalidate_sample = false;
		bool			accumulated_path  = false;
		cugar::Vector4f accumulated_value = 0.0f;

		if (hit.t > 0.0f && hit.triId >= 0)
		{
			// store the compact vertex information
			Path path(s + t, context.mut_vertices + chain_id, context.n_chains);
			path.v_E(in_bounce + 1) = VertexGeometryId(hit.triId, hit.u, hit.v);

			// track the bsdf's at each vertex
			PathCache<float4> f_path(s + t, context.mut_f_vertices + chain_id, context.n_chains);

			// setup the old path
			const Path old_path(s + t, context.vertices + chain_id, context.n_chains);
			PathCache<float4> old_f_path(s + t, context.f_vertices + chain_id, context.n_chains);

			// setup the eye vertex
			EyeVertex ev;
			ev.setup(ray, hit, w.xyz(), path_weights, in_bounce, renderer);

			// perform a single bidirectional connection if and only if we are at vertex 't and 's > 0
			if (in_bounce + 2 == t && s > 0)	// NOTE: in_bounce + 2 represents the technique, i.e. the number of eye subpath vertices
			{
				// fetch the light vertex stored for this chain
				const uint32 light_idx = chain_id;

				const uint32 light_depth = s - 1; // NOTE: s represents the technique, i.e. the number of light subpath vertices. The last vertex has index s - 1.

				// setup a light vertex
				cugar::Vector4f	 light_pos		= context.light_vertices.vertex_pos[light_idx];
				const uint2		 light_in		= context.light_vertices.vertex_input[light_idx];
				uint4            light_gbuffer	= context.light_vertices.vertex_gbuffer[light_idx];
				PathWeights      light_weights	= context.light_vertices.vertex_weights[light_idx];
				const uint32     light_path_id	= context.light_vertices.vertex_path_id[light_idx];

				if (light_path_id != uint32(-1))
				{
					// setup the light vertex
					LightVertex lv(light_pos, light_in, light_gbuffer, light_weights, light_depth, renderer);

					// evaluate the connection
					cugar::Vector3f out;
					cugar::Vector3f out_f;
					cugar::Vector3f out_f_s;
					cugar::Vector3f out_f_L;
					float			out_G;
					float			d;
					float			mis_w;

					eval_connection_terms(ev, lv, out, out_f_s, out_f_L, out_G, d, mis_w, false);

					//if (cugar::length( lv.edf.color - cugar::Vector3f(3.0f, 5.0f, 10.0f)) > 1.0f)
					//	printf("%f, %f, %f\n", lv.edf.color.x, lv.edf.color.y, lv.edf.color.z);

					if (!context.options.mis)
						mis_w = 1.0f;

					out_f = out_f_L * out_f_s;

					// cache the connecting bsdf's
					f_path.v_E(t-1) = cugar::Vector4f(out_f_s, 0.0f);
					f_path.v_L(s-1) = cugar::Vector4f(out_f_L, 0.0f);

					// compute the G factor of the old path
					const float old_G = old_path.G(s-1, renderer);

					// compute the acceptance rate factors we need
					context.Q_old[chain_id] *= old_f_path.v_E(t-1) * old_f_path.v_L(s-1) * old_G;
					context.Q_new[chain_id] *=     f_path.v_E(t-1) *     f_path.v_L(s-1) * out_G;
					context.discard_invalid( chain_id );

					cugar::Vector4f out_w = cugar::Vector4f( ev.alpha * lv.alpha * out_f * mis_w, 0.0f );

					if (valid_non_zero_sample( out_w.xyz() ))
					{
						// enqueue the output ray
						Ray out_ray;
						out_ray.origin	= ev.geom.position + ev.in * SHADOW_BIAS; // move the origin slightly towards the viewer
						out_ray.dir		= (lv.geom.position - out_ray.origin); //out;
						out_ray.tmin	= SHADOW_TMIN;
						out_ray.tmax	= 0.9999f;

						const PixelInfo out_pixel = in_bounce ?
							pixel_info :										// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, FBufferDesc::DIRECT_C);	// otherwise (i.e. this is the first bounce) choose the direct-lighting output channel

						context.shadow_queue.warp_append(
							out_pixel,
							out_ray,
							out_w,
							in_bounce );

						accumulated_path = true;
					}
					else
					{
						// kill the new sample
						invalidate_sample = true;
					}
				}
				else
				{
					// kill the new sample
					invalidate_sample = true;
				}
			}

			// accumulate the emissive component along the incoming direction
			if (in_bounce + 2 == t && s == 0) // NOTE: in_bounce + 2 represents the technique, i.e. the number of eye subpath vertices
			{
				// evaluate the edf's output along the incoming direction
				VertexGeometry	light_vertex_geom;
				float			light_pdf;
				Edf				light_edf;

				// recover the pdf and EDF information
				context.mesh_light.map(ev.geom_id.prim_id, ev.geom_id.uv, &light_vertex_geom, &light_pdf, &light_edf);

				const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, ev.in);

				// cache the bsdf corresponding to the light vertex
				f_path.v_L(0) = cugar::Vector4f(f_L, 0.0f);

				// compute the acceptance rate factors we need
				context.Q_old[chain_id] *= old_f_path.v_L(0);
				context.Q_new[chain_id] *=     f_path.v_L(0);
				context.discard_invalid( chain_id );

				float mis_w = 1.0f;
				if (context.options.mis)
				{
					// compute the MIS weight
					const float p_L = light_edf.p(light_vertex_geom, light_vertex_geom.position, ev.in, cugar::kProjectedSolidAngle);
					const float pGp = p_L * light_pdf;
					const float prev_pGp = ev.prev_pG * p_L;
					mis_w = mis_selector(
							0, ev.depth + 2,
							(ev.depth == 0 || pGp == 0.0f) ? 1.0f :
							bpt_mis(pGp, prev_pGp, ev.pGp_sum));
				}

				const cugar::Vector4f out_w = cugar::Vector4f(w.xyz() * f_L * mis_w, 0.0f);

				if (valid_non_zero_sample( out_w.xyz() ))
					accumulated_value = out_w;
				else
				{
					// kill the new sample
					invalidate_sample = true;
				}

				accumulated_path = false;
			}

			// trace a bounce ray
			if (in_bounce + 2 < t) // NOTE: in_bounce + 2 represents the technique, i.e. the number of eye subpath vertices
			{
				// fetch the randoms we need
				cugar::Vector3f samples;
				for (uint32 i = 0; i < 3; ++i)
					samples[i] = context.u_m(chain_id, (s + t - in_bounce - 2) * 3 + i); // remember we need 6 coordinates just to sample the second vertex

				const VertexGeometryId old_v	  = old_path.v_E(in_bounce + 1);
				const VertexGeometryId old_v_next = old_path.v_E(in_bounce + 2);

				// compute the outgoing direction in the old path
				const cugar::Vector3f out_old = cugar::normalize(
					interpolate_position(renderer.mesh, old_v_next) -
					interpolate_position(renderer.mesh, old_v));

				cugar::Vector3f out = out_old;

				const float u_p = context.enable_mutations ?
					context.u_m(chain_id, (s + t - in_bounce - 2) * 3 + 2) :
					2.0f; // a value higher than 1

				if (u_p < context.options.exp_perturbations)
				{
					// apply a spherical perturbation
					out =  spherical_perturbation(ev.geom, out_old, samples.xy(), context.options.perturbation_radius);
				}
				else if (u_p < context.options.H_perturbations
							 + context.options.exp_perturbations)
				{
					// reconstruct the old vertex
					const EyeVertex old_ev = reconstruct_vertex<EyeVertex>( old_path.v_E(in_bounce), old_v, in_bounce, renderer );

					// apply a H-perturbation
					out = H_perturbation(old_ev, out_old, ev, samples.xy(), context.options.perturbation_radius);
				}

				// evaluate the bsdf
				cugar::Vector3f f = ev.bsdf.f(ev.geom, ev.in, out);

				// cache this vertex' bsdf
				f_path.v_E(in_bounce + 1) = cugar::Vector4f(f, 0.0f);

				// update the acceptance rate factors
				if (u_p < context.options.exp_perturbations)
				{
					context.Q_old[chain_id] *= old_f_path.v_E(in_bounce + 1) * cos_theta( interpolate_normal(renderer.mesh, old_v), out_old );
					context.Q_new[chain_id] *=     f_path.v_E(in_bounce + 1) * cos_theta( ev.geom.normal_s,                         out );
					context.discard_invalid(chain_id);
				}
				else if (u_p < context.options.H_perturbations
							 + context.options.exp_perturbations)
				{
					// compute the geometric portion of the acceptance ratio
					const EyeVertex old_ev = reconstruct_vertex<EyeVertex>( old_path.v_E(in_bounce), old_v, in_bounce, renderer );

					context.Q_old[chain_id] *= old_f_path.v_E(in_bounce + 1) * H_perturbation_geometric_density( old_ev, out_old ) * cos_theta( old_ev.geom.normal_s, out_old );
					context.Q_new[chain_id] *=     f_path.v_E(in_bounce + 1) * H_perturbation_geometric_density( ev, out         ) * cos_theta( ev.geom.normal_s,	  out );
					context.discard_invalid(chain_id);
				}
				else
				{
					context.Q_old[chain_id] *= old_f_path.v_E(in_bounce + 1) * cos_theta( interpolate_normal(renderer.mesh, old_v), out_old );
					context.Q_new[chain_id] *=     f_path.v_E(in_bounce + 1) * cos_theta( ev.geom.normal_s,                         out );
					context.discard_invalid( chain_id );
				}

				// we use the fourth component to track parts of the acceptance rate
				const cugar::Vector4f out_w = cugar::Vector4f(f * w.xyz(), w.w);

				if (valid_non_zero_sample( out_w.xyz() ))
				{
					const uint32 channel = FBufferDesc::COMPOSITED_C;

					// enqueue the output ray
					Ray out_ray;
					out_ray.origin	= ev.geom.position;
					out_ray.dir		= out;
					out_ray.tmin	= 1.0e-4f;
					out_ray.tmax	= 1.0e8f;

					const PixelInfo out_pixel = in_bounce ?
						pixel_info :									// if this sample is a secondary bounce, use the previously selected channel
						PixelInfo(pixel_info.pixel, channel);			// otherwise (i.e. this is the first bounce) choose the output channel for the rest of the path

					const float p_proj = ev.bsdf.p(ev.geom, ev.in, out, cugar::kProjectedSolidAngle);
					const TempPathWeights out_path_weights( ev, out, p_proj );

					scatter_queue.warp_append(
						out_pixel, out_ray,
						out_w,
						in_bounce + 1,
						out_path_weights );	// only needed for MIS

					accumulated_path = true;
				}
				else
				{
					// kill the new sample
					invalidate_sample = true;
				}
			}
		}
		else
		{
			// hit the environment - perform sky lighting
			//

			// kill the new sample
			invalidate_sample = true;
		}

		if (invalidate_sample)
		{
			// kill the new sample
			context.discard( chain_id );
		}

		if (accumulated_path == false)
		{
			// perform the MH acceptance-rejection test
			accept_reject_accumulate(chain_id, s, t, accumulated_value, context, renderer);
		}
	}

	FERMAT_DEVICE
	void solve_occlusion_mlt(MLTContext& context, RenderingContextView& renderer, const uint32 task_id)
	{
		const PixelInfo		  pixel_info	= context.shadow_queue.pixels[task_id];
		const Hit			  hit			= context.shadow_queue.hits[task_id];
		const cugar::Vector4f w				= context.shadow_queue.weights[task_id];

		const uint32 chain_id = pixel_info.pixel;
		const uint32 s = context.st[chain_id].z;
		const uint32 t = context.st[chain_id].w;

		const float vis = (hit.t < 0.0f) ? 1.0f : 0.0f;

		// update the acceptance rate with visibility
		context.Q_new[chain_id] *= vis;

		// perform the MH acceptance-rejection test
		accept_reject_accumulate(chain_id, s, t, w * vis, context, renderer);
	}

	//------------------------------------------------------------------------------

} // anonymous namespace
