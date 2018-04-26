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

#include <cmlt.h>
#include <renderer.h>
#include <optix_prime/optix_primepp.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/sampling/random.h>
#include <cugar/sampling/distributions.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_context.h>
#include <bpt_control.h>
#include <bpt_samplers.h>
#include <random_sequence.h>
#include <path_inversion.h>
#include <ray_queues.h>
#include <vector>

#define SHIFT_RES	256u


#define DEBUG_PIXEL (714 + 66*1600)

namespace {

	///@addtogroup Fermat
	///@{

	///@addtogroup CMLTModule
	///@{

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float bounded_access(const float* vec, const uint32 i) { return (i == uint32(-1)) ? 0.0f : vec[i]; }

	FERMAT_HOST_DEVICE
	uint32 chain_coordinate_index(const uint32 n_chains, const uint32 idx, const uint32 dim)
	{
		return dim*n_chains + idx;
	}

	struct StridedRandoms
	{
		FERMAT_HOST_DEVICE
		StridedRandoms(const float* _u, const uint32 _stride) : u(_u), stride(_stride) {}

		FERMAT_HOST_DEVICE
		float next()
		{
			const float r = *u; u += stride;
			return r;
		}

		const float* u;
		uint32		 stride;
	};

	///@} CMLT
	///@} Fermat

} // anonymous namespace

///@addtogroup Fermat
///@{

///@addtogroup CMLTModule
///@{

	/// The CUDA context class for the CMLT renderer
	///
	struct CMLTContext : BPTContextBase
	{
		CMLTContext(
			CMLT&					_cmlt,
			const RendererView&		_renderer) :
			BPTContextBase(
				_renderer,
				_cmlt.m_light_vertices.view(),
				_cmlt.m_queues.view(_cmlt.m_n_init_paths, _cmlt.m_n_init_light_paths)),
			options					(_cmlt.m_options),
			sequence				(_cmlt.m_sequence.view()),
			connections_value		(_cmlt.m_connections_value.ptr()),
			connections_index		(_cmlt.m_connections_index.ptr()),
			connections_counter		(_cmlt.m_connections_counter.ptr()),
			st_norms				(_cmlt.m_st_norms.ptr()),
			st_norms_cdf			(_cmlt.m_st_norms_cdf.ptr()),
			seeds					(_cmlt.m_seeds.ptr()),
			st						(reinterpret_cast<char4*>(_cmlt.m_seeds.ptr())), // NOTE: aliased to seeds!
			mut_u					(_cmlt.m_mut_u.ptr()),
			light_u					(_cmlt.m_light_u.ptr()),
			eye_u					(_cmlt.m_eye_u.ptr()),
			vertices_l				(_cmlt.m_vertices.ptr()),
			vertices_e				(vertices_l		+ _cmlt.m_options.n_chains * (_cmlt.m_options.max_path_length + 1)),
			mut_vertices_l			(vertices_e		+ _cmlt.m_options.n_chains * (_cmlt.m_options.max_path_length + 1)),
			mut_vertices_e			(mut_vertices_l + _cmlt.m_options.n_chains * (_cmlt.m_options.max_path_length + 1)),
			path_value				(_cmlt.m_path_value.ptr()),
			path_pdf				(_cmlt.m_path_pdf.ptr()),
			new_path_value			(_cmlt.m_path_value.ptr() + _cmlt.m_options.n_chains),
			new_path_st				(reinterpret_cast<char2*>(_cmlt.m_path_pdf.ptr() + _cmlt.m_options.n_chains)),
			rejections				(_cmlt.m_rejections.ptr()),
			n_chains				(_cmlt.m_options.n_chains),
			mutation_type			(PerturbedPrimaryCoords::CauchyPerturbation),
			enable_accumulation		(true)
		{}

		CMLTOptions			options;
		TiledSequenceView	sequence;
		float4*				connections_value;
		uint4*				connections_index;
		uint32*				connections_counter;
		float*				st_norms;
		float*				st_norms_cdf;
		uint32*				seeds;
		char4*				st;
		float*				mut_u;
		float*				light_u;
		float*				eye_u;
		VertexGeometryId*	vertices_l;
		VertexGeometryId*	vertices_e;
		VertexGeometryId*	mut_vertices_l;
		VertexGeometryId*	mut_vertices_e;
		float4*				new_path_value;
		char2*				new_path_st;
		float4*				path_value;
		float*				path_pdf;
		uint32*				rejections;
		uint32				n_chains;
		float				pdf_norm;

		PerturbedPrimaryCoords::Type	mutation_type;
		bool							enable_accumulation;

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		float u_ar(const uint32 chain_id) const { return mut_u ? mut_u[chain_coordinate_index(n_chains, chain_id, (options.max_path_length + 1) * 3 * 2)] : 0.5f; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		float& u_L(const uint32 chain_id, const uint32 dim) { return light_u[chain_coordinate_index(n_chains, chain_id, dim)]; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		float& u_E(const uint32 chain_id, const uint32 dim) { return eye_u[chain_coordinate_index(n_chains, chain_id, dim)]; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		PerturbedPrimaryCoords light_primary_coords() const
		{
			return PerturbedPrimaryCoords(
				n_chains,
				light_u, 0u,
				mut_u, 0u,
				options.light_perturbations ? mutation_type : PerturbedPrimaryCoords::Null,
				options.perturbation_radius);
		}
		
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		PerturbedPrimaryCoords eye_primary_coords() const
		{
			return PerturbedPrimaryCoords(
				n_chains,
				eye_u, 0u,
				mut_u, (options.max_path_length + 1),
				options.eye_perturbations ? mutation_type : PerturbedPrimaryCoords::Null,
				options.perturbation_radius);
		}
	};

///@} CMLTModule
///@} Fermat
	
namespace { // anonymous namespace

	///@addtogroup Fermat
	///@{

	///@addtogroup CMLTModule
	///@{

	struct CMLTPresamplingBPTConfig : BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		CMLTPresamplingBPTConfig(CMLTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kAll,
				_context.options.single_connection ? VertexOrdering::kRandomOrdering : VertexOrdering::kPathOrdering,
				VertexSampling::kAll,
				_context.options.rr) {}
	};

	struct CMLTChainBPTConfig : BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		CMLTChainBPTConfig(CMLTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kEnd,
				VertexOrdering::kPathOrdering,
				VertexSampling::kEnd,
				_context.options.rr),
				st(_context.st) {}

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		bool terminate_light_subpath(const uint32 path_id, const uint32 s) const { return s == st[path_id].z; }
		//bool terminate_light_subpath(const uint32 path_id, const uint32 s) const { return (st[path_id].z == 0) || s >= BPTConfigBase::max_path_length + 1; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		bool terminate_eye_subpath(const uint32 path_id, const uint32 t) const { return t == st[path_id].w; }
		//bool terminate_eye_subpath(const uint32 path_id, const uint32 t) const { return t + st[path_id].z >= BPTConfigBase::max_path_length + 1; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		bool store_light_vertex(const uint32 path_id, const uint32 s, const bool absorbed) const { return (s == st[path_id].z); }
		/*bool store_light_vertex(const uint32 path_id, const uint32 s, const bool absorbed) const
		{
			if (absorbed)
			{
				// store the new path length
				st[path_id].z = s;
			}
			return absorbed;
		}*/

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		bool perform_connection(const uint32 path_id, const uint32 t, const bool absorbed) const { return t == st[path_id].w && st[path_id].z > 0; }
		//bool perform_connection(const uint32 path_id, const uint32 t, const bool absorbed) const { return absorbed == true && st[path_id].z > 0; }

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		bool accumulate_emissive(const uint32 path_id, const uint32 t, const bool absorbed) const { return t == st[path_id].w && st[path_id].z == 0; }
		//bool accumulate_emissive(const uint32 path_id, const uint32 t, const bool absorbed) const { return absorbed == true && st[path_id].z == 0; }

		// store the compact vertex information
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		void visit_light_vertex(
			const uint32			path_id,
			const uint32			depth,
			const VertexGeometryId	v,
			CMLTContext&			context,
			RendererView&			renderer) const
		{
			context.mut_vertices_l[path_id + depth * context.n_chains] = v;
		}

		// store the compact vertex information
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		void visit_eye_vertex(
			const uint32			path_id,
			const uint32			depth,
			const VertexGeometryId	v_id,
			const EyeVertex&		v,
			CMLTContext&			context,
			RendererView&			renderer) const
		{
			context.mut_vertices_e[path_id + depth * context.n_chains] = v_id;
		}

		/*const*/ char4* st;
	};

	FERMAT_DEVICE FERMAT_FORCEINLINE
	void reject_accumulate(const uint32 chain_id, CMLTContext& context, RendererView& renderer)
	{
		// TODO: keep track of the old sink!
		const float old_pdf = context.path_pdf[chain_id];

		// perform the MH acceptance-rejection test
		const cugar::Vector2f old_uv(
			context.u_E(chain_id, 3 + 0),
			context.u_E(chain_id, 3 + 1));

		const uint32 old_pixel_x = cugar::quantize(old_uv.x, renderer.res_x);
		const uint32 old_pixel_y = cugar::quantize(old_uv.y, renderer.res_y);
		const uint32 old_pixel = old_pixel_x + old_pixel_y*renderer.res_x;

		if (context.enable_accumulation)
		{
			if (old_pdf > 0)
			{
				const float4 old_value = context.path_value[chain_id];

				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).x, context.pdf_norm * (old_value.x / old_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).y, context.pdf_norm * (old_value.y / old_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).z, context.pdf_norm * (old_value.z / old_pdf) / float(renderer.instance + 1));
			}
		}

		// increase the rejections counter
		context.rejections[chain_id]++;

		//if ((context.rejections[chain_id] % 20) == 0)
		//if (context.rejections[chain_id] > 20 && context.rejections[chain_id] > context.rejections[context.n_chains])
		//	printf("chain[%u,%u:%u] stuck for %u iterations (val: %f)\n", context.st[chain_id].z, context.st[chain_id].w, chain_id, context.rejections[chain_id], old_pdf);

		atomicMax( &context.rejections[context.n_chains], context.rejections[chain_id] );
	}

	FERMAT_DEVICE FERMAT_FORCEINLINE
	void accept_reject_accumulate(const uint32 chain_id, const uint32 s, const uint32 t, const cugar::Vector4f w, CMLTContext& context, RendererView& renderer)
	{
		// perform the MH acceptance-rejection test
		const float new_pdf = cugar::max_comp(w.xyz());
		const float old_pdf = context.path_pdf[chain_id];

		if (new_pdf == 0.0f)
		{
			reject_accumulate(chain_id, context, renderer);
			return;
		}

		PerturbedPrimaryCoords eye_primary_coords = context.eye_primary_coords();

		const cugar::Vector2f old_uv(
			eye_primary_coords.u(chain_id, 3 + 0),
			eye_primary_coords.u(chain_id, 3 + 1));

		const cugar::Vector2f new_uv(
			eye_primary_coords.perturbed_u(chain_id, 3 + 0),
			eye_primary_coords.perturbed_u(chain_id, 3 + 1));

		const uint32 old_pixel_x = cugar::quantize(old_uv.x, renderer.res_x);
		const uint32 old_pixel_y = cugar::quantize(old_uv.y, renderer.res_y);
		const uint32 old_pixel   = old_pixel_x + old_pixel_y*renderer.res_x;

		const uint32 new_pixel_x = cugar::quantize(new_uv.x, renderer.res_x);
		const uint32 new_pixel_y = cugar::quantize(new_uv.y, renderer.res_y);
		const uint32 new_pixel   = new_pixel_x + new_pixel_y*renderer.res_x;

		const float ar = old_pdf ? fminf(1.0f, new_pdf / old_pdf) : 1.0f;

		if (context.enable_accumulation)
		{
			if (old_pdf > 0)
			{
				const float4 old_value = context.path_value[chain_id];

				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).x, context.pdf_norm * (1.0f - ar) * (old_value.x / old_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).y, context.pdf_norm * (1.0f - ar) * (old_value.y / old_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).z, context.pdf_norm * (1.0f - ar) * (old_value.z / old_pdf) / float(renderer.instance + 1));
			}
			if (new_pdf > 0)
			{
				const float4 new_value = w;

				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).x, context.pdf_norm * ar * (new_value.x / new_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).y, context.pdf_norm * ar * (new_value.y / new_pdf) / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).z, context.pdf_norm * ar * (new_value.z / new_pdf) / float(renderer.instance + 1));
			}
		}

		// fetch the old st
		const uint32 s_old = context.st[chain_id].x;
		const uint32 t_old = context.st[chain_id].y;

		const float st_change_ratio = 1.0f; // assume it's symmetric
		const float st_norms_ratio  = (context.st_norms[s + t * (context.options.max_path_length + 2)] / context.st_norms[s_old + t_old * (context.options.max_path_length + 2)]) * st_change_ratio;

		if (context.u_ar(chain_id) * old_pdf < new_pdf * st_norms_ratio)
		{
			context.path_value[chain_id] = w;
			context.path_pdf[chain_id] = new_pdf;

			// write out the successful st
			context.st[chain_id] = make_char4(s, t, s, t);

			PerturbedPrimaryCoords light_primary_coords = context.light_primary_coords();

			// copy the successful mutation coordinates
			for (uint32 i = 0; i < (context.options.max_path_length + 1) * 3; ++i)
				light_primary_coords.u(chain_id, i) = light_primary_coords.perturbed_u(chain_id, i);

			// copy the successful mutation coordinates
			for (uint32 i = 0; i < (context.options.max_path_length + 1) * 3; ++i)
				eye_primary_coords.u(chain_id, i) = eye_primary_coords.perturbed_u(chain_id, i);

			// copy the compact vertex information
			for (uint32 i = 0; i < s; ++i)
				context.vertices_l[chain_id + i * context.n_chains] = context.mut_vertices_l[chain_id + i * context.n_chains];

			for (uint32 i = 0; i < t; ++i)
				context.vertices_e[chain_id + i * context.n_chains] = context.mut_vertices_e[chain_id + i * context.n_chains];

			// reset the rejections counter
			context.rejections[chain_id] = 0;
		}
		else
		{
			// increase the rejections counter
			context.rejections[chain_id]++;

			//if ((context.rejections[chain_id] % 20) == 0)
			//if (context.rejections[chain_id] > 20 && context.rejections[chain_id] > context.rejections[context.n_chains])
			//	printf("chain[%u,%u:%u] stuck for %u iterations (val: %f, %f)\n", s, t, chain_id, context.rejections[chain_id], old_pdf, new_pdf);

			atomicMax( &context.rejections[context.n_chains], context.rejections[chain_id] );
		}
	}

	///
	/// The \ref SampleSinkAnchor "Sample Sink" used by the BPT presampling/seeding pass
	///
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
			CMLTContext&			context,
			RendererView&			renderer)
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
	///
	/// The \ref SampleSinkAnchor "Sample Sink" used the MLT pass
	///
	struct AccumulateRejectSink : SampleSinkBase
	{
		FERMAT_HOST_DEVICE
		AccumulateRejectSink() {}

		FERMAT_HOST_DEVICE
		void sink(
			const uint32			channel,
			const cugar::Vector4f	value,
			const uint32			light_path_id,
			const uint32			eye_path_id,
			const uint32			s,
			const uint32			t,
			CMLTContext&			context,
			RendererView&			renderer)
		{
			context.new_path_value[ eye_path_id ]	= value;
			context.new_path_st[ eye_path_id ]		= make_char2(s,t);
		}
	};

	//------------------------------------------------------------------------------

	__global__
	void accept_reject_mlt_kernel(CMLTContext context, RendererView renderer)
	{
		const uint32 chain_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (chain_id < context.n_chains)
			accept_reject_accumulate(chain_id, context.new_path_st[chain_id].x, context.new_path_st[chain_id].y, context.new_path_value[chain_id], context, renderer);
	}

	void accept_reject_mlt(CMLTContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize));
		accept_reject_mlt_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	//------------------------------------------------------------------------------

	/// implement plain MMLT swaps
	///
	__global__
	void mmlt_swap_kernel(CMLTContext context, RendererView renderer)
	{
		const uint32 chain_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (chain_id < context.n_chains)
		{
			const uint32 s = context.st[chain_id].x;
			const uint32 t = context.st[chain_id].y;

			// setup the random stream for this chain
			StridedRandoms random(context.mut_u + chain_id, context.n_chains);

			// generate a candidate (s_new,t_new) pair with s_new + t_new = s + t
			FERMAT_ASSERT(s + t > 1);

		#if 0
			// select a technique at random
			const uint32 s_new = cugar::quantize(random.next(), s + t - 1);
			const float  st_change_ratio = 1.0f;
		#elif 1
			// perform a random walk on s
			const uint32 s_new = random.next() > 0.5f ? (t > 2 ? s + 1 : 0) : (s > 0 ? s - 1 : s + t - 2);
			const float  st_change_ratio = 1.0f;
		#else
			const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

			// propose a technique with the same path length k = s + t - 1 according to their CDF
			const uint32 k = s + t - 1;
			const float* st_norms_cdf = context.st_norms_cdf + k * (context.options.max_path_length + 2);
			const uint32 s_new = cugar::upper_bound_index( fminf( random.next(), one ), st_norms_cdf, k + 2);
			const float  st_change_ratio =
				(bounded_access(st_norms_cdf, s) - bounded_access(st_norms_cdf, s - 1)) /
				(bounded_access(st_norms_cdf, s_new) - bounded_access(st_norms_cdf, s_new - 1));
		#endif

			const uint32 t_new = s + t - s_new;
			FERMAT_ASSERT(s_new < s + t); // NOTE: it should be <= s + t, but we don't support t = 0
			FERMAT_ASSERT(s_new + t_new == s + t);

			// for now, make sure we never generate a technique with less than 2 eye vertices
			if (t < 2)
				return;

			// write the proposed s,t coordinates out
			context.st[chain_id].z = s_new;
			context.st[chain_id].w = t_new;
		}
	}

	/// implement plain MMLT swaps
	///
	void mmlt_swap(CMLTContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize));
		mmlt_swap_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	//------------------------------------------------------------------------------

	/// implement chart swaps
	///
	__global__
	void chart_swap_kernel(CMLTContext context, RendererView renderer)
	{
		const uint32 chain_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (chain_id < context.n_chains)
		{
			const uint32 s = context.st[chain_id].x;
			const uint32 t = context.st[chain_id].y;
			
			// setup the random stream for this chain
			StridedRandoms random(context.mut_u + chain_id, context.n_chains);

			// generate a candidate (s_new,t_new) pair with s_new + t_new = s + t
			FERMAT_ASSERT(s + t > 1);

		  #if 0
			// select a technique at random
			const uint32 s_new = cugar::quantize( random.next(), s + t - 1 );
			const float  st_change_ratio = 1.0f;
		  #elif 0
			// perform a random walk on s
			const uint32 s_new = random.next() > 0.5f ? (t > 2 ? s + 1 : 0) : (s > 0 ? s - 1 : s + t - 2);
			const float  st_change_ratio = 1.0f;
		  #else
			const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

			// propose a technique with the same path length k = s + t - 1 according to their CDF
			const uint32 k = s + t - 1;
			const float* st_norms_cdf = context.st_norms_cdf + k * (context.options.max_path_length + 2);
			const uint32 s_new = cugar::min( cugar::upper_bound_index( fminf( random.next(), one ), st_norms_cdf, k + 2 ), k /*+ 1*/ ); // note: it should be k + 1, but we don't support t = 0
			const float  st_change_ratio =
				(bounded_access(st_norms_cdf, s) - bounded_access(st_norms_cdf, s - 1)) /
				(bounded_access(st_norms_cdf, s_new) - bounded_access(st_norms_cdf, s_new - 1));
		  #endif
			const uint32 t_new = s + t - s_new;
			FERMAT_ASSERT(s_new < s + t); // NOTE: it should be <= s + t, but we don't support t = 0
			FERMAT_ASSERT(s_new + t_new == s + t);
			
			// for now, make sure we never generate a technique with less than 2 eye vertices
			if (t_new < 2)
				return;
			
			// setup the path wrapper
			BidirPath path( s, t, context.vertices_l + chain_id, context.vertices_e + chain_id, context.n_chains );

			float u_new[32];
			float u_old[32];
			float pdf_new;
			
			const float st_norms_ratio = (context.st_norms[s_new + t_new * (context.options.max_path_length + 2)] / context.st_norms[s + t * (context.options.max_path_length + 2)]) * st_change_ratio;

			if (s_new > s)
			{
				// invert the [s,s_new) vertices of the light subpath
				if (invert_light_subpath(path, s, s_new, u_new, &pdf_new, renderer, random) == false)
					return;

				// compute the eye subpath inversion pdf for the vertices [t_new,t)
				for (uint32 i = t_new; i < t; ++i)
				{
					u_old[(i - t_new) * 3 + 0] = context.u_E(chain_id, i * 3 + 0);
					u_old[(i - t_new) * 3 + 1] = context.u_E(chain_id, i * 3 + 1);
					u_old[(i - t_new) * 3 + 2] = context.u_E(chain_id, i * 3 + 2);
				}

				const float pdf_old = eye_subpath_inversion_pdf(path, t_new, t, u_old, renderer);

				//if (random.next() < (pdf_new / pdf_old) * st_norms_ratio)
				if (random.next() < (pdf_old / pdf_new) * st_norms_ratio)
				{
					// accept the proposal
					context.st[chain_id] = make_char4( s_new, t_new, s_new, t_new );

					for (uint32 i = s; i < s_new; ++i)
					{
						context.u_L(chain_id, i * 3 + 0) = u_new[(i - s) * 3 + 0];
						context.u_L(chain_id, i * 3 + 1) = u_new[(i - s) * 3 + 1];
						context.u_L(chain_id, i * 3 + 2) = u_new[(i - s) * 3 + 2];
					}
				}
			}
			else if (t_new > t)
			{
				// invert the [t,t_new) vertices of the eye subpath
				if (invert_eye_subpath(path, t, t_new, u_new, &pdf_new, renderer, random) == false)
					return;

				// compute the light subpath inversion pdf for the vertices [s_new,s)
				for (uint32 i = s_new; i < s; ++i)
				{
					u_old[(i - s_new) * 3 + 0] = context.u_L(chain_id, i * 3 + 0);
					u_old[(i - s_new) * 3 + 1] = context.u_L(chain_id, i * 3 + 1);
					u_old[(i - s_new) * 3 + 2] = context.u_L(chain_id, i * 3 + 2);
				}

				const float pdf_old = light_subpath_inversion_pdf(path, s_new, s, u_old, renderer);

				//if (random.next() < (pdf_new / pdf_old) * st_norms_ratio)
				if (random.next() < (pdf_old / pdf_new) * st_norms_ratio)
				{
					// accept the proposal
					context.st[chain_id] = make_char4(s_new, t_new, s_new, t_new);

					for (uint32 i = t; i < t_new; ++i)
					{
						context.u_E(chain_id, i * 3 + 0) = u_new[(i - t) * 3 + 0];
						context.u_E(chain_id, i * 3 + 1) = u_new[(i - t) * 3 + 1];
						context.u_E(chain_id, i * 3 + 2) = u_new[(i - t) * 3 + 2];
					}
				}
			}
		}
	}

	/// implement chart swaps
	///
	void chart_swap(CMLTContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize));
		chart_swap_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	//------------------------------------------------------------------------------

	/// sample the seed paths
	///
	__global__
	void sample_seeds_kernel(const float random_shift, const uint32 n_connections, const float* connections_cdf, const uint4* connections_index, const uint32 n_seeds, uint32* seeds, uint32* st_counters, const uint32 max_path_length)
	{
		const uint32 seed_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (seed_id < n_seeds)
		{
			const float r = (seed_id + random_shift) / float(n_seeds);

			const uint32 seed = cugar::upper_bound_index( r * connections_cdf[n_connections-1], connections_cdf, n_connections);

			const uint4 connection = connections_index[seed];

			seeds[seed_id] = seed | (connection.z << 24) | (connection.w << 28);

			// keep stats
			atomicAdd(st_counters + connection.z + connection.w*(max_path_length + 2), 1u);
		}
	}

	/// sample the seed paths
	///
	void sample_seeds(const float random_shift, const uint32 n_connections, const float* connections_cdf, const uint4* connections_index, const uint32 n_seeds, uint32* seeds, uint32* st_counters, const uint32 max_path_length)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		sample_seeds_kernel<<< gridSize, blockSize >>>(random_shift, n_connections, connections_cdf, connections_index, n_seeds, seeds, st_counters, max_path_length);
	}

	//------------------------------------------------------------------------------

	/// recover the primary space coordinates of the sampled seed paths
	///
	__global__
	void recover_primary_coordinates_kernel(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, CMLTContext context, RendererView renderer)
	{
		const uint32 seed_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (seed_id < n_seeds)
		{
			const uint32 seed = seeds[seed_id];

			const uint4 connection = context.connections_index[seed & 0xFFFFFF];

			const uint32 light_idx	= connection.x;
			const uint32 eye_idx	= connection.y;
			const uint32 s			= connection.z;
			const uint32 t			= connection.w;

			TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

			for (uint32 i = 0; i < s; ++i)
			{
				if (i == 0)
				{
					// the first vertex might be somewhat special
					if (context.options.use_vpls)
						context.u_L(seed_id, i * 3 + 0) = (float(light_idx) + light_primary_coords.sample(light_idx, i, 0)) / float(n_lights);
					else
						context.u_L(seed_id, i * 3 + 0) = light_primary_coords.sample(light_idx, i, 0);

					context.u_L(seed_id, i * 3 + 1) = light_primary_coords.sample(light_idx, i, 1);
					context.u_L(seed_id, i * 3 + 2) = light_primary_coords.sample(light_idx, i, 2);
				}
				else
				{
					// fetch the regular coordinates
					context.u_L(seed_id, i * 3 + 0) = light_primary_coords.sample(light_idx, i, 0);
					context.u_L(seed_id, i * 3 + 1) = light_primary_coords.sample(light_idx, i, 1);
					context.u_L(seed_id, i * 3 + 2) = light_primary_coords.sample(light_idx, i, 2);
				}
			}

			PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.res_x, renderer.res_y);

			for (uint32 i = 0; i < t; ++i)
			{
				if (i == 0)
				{
					// we set the lens sample to (0,0,0)
					context.u_E(seed_id, i * 3 + 0) = 0;
					context.u_E(seed_id, i * 3 + 1) = 0;
					context.u_E(seed_id, i * 3 + 2) = 0;
				}
				else
				{
					// fetch the regular coordinates
					context.u_E(seed_id, i * 3 + 0) = eye_primary_coords.sample(eye_idx, i, 0);
					context.u_E(seed_id, i * 3 + 1) = eye_primary_coords.sample(eye_idx, i, 1);
					context.u_E(seed_id, i * 3 + 2) = eye_primary_coords.sample(eye_idx, i, 2);
				}
			}

			// write the st info
			context.st[seed_id] = make_char4(s, t, s, t);
		}
	}
	/// recover the primary space coordinates of the sampled seed paths
	///
	void recover_primary_coordinates(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, CMLTContext context, RendererView renderer)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		recover_primary_coordinates_kernel <<< gridSize, blockSize >>>(n_seeds, seeds, n_lights, context, renderer);
	}

	///@} CMLT
	///@} Fermat

} // anonymous namespace

CMLT::CMLT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void CMLT::init(int argc, char** argv, Renderer& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	// if we perform a single connection, RR must be enabled
	if (m_options.single_connection)
		m_options.rr = true;

	// TODO: re-enable when light tracing is implemented
	m_options.light_tracing = 0.0f;

	// compute how long our chains are
	const uint32 chain_length = (m_options.spp * n_pixels) / m_options.n_chains;

	// check whether how many actual skips we can afford
	m_options.startup_skips = cugar::min(m_options.startup_skips, chain_length - 1);

	fprintf(stderr, "  CMLT settings:\n");
	fprintf(stderr, "    spp            : %u\n", m_options.spp);
	fprintf(stderr, "    chains         : %u\n", m_options.n_chains);
	fprintf(stderr, "    chain-length   : %u\n", chain_length);
	fprintf(stderr, "    startup-skips  : %u\n", m_options.startup_skips);
	fprintf(stderr, "    charted-swaps  : %u\n", m_options.swap_frequency);
	fprintf(stderr, "    mmlt-swaps     : %u\n", m_options.mmlt_frequency);
	fprintf(stderr, "    single-conn    : %u\n", m_options.single_connection);
	fprintf(stderr, "    RR             : %u\n", m_options.rr);
	fprintf(stderr, "    path-length    : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee     : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf    : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee   : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf  : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    light-tracing  : %u\n", m_options.light_tracing ? 1 : 0);

	// compute the number of light paths
	const uint32 n_light_paths = n_pixels;
	const uint32 n_chains = m_options.n_chains;

	fprintf(stderr, "  creatign mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.m_mesh_lights.init( n_light_paths, renderer.m_mesh.view(), renderer.m_mesh_d.view(), renderer.m_texture_views_h.ptr(), renderer.m_texture_views_d.ptr() );

	fprintf(stderr, "  creatign mesh lights... done\n");

	const uint32 queue_size = std::max(n_pixels, n_light_paths) * (2 + m_options.max_path_length);

	// pre-alloc all buffers
	m_queues.alloc(n_pixels, n_light_paths, m_options.max_path_length);

	// build the set of shifts
	const uint32 n_dimensions = (m_options.max_path_length + 1) * 2 * 6;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	m_sequence.setup(n_dimensions, SHIFT_RES);

	fprintf(stderr, "  allocating light vertex storage... started (%u paths, %u vertices)\n", n_light_paths, n_light_paths * m_options.max_path_length);
	m_light_vertices.alloc(n_light_paths, n_light_paths * m_options.max_path_length);
	fprintf(stderr, "  allocating light vertex storage... done\n");

	if (m_options.use_vpls)
		cudaMemcpy(m_light_vertices.vertex.ptr(), cugar::raw_pointer(renderer.m_mesh_lights.vpls), sizeof(VPL)*n_light_paths, cudaMemcpyDeviceToDevice);

	fprintf(stderr, "  allocating bpt connections... started\n");
  #if SINGLE_CONNECTION
	m_connections_value.alloc(n_pixels * (m_options.max_path_length + 1));
	m_connections_index.alloc(n_pixels * (m_options.max_path_length + 1));
	m_connections_cdf.alloc(n_pixels * (m_options.max_path_length + 1));
  #else
	m_connections_value.alloc(n_pixels * m_options.max_path_length * (m_options.max_path_length + 1));
	m_connections_index.alloc(n_pixels * m_options.max_path_length * (m_options.max_path_length + 1));
	m_connections_cdf.alloc(n_pixels * m_options.max_path_length * (m_options.max_path_length + 1));
  #endif
	m_connections_counter.alloc(1);
	m_seeds.alloc(n_chains);
	fprintf(stderr, "  allocating bpt connections... done\n");

	fprintf(stderr, "  allocating chain storage... started\n");
	m_mut_u.alloc(n_chains*((m_options.max_path_length + 1) * 3 * 2 + 2));
	m_light_u.alloc(n_chains*(m_options.max_path_length + 1) * 3);
	m_eye_u.alloc(n_chains*(m_options.max_path_length + 1) * 3);
	m_path_value.alloc(n_chains * 2);
	m_path_pdf.alloc(n_chains * 2);
	m_rejections.alloc(n_chains + 1);
	m_vertices.alloc(n_chains*(m_options.max_path_length + 1) * 4);
	m_st_counters.alloc((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	m_st_norms.alloc((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	m_st_norms_cdf.alloc((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	fprintf(stderr, "  allocating chain storage... done\n");

	m_n_init_light_paths = n_light_paths;
	m_n_init_paths		 = n_pixels;
}

struct max_comp_functor
{
	typedef float4	argument_type;
	typedef float   result_type;

	FERMAT_HOST_DEVICE
	float operator() (const argument_type c) { return cugar::max3(c.x, c.y, c.z); }
};

// sample seed paths from the stored bidirectional connections
//
void CMLT::sample_seeds(const uint32 n_chains)
{
	cugar::device_vector<uint8> temp_storage;

	// compute the connections CDF
	cugar::inclusive_scan<cugar::device_tag>(
		m_n_connections,
		thrust::make_transform_iterator(m_connections_value.ptr(), max_comp_functor()),
		m_connections_cdf.ptr(),
		thrust::plus<float>(),
		temp_storage);

	// zero out the stats
	cudaMemset(m_st_counters.ptr(), 0x00, sizeof(uint32)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2));

	// resample n_chains of them
	::sample_seeds(m_random.next(), m_n_connections, m_connections_cdf.ptr(), m_connections_index.ptr(), n_chains, m_seeds.ptr(), m_st_counters.ptr(), m_options.max_path_length);

	// and sort them
	cugar::radix_sort<cugar::device_tag>(n_chains, m_seeds.ptr(), temp_storage);

	m_image_brightness = m_connections_cdf[m_n_connections - 1] / float(m_n_init_paths);
}

// recover primary sample space coordinates of the sampled light and eye subpaths
//
void CMLT::recover_primary_coordinates(CMLTContext& context, RendererView& renderer_view)
{
	::recover_primary_coordinates(context.n_chains, m_seeds.ptr(), m_n_init_light_paths, context, renderer_view );
}

// build a CDF on st_norms
void CMLT::build_st_norms_cdf()
{
	float st_norms[1024];
	float st_norms_cdf[1024] = { 0 };

	cudaMemcpy(st_norms, m_st_norms.ptr(), sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);

	for (uint32 k = 1; k <= m_options.max_path_length; ++k)
	{
		// consider all the (k+2) paths of length k
		for (uint32 s = 0; s < k + 2; ++s)
		{
			const uint32 t = k + 1 - s;

			const float norm = st_norms[s + t*(m_options.max_path_length + 2)] / float(m_n_init_paths);

			st_norms_cdf[k * (m_options.max_path_length + 2) + s] = norm;
		}

		// compute the cumulative sum
		for (uint32 i = 1; i < k + 2; ++i)
			st_norms_cdf[k * (m_options.max_path_length + 2) + i] +=
			st_norms_cdf[k * (m_options.max_path_length + 2) + i - 1];

		// and normalize it
		const float inv_sum = 1.0f / st_norms_cdf[k * (m_options.max_path_length + 2) + k + 1];
		for (uint32 i = 0; i < k + 2; ++i)
			st_norms_cdf[k * (m_options.max_path_length + 2) + i] *= inv_sum;

		//for (uint32 i = 0; i < k + 2; ++i)
		//	fprintf(stderr, "cdf[%u][%u] = %f\n", k, i, st_norms_cdf[k * (m_options.max_path_length + 2) + i]);
		//fgetc(stdin);
	}
	cudaMemcpy(m_st_norms_cdf.ptr(), st_norms_cdf, sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyHostToDevice);
}

void CMLT::render(const uint32 instance, Renderer& renderer)
{
	// clear the global timer at instance zero
	if (instance == 0)
		m_time = 0.0f;

	// fetch the renderer view
	RendererView renderer_view = renderer.view(instance);

	// pre-multiply the previous frame for blending
	renderer.multiply_frame(float(instance) / float(instance + 1));

	cugar::Timer timer;

	float chart_swap_time	   = 0.0f;
	float presampling_time	   = 0.0f;
	float seed_resampling_time = 0.0f;

	const uint2	 res			= renderer.res();
	const uint32 n_pixels		= res.x * res.y;
	const uint32 n_chains		= m_options.n_chains;
	const uint32 chain_length	= (m_options.spp * n_pixels) / n_chains;

	// initialize the sampling sequence for this frame
	m_sequence.set_instance(instance);

	// setup our BPT context
	CMLTContext context(*this, renderer_view);

	bool do_reseeding = (instance % m_options.reseeding) == 0;

	// perform the BPT presampling pass
	if (do_reseeding)
	{
		timer.start();

		// reset the connections counter
		cudaMemset(context.connections_counter, 0x00, sizeof(uint32));

		// zero out the stats
		cudaMemset(m_st_norms.ptr(), 0x00, sizeof(float)*(context.options.max_path_length + 2)*(context.options.max_path_length + 2));

		TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

		PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.m_res_x, renderer.m_res_y);

		CMLTPresamplingBPTConfig config(context);

		ConnectionsSink connections_sink;

		bpt::sample_paths(
			m_n_init_paths,
			m_n_init_light_paths,
			eye_primary_coords,
			light_primary_coords,
			connections_sink,
			context,
			config,
			renderer,
			renderer_view);

		timer.stop();
		presampling_time = timer.seconds();

		timer.start();

		// fetch the number of connections we found
		cudaMemcpy(&m_n_connections, context.connections_counter, sizeof(uint32), cudaMemcpyDeviceToHost);
		//fprintf(stderr, "  n_connections: %u\n", m_n_connections);

		// exit if we didn't find any valid path
		if (m_n_connections == 0)
			return;

		sample_seeds(n_chains);

		// print out the chain stats
		if (instance == 0)
		{
			fprintf(stderr, "  image brightness: %f\n", m_image_brightness);
			fprintf(stderr, "  st chains\n");
			uint32 st_counters[1024];
			float  st_norms[1024];
			cudaMemcpy(st_counters, m_st_counters.ptr(), sizeof(uint32)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);
			cudaMemcpy(st_norms, m_st_norms.ptr(), sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);
			for (uint32 s = 0; s < m_options.max_path_length; ++s)
			{
				for (uint32 t = 0; t < m_options.max_path_length + 2; ++t)
				{
					if (s + t <= m_options.max_path_length + 1)
						fprintf(stderr, "    [%u,%u] : %7u, %f\n", s, t, st_counters[s + t*(m_options.max_path_length + 2)], st_norms[s + t*(m_options.max_path_length + 2)] / float(m_n_init_paths));
				}
			}
		}

		// build a CDF on st_norms
		build_st_norms_cdf();

		// setup the normalization constant
		context.pdf_norm = m_image_brightness * float(res.x * res.y) / float((chain_length - context.options.startup_skips) * n_chains);

		// recover the primary coordinates of the selected seed paths
		recover_primary_coordinates(context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("recover primary coordinates"));

		// initialize the initial path pdfs to zero
		cudaMemset(context.path_pdf, 0x00, sizeof(float)*n_chains);

		// initialize the rejection counters to zero
		cudaMemset(context.rejections, 0x00, sizeof(uint32)*n_chains);

		timer.stop();
		seed_resampling_time = timer.seconds();
	}

	timer.start();

	// initialize the random generator
	DeviceRandomSequence sequence( cugar::hash(instance) );

	for (uint32 l = 0; l < chain_length; ++l)
	{
		// disable any kind of mutations for the first step, and enable them later on
		context.mutation_type = (l == 0 && do_reseeding) ? PerturbedPrimaryCoords::Null : PerturbedPrimaryCoords::CauchyPerturbation;

		// enable accumulation only after a few steps
		context.enable_accumulation = (l >= context.options.startup_skips) ? true : false;

		if (!do_reseeding || l > 0)
		{
			// generate all the random numbers needed for mutations
			sequence.next(n_chains*((context.options.max_path_length + 1) * 2 * 3 + 2), context.mut_u);
		}

		if (m_options.mmlt_frequency && l && ((l % m_options.mmlt_frequency) == 0))
		{
			// set all mutation components to zero, in order to keep path coordinates unchanged (being careful to leave the one used for the acceptance/rejection test out)
			//cudaMemset(context.mut_u, 0x00, n_chains*((m_options.max_path_length + 1) * 3));

			// propose a swap
			mmlt_swap(context, renderer_view);
		}

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
			
		// sample a set of bidirectional paths corresponding to our current primary coordinates
		CMLTChainBPTConfig config(context);

		PerturbedPrimaryCoords light_primary_coords = context.light_primary_coords();
		PerturbedPrimaryCoords eye_primary_coords	= context.eye_primary_coords();

		AccumulateRejectSink accept_reject_sink;

		// initialize the initial new path values to zero
		cudaMemset(context.new_path_value, 0x00, sizeof(float4)*n_chains);

		bpt::sample_paths(
			context.n_chains,
			context.n_chains,
			eye_primary_coords,
			light_primary_coords,
			accept_reject_sink,
			context,
			config,
			renderer,
			renderer_view,
			true);			// lazy shadows

		accept_reject_mlt( context, renderer_view );

		if (m_options.swap_frequency && l && ((l % m_options.swap_frequency) == 0))
		{
			cugar::ScopedTimer<float> chart_swap_timer( &chart_swap_time );

			// generate all the random numbers needed for mutations
			sequence.next(n_chains*((m_options.max_path_length + 1) * 3 * 2 + 2), context.mut_u);

			// perform a technique swap
			chart_swap(context, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("technique swap"));
		}
	}

	uint32 max_iter_stuck = 0;
	cudaMemcpy(&max_iter_stuck, context.rejections + n_chains, sizeof(uint32), cudaMemcpyDeviceToHost);

	timer.stop();
	const float mlt_time = timer.seconds();

	m_time += presampling_time + seed_resampling_time + mlt_time;

	fprintf(stderr, "\r  %.1fs (%.1fms = init: %.1fms, seed: %.1fms, mut: %.1fms, c-swaps: %.1fms, stuck: %u)        ",
		m_time,
		(presampling_time + seed_resampling_time + mlt_time) * 1000.0f,
		presampling_time * 1000.0f,
		seed_resampling_time * 1000.0f,
		mlt_time * 1000.0f,
		chart_swap_time * 1000.0f,
		max_iter_stuck);
}
