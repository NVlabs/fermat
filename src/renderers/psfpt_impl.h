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

#include <psfpt.h>
#include <renderer.h>
#include <rt.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/memory_arena.h>
#include <pathtracer_core.h>
#include <pathtracer_queues.h>
#include <pathtracer_kernels.h>
#include <psfpt_vertex_processor.h>


#define SHIFT_RES	256u

#define HASH_SIZE (64u * 1024u * 1024u)


namespace {

	typedef cugar::cuda::SyncFreeHashMap<uint64, uint32, 0xFFFFFFFFFFFFFFFFllu> HashMap;

	// a queue of references to PSF cells that will need to be blended in after path sampling
	//
	struct PSFRefQueue
	{
		float4*		weights_d;	// diffuse path weight
		float4*		weights_g;	// glossy path weight
		uint2*		pixels;
		uint32*		size;

		FERMAT_DEVICE
		void warp_append(const PixelInfo pixel, const PSFPTVertexProcessor::CacheInfo cache_slot, const float4 weight_d, const float4 weight_g)
		{
			const uint32 slot = cugar::cuda::warp_increment(size);

			weights_d[slot] = weight_d;
			weights_g[slot] = weight_g;

			pixels[slot] = make_uint2(pixel.packed, cache_slot.packed);
		}
	};

	// the internal path tracing context
	//
	template <typename TDirectLightingSampler>
	struct PSFPTContext : PTContextBase<PSFPTOptions>, PTContextQueues
	{
		PSFRefQueue	ref_queue;

		HashMap		psf_hashmap;
		float4*		psf_values;

		TDirectLightingSampler dl;
	};

	// initialize the RL storage for mesh VTLs
	void init(ClusteredRLStorage* vtls_rl, const MeshVTLStorage* mesh_vtls)
	{
		vtls_rl->init(
			VTL_RL_HASH_SIZE,
			mesh_vtls->get_bvh_clusters_count(),
			mesh_vtls->get_bvh_cluster_offsets());
	}
	// initialize the RL storage for mesh VTLs
	void init(AdaptiveClusteredRLStorage* vtls_rl, const MeshVTLStorage* mesh_vtls)
	{
		vtls_rl->init(
			VTL_RL_HASH_SIZE,
			mesh_vtls->get_bvh_nodes(),
			mesh_vtls->get_bvh_parents(),
			mesh_vtls->get_bvh_ranges(),
			mesh_vtls->get_bvh_clusters_count(),
			mesh_vtls->get_bvh_clusters(),
			mesh_vtls->get_bvh_cluster_offsets());
	}
	
	// the kernel blending/splatting PSF references into the framebuffer
	//
	template <typename TDirectLightingSampler>
	__global__
	void psf_blending_kernel(const uint32 in_queue_size, PSFPTContext<TDirectLightingSampler> context, RenderingContextView renderer, const float frame_weight)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.shadow_queue.size
		{
			typedef PSFPTVertexProcessor::CacheInfo CacheInfo;

			// fetch a reference from the ref queue
			const PixelInfo		  pixel_info = context.ref_queue.pixels[thread_id].x;
			const CacheInfo		  cache_info = context.ref_queue.pixels[thread_id].y;
			const cugar::Vector4f w_d = context.ref_queue.weights_d[thread_id];
			const cugar::Vector4f w_g = context.ref_queue.weights_g[thread_id];

			// check if it's valid
			if (cache_info.is_valid())
			{
				// dereference the hashmap cell
				const uint32 cache_slot = cache_info.pixel;

				cugar::Vector4f cache_value = context.psf_values[cache_slot];
								cache_value /= cache_value.w; // normalize

				// compue the total weight
				const cugar::Vector3f w =
					((pixel_info.comp & Bsdf::kDiffuseMask) ? w_d.xyz() : cugar::Vector3f(0.0f)) +
					((pixel_info.comp & Bsdf::kGlossyMask)  ? w_g.xyz() : cugar::Vector3f(0.0f));

				// add to the composited framebuffer
				add_in<false>(renderer.fb(FBufferDesc::COMPOSITED_C), pixel_info.pixel, cugar::min( cache_value.xyz() * w, context.options.firefly_filter ), frame_weight);

				// add to the diffuse channel, if the diffuse component is present
				if (pixel_info.comp & Bsdf::kDiffuseMask)
					add_in<true>(renderer.fb(FBufferDesc::DIFFUSE_C),     pixel_info.pixel, cache_value.xyz() * w_d.xyz(), frame_weight);

				// add to the glossy channel, if the glossy component is present
				if (pixel_info.comp & Bsdf::kGlossyMask)
					add_in<true>(renderer.fb(FBufferDesc::SPECULAR_C),    pixel_info.pixel, cache_value.xyz() * w_g.xyz(), frame_weight);
			}
		}
	}

	// dispatch the blending kernel
	//
	template <typename TDirectLightingSampler>
	void psf_blending(const uint32 in_queue_size, PSFPTContext<TDirectLightingSampler> context, RenderingContextView renderer)
	{
		if (!in_queue_size)
			return;

		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		psf_blending_kernel << < gridSize, blockSize >> > (in_queue_size, context, renderer, 1.0f / float(renderer.instance + 1));
	}

	// alloc all internal queues
	//
	void alloc_queues(
		PSFPTOptions			options,
		const uint32			n_pixels,
		PTRayQueue&				input_queue,
		PTRayQueue&				scatter_queue,
		PTRayQueue&				shadow_queue,
		PSFRefQueue&			ref_queue,
		cugar::memory_arena&	arena)
	{
		::alloc_queues( options, n_pixels, input_queue, scatter_queue, shadow_queue, arena );

		ref_queue.weights_d		= arena.alloc<float4>(n_pixels * (options.max_path_length + 1));
		ref_queue.weights_g		= arena.alloc<float4>(n_pixels * (options.max_path_length + 1));
		ref_queue.pixels		= arena.alloc<uint2>(n_pixels * (options.max_path_length + 1));
		ref_queue.size			= arena.alloc<uint32>(1);
	}

} // anonymous namespace

PSFPT::PSFPT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
	m_mesh_vtls = new MeshVTLStorage;
	m_vtls_rl = new VTLRLStorage;
}

void PSFPT::init(int argc, char** argv, RenderingContext& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	const char* nee_alg[] = { "mesh", "vpl", "rl" };

	fprintf(stderr, "  PSFPT settings:\n");
	fprintf(stderr, "    path-length     : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee      : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf     : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee    : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf   : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights  : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    direct lighting : %u\n", m_options.direct_lighting ? 1 : 0);
	fprintf(stderr, "    diffuse         : %u\n", m_options.diffuse_scattering ? 1 : 0);
	fprintf(stderr, "    glossy          : %u\n", m_options.glossy_scattering ? 1 : 0);
	fprintf(stderr, "    indirect glossy : %u\n", m_options.indirect_glossy ? 1 : 0);
	fprintf(stderr, "    RR              : %u\n", m_options.rr ? 1 : 0);
	fprintf(stderr, "    nee algorithm   : %s\n", nee_alg[ m_options.nee_type ]);
	fprintf(stderr, "    filter width    : %f\n", m_options.psf_width);
	fprintf(stderr, "    filter depth    : %u\n", m_options.psf_depth);
	fprintf(stderr, "    filter min-dist : %f\n", m_options.psf_min_dist);
	fprintf(stderr, "    firefly filter  : %f\n", m_options.firefly_filter);

	// allocate the PSF cache storage
	m_psf_hash.resize(HASH_SIZE);
	m_psf_values.alloc(HASH_SIZE);

	// pre-alloc queue storage
	{
		// determine how much storage we will need
		cugar::memory_arena arena;

		PTRayQueue	input_queue;
		PTRayQueue	scatter_queue;
		PTRayQueue	shadow_queue;
		PSFRefQueue ref_queue;

		alloc_queues(
			m_options,
			n_pixels,
			input_queue,
			scatter_queue,
			shadow_queue,
			ref_queue,
			arena );

		// alloc space for device timers
		arena.alloc<int64>( 16 );

		fprintf(stderr, "  allocating queue storage: %.1f MB\n", float(arena.size) / (1024*1024));
		m_memory_pool.alloc(arena.size);
	}

	// build the set of shifts
	const uint32 n_dimensions = 6 * (m_options.max_path_length + 1);
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);
	m_sequence.setup(n_dimensions, SHIFT_RES);

	const uint32 n_light_paths = n_pixels;

	fprintf(stderr, "  creating mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.get_mesh_lights().init( n_light_paths, renderer, 0u );

	fprintf(stderr, "  creating mesh lights... done\n");

	// compute the scene bbox
	m_bbox = renderer.compute_bbox();

	// disable smart algorithms if there are no emissive surfaces
	if (renderer.get_mesh_lights().get_vpl_count() == 0)
		m_options.nee_type = NEE_ALGORITHM_MESH;

	if (m_options.nee_type == NEE_ALGORITHM_RL)
	{
		fprintf(stderr, "  creating mesh VTLs... started\n");
		m_mesh_vtls->init(n_light_paths, renderer, 0u );
		fprintf(stderr, "  creating mesh VTLs... done (%u VTLs, %u clusters)\n", m_mesh_vtls->get_vtl_count(), m_mesh_vtls->get_bvh_clusters_count());

		fprintf(stderr, "  initializing VTLs RL... started\n");
		::init( m_vtls_rl, m_mesh_vtls );
		fprintf(stderr, "  initializing VTLs RL... done (%.1f MB)\n", m_vtls_rl->needed_bytes(VTL_RL_HASH_SIZE, m_mesh_vtls->get_bvh_clusters_count()) / float(1024*1024));
	}
}

void PSFPT::render(const uint32 instance, RenderingContext& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.rescale_frame( instance );

	//render_pass( instance, renderer, PSFPT::kPresamplePass );
	render_pass( instance, renderer, PSFPT::kFinalPass );

	renderer.update_variances( instance );

	// clamp the framebuffer contents to a reasonably high value, just to avoid outrageous fireflies
	renderer.clamp_frame( 100.0f );
}

void PSFPT::render_pass(const uint32 instance, RenderingContext& renderer, const PassType pass_type)
{
	//fprintf(stderr, "render started (%u)\n", instance);
	//! [PSFPT::render-1]
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;
	
	// carve an arena out of the pre-allocated memory pool
	cugar::memory_arena arena( m_memory_pool.ptr() );

	// alloc all the queues
	PTRayQueue  input_queue;
	PTRayQueue  scatter_queue;
	PTRayQueue  shadow_queue;
	PSFRefQueue ref_queue;

	alloc_queues(
		m_options,
		n_pixels,
		input_queue,
		scatter_queue,
		shadow_queue,
		ref_queue,
		arena );

	// fetch a view of the renderer
	RenderingContextView renderer_view = renderer.view(instance);

	//! [PSFPT::instantiate_vertex_processor]
	// instantiate our vertex processor
	PSFPTVertexProcessor vertex_processor( m_options.firefly_filter );
	//! [PSFPT::instantiate_vertex_processor]
	//! [PSFPT::render-1]

	// alloc space for device timers
	uint64* device_timers = arena.alloc<uint64>( 16 );

	cugar::Timer timer;
	timer.start();

	PTLoopStats stats;

	if (m_options.nee_type == NEE_ALGORITHM_RL)
	{
		if ((instance % 32) == 0)
		{
			// clear the RL hash tables after a bunch of iterations to avoid overflow...
			m_vtls_rl->clear();
		}
		else
		{
			// update the vtl cdfs
			m_vtls_rl->update();
			CUDA_CHECK(cugar::cuda::sync_and_check_error("vtl-rl update"));
		}
	}

	// setup the samples for this frame
	m_sequence.set_instance(instance);
	{
		// use the RL direct-lighting sampler
		if (m_options.nee_type == NEE_ALGORITHM_RL)
		{
			PSFPTContext<DirectLightingRL> context;
			context.options			= m_options;
			context.in_bounce		= 0;
			context.in_queue		= input_queue;
			context.scatter_queue	= scatter_queue;
			context.shadow_queue	= shadow_queue;
			context.sequence		= m_sequence.view();
			context.frame_weight	= 1.0f / float(renderer_view.instance + 1);
			context.device_timers	= device_timers;
			context.bbox			= m_bbox;
			context.dl				= DirectLightingRL(
				view( *m_vtls_rl ),
				m_mesh_vtls->view() );
			context.ref_queue		= ref_queue;
			context.psf_hashmap		= HashMap(
				HASH_SIZE,
				m_psf_hash.m_keys.ptr(),
				m_psf_hash.m_unique.ptr(),
				m_psf_hash.m_slots.ptr(),
				m_psf_hash.m_size.ptr()
			);
			context.psf_values = m_psf_values.ptr();

			// initialize the shading cache
			if ((instance % m_options.psf_temporal_reuse) == 0)
				m_psf_hash.clear();

			// reset the reference queue size
			cudaMemset(context.ref_queue.size, 0x00, sizeof(uint32));
			CUDA_CHECK(cugar::cuda::sync_and_check_error("clear reference queue"));
	
			// perform the actual path tracing
			path_trace_loop( context, vertex_processor, renderer, renderer_view, stats );

			// blend-in the PSF references
			if (pass_type == PSFPT::kFinalPass)
			{
				uint32 ref_queue_size;
				cudaMemcpy(&ref_queue_size, context.ref_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

				psf_blending(ref_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("psf blending"));
			}
		}
		else // use the regular mesh emitter direct-lighting sampler
		{
			// select which instantiation of the mesh light to use (VPLs or the plain mesh)
			MeshLight mesh_light = m_options.nee_type == NEE_ALGORITHM_VPL ? renderer_view.mesh_vpls : renderer_view.mesh_light;

			//! [PSFPT::render-2]
			PSFPTContext<DirectLightingMesh> context;
			context.options			= m_options;
			context.in_bounce		= 0;
			context.in_queue		= input_queue;
			context.scatter_queue	= scatter_queue;
			context.shadow_queue	= shadow_queue;
			context.sequence		= m_sequence.view();
			context.frame_weight	= 1.0f / float(renderer_view.instance + 1);
			context.device_timers	= device_timers;
			context.bbox			= m_bbox;
			context.dl				= DirectLightingMesh( mesh_light );
			context.ref_queue		= ref_queue;
			context.psf_hashmap		= HashMap(
				HASH_SIZE,
				m_psf_hash.m_keys.ptr(),
				m_psf_hash.m_unique.ptr(),
				m_psf_hash.m_slots.ptr(),
				m_psf_hash.m_size.ptr()
			);
			context.psf_values = m_psf_values.ptr();

			// initialize the shading cache
			if ((instance % m_options.psf_temporal_reuse) == 0)
				m_psf_hash.clear();

			// reset the reference queue size
			cudaMemset(context.ref_queue.size, 0x00, sizeof(uint32));
			CUDA_CHECK(cugar::cuda::sync_and_check_error("clear reference queue"));

			// perform the actual path tracing
			path_trace_loop( context, vertex_processor, renderer, renderer_view, stats );

			// blend-in the PSF references
			if (pass_type == PSFPT::kFinalPass)
			{
				uint32 ref_queue_size;
				cudaMemcpy(&ref_queue_size, context.ref_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

				psf_blending(ref_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("psf blending"));
			}
			//! [PSFPT::render-2]
		}
	}
	timer.stop();
	const float time = timer.seconds();
	// clear the global timer at instance zero
	if (instance == 0)
		m_time = time;
	else
		m_time += time;

	fprintf(stderr, "\r  %.1fs (%.1fms = rt[%.1fms + %.1fms + %.1fms] + shade[%.1fms + %.1fms] - %uK cells)        ",
		m_time,
		time * 1000.0f,
		stats.primary_rt_time * 1000.0f,
		stats.path_rt_time * 1000.0f,
		stats.shadow_rt_time * 1000.0f,
		stats.path_shade_time * 1000.0f,
		stats.shadow_shade_time * 1000.0f,
		m_psf_hash.size() / 1000);

#if defined(DEVICE_TIMING) && DEVICE_TIMING
	if (instance % 64 == 0)
		print_timer_stats( device_timers, stats );
#endif

	if (instance) // skip the first frame
	{
		m_stats.primary_rt_time += stats.primary_rt_time;
		m_stats.path_rt_time += stats.path_rt_time;
		m_stats.shadow_rt_time += stats.shadow_rt_time;
		m_stats.path_shade_time += stats.path_shade_time;
		m_stats.shadow_shade_time += stats.shadow_shade_time;
	}
}
