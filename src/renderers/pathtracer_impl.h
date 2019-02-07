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

#include <pathtracer.h>
#include <renderer.h>
#include <rt.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/memory_arena.h>
#include <pathtracer_core.h>
#include <pathtracer_queues.h>
#include <pathtracer_kernels.h>
#include <pathtracer_vertex_processor.h>

#define SHIFT_RES	256

///@addtogroup Fermat
///@{

///@defgroup PTModule
/// This module defines a path tracing renderer implemented on top of the \ref PTLib library.
///@{

///@defgroup PTModuleDetails
/// This module defines a path tracing renderer implemented on top of the \ref PTLib library.
///@{

namespace {

	//! [PT::PathTracingContext]
	// the internal path tracing context
	//
	template <typename TDirectLightingSampler>
	struct PathTracingContext : PTContextBase<PTOptions>, PTContextQueues
	{
		TDirectLightingSampler dl;
	};
	//! [PT::PathTracingContext]

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

} // anonymous namespace

PathTracer::PathTracer() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
	m_mesh_vtls = new MeshVTLStorage;
	m_vtls_rl = new VTLRLStorage;
}

void PathTracer::init(int argc, char** argv, RenderingContext& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	const char* nee_alg[] = { "mesh", "vpl", "rl" };

	fprintf(stderr, "  PT settings:\n");
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

	// pre-alloc queue storage
	{
		// determine how much storage we will need
		cugar::memory_arena arena;

		PTRayQueue	input_queue;
		PTRayQueue	scatter_queue;
		PTRayQueue	shadow_queue;

		alloc_queues(
			m_options,
			n_pixels,
			input_queue,
			scatter_queue,
			shadow_queue,
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

void PathTracer::update_vtls_rl(const uint32 instance)
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

/// \anchor PathTracer::render_impl
///
void PathTracer::render(const uint32 instance, RenderingContext& renderer)
{
	//! [PT::render-1]
	// pre-multiply the previous frame for blending
	renderer.rescale_frame( instance );

	//fprintf(stderr, "render started (%u)\n", instance);
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	cugar::memory_arena arena( m_memory_pool.ptr() );

	PTRayQueue in_queue;
	PTRayQueue scatter_queue;
	PTRayQueue shadow_queue;

	alloc_queues(
		m_options,
		n_pixels,
		in_queue,
		scatter_queue,
		shadow_queue,
		arena );

	// fetch a view of the renderer
	RenderingContextView renderer_view = renderer.view(instance);

	//! [PT::instantiate_vertex_processor]
	// instantiate the vertex processor
	PTVertexProcessor vertex_processor;
	//! [PT::instantiate_vertex_processor]
	//! [PT::render-1]

	// alloc space for device timers
	uint64* device_timers = arena.alloc<uint64>( 16 );

	cugar::Timer timer;
	timer.start();

	PTLoopStats stats;

	// update the direct-lighting VTLs Reinforcement-Learning tables
	if (m_options.nee_type == NEE_ALGORITHM_RL)
		update_vtls_rl( instance );

	// setup the samples for this frame
	m_sequence.set_instance(instance);

	// use our PTLib pathtracer
	{
		//! [PT::render-2]
		// use the Reinforcement-Learning direct-lighting sampler
		if (m_options.nee_type == NEE_ALGORITHM_RL)
		{
			// initialize the path-tracing context
			PathTracingContext<DirectLightingRL> context;
			context.options			= m_options;
			context.in_bounce		= 0;
			context.in_queue		= in_queue;
			context.scatter_queue	= scatter_queue;
			context.shadow_queue	= shadow_queue;
			context.sequence		= m_sequence.view();
			context.frame_weight	= 1.0f / float(renderer_view.instance + 1);
			context.device_timers	= device_timers;
			context.bbox			= m_bbox;
			context.dl				= DirectLightingRL(
				view( *m_vtls_rl ),
				m_mesh_vtls->view() );

			// instantiate the actual path tracing loop
			path_trace_loop( context, vertex_processor, renderer, renderer_view, stats );
		}
		else // use the regular mesh emitter direct-lighting sampler
		{
			// select which instantiation of the mesh light to use (VPLs or the plain mesh)
			MeshLight mesh_light = m_options.nee_type == NEE_ALGORITHM_VPL ? renderer_view.mesh_vpls : renderer_view.mesh_light;

			// initialize the path-tracing context
			PathTracingContext<DirectLightingMesh> context;
			context.options			= m_options;
			context.in_bounce		= 0;
			context.in_queue		= in_queue;
			context.scatter_queue	= scatter_queue;
			context.shadow_queue	= shadow_queue;
			context.sequence		= m_sequence.view();
			context.frame_weight	= 1.0f / float(renderer_view.instance + 1);
			context.device_timers	= device_timers;
			context.bbox			= m_bbox;
			context.dl				= DirectLightingMesh( mesh_light );

			// instantiate the actual path tracing loop
			path_trace_loop( context, vertex_processor, renderer, renderer_view, stats );
		}
		//! [PT::render-2]
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
		m_options.nee_type == NEE_ALGORITHM_RL ? m_vtls_rl->size() / 1000 : 0);

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
	renderer.update_variances( instance );
}


void PathTracer::keyboard(unsigned char character, int x, int y, bool& invalidate)
{
	invalidate = false;

	switch (character)
	{
	case 'i':
		m_options.direct_lighting = !m_options.direct_lighting;
		invalidate = true;
		break;
	}
}

// dump some speed stats
//
void PathTracer::dump_speed_stats(FILE* stats)
{
	fprintf(stats, "%f, %f, %f, %f, %f\n",
		m_stats.primary_rt_time.mean() * 1000.0f,
		m_stats.path_rt_time.mean() * 1000.0f,
		m_stats.shadow_rt_time.mean() * 1000.0f,
		m_stats.path_shade_time.mean() * 1000.0f,
		m_stats.shadow_shade_time.mean() * 1000.0f);
}

///@} PTModuleDetails
///@} PTModule
///@} Fermat
