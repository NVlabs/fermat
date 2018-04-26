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

#include <bpt_context.h>
#include <bpt_utils.h>
#include <bpt_options.h>
#include <bpt_kernels.h>

namespace bpt {

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
void sample_light_subpaths(
	const uint32		n_light_paths,
	TPrimaryCoordinates	primary_coords,
	TBPTContext&		context,
	const TBPTConfig&	config,
	Renderer&			renderer,
	RendererView&		renderer_view)
{
	// reset the output queue size
	cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

	// reset the output vertex counter
	cudaMemset(context.light_vertices.vertex_counter, 0x00, sizeof(uint32));

	// generate primary light vertices from the mesh light samples, including the sampling of a direction
	bpt::generate_primary_light_vertices(n_light_paths, primary_coords, context, renderer_view, config);

	// swap the input and output queues
	std::swap(context.in_queue, context.scatter_queue);

	// for each bounce: trace rays, process hits (store light_vertices, generate sampling directions)
	for (context.in_bounce = 0;
		context.in_bounce < context.options.max_path_length - 1;
		context.in_bounce++)
	{
		uint32 in_queue_size;

		// read out the number of output rays produced
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		//fprintf(stderr, "  bounce %u:\n    rays: %u\n", bounce, in_queue_size);

		// reset the output queue counters
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// trace the rays generated at the previous bounce
		//
		//fprintf(stderr, "    trace\n");
		{
			optix::prime::Query query = renderer.m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
			query->setRays(in_queue_size, Ray::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.rays);
			query->setHits(in_queue_size, Hit::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.hits);
			query->execute(0);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
		}

		// process the light_vertex hits at this bounce
		//
		bpt::process_secondary_light_vertices(in_queue_size, n_light_paths, primary_coords, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("process light vertices"));

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);
	}

	//uint32 vertex_counts[1024];
	//cudaMemcpy(&vertex_counts[0], context.light_vertices.vertex_counts, sizeof(uint32) * context.options.max_path_length, cudaMemcpyDeviceToHost);
	//for (uint32 s = 0; s < context.options.max_path_length; ++s)
	//	fprintf(stderr, "  light vertices[%u] : %u (%u))\n", s, vertex_counts[s] - (s ? vertex_counts[s-1] : 0), vertex_counts[s]);
}

template <typename TSampleSink, typename TBPTContext>
void solve_shadows(TSampleSink sample_sink, TBPTContext& context, Renderer& renderer, RendererView& renderer_view)
{
	uint32 shadow_queue_size;
	cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

	// trace the rays
	//
	if (shadow_queue_size)
	{
		//fprintf(stderr, "      trace occlusion (%u queries)\n", shadow_queue_size);
		optix::prime::Query query = renderer.m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
		query->setRays(shadow_queue_size, Ray::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.shadow_queue.rays);
		query->setHits(shadow_queue_size, Hit::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.shadow_queue.hits);
		query->execute(0);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("trace occlusion"));
	}

	// shade the results
	//
	if (shadow_queue_size)
	{
		//fprintf(stderr, "      solve occlusion\n");

		bpt::solve_occlusions(shadow_queue_size, sample_sink, context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));

		// reset the shadow queue counter
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));
	}
}

template <typename TPrimaryCoordinates, typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void sample_eye_subpaths(
	const uint32		n_eye_paths,
	const uint32		n_light_paths,
	TPrimaryCoordinates	primary_coords,
	TSampleSink			sample_sink,
	TBPTContext&		context,
	const TBPTConfig&	config,
	Renderer&			renderer,
	RendererView&		renderer_view,
	const bool			lazy_shadows = false)
{
	// reset the output queue counters
	cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
	cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

	//fprintf(stderr, "    trace primary rays : started\n");

	// generate the primary rays
	bpt::generate_primary_eye_vertices(n_eye_paths, n_light_paths, primary_coords, context, renderer_view, config);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	//fprintf(stderr, "    trace primary rays: done\n");

	for (context.in_bounce = 0;
		 context.in_bounce < context.options.max_path_length;
		 context.in_bounce++)
	{
		uint32 in_queue_size;

		// read out the number of output rays produced by the previous pass
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);
		CUDA_CHECK(cugar::cuda::check_error("memcpy"));

		//fprintf(stderr, "    input queue size: %u\n", in_queue_size);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		//fprintf(stderr, "  bounce %u:\n    rays: %u\n", bounce, in_queue_size);

		// reset the output queue counters
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// trace the rays generated at the previous bounce
		//
		//fprintf(stderr, "    trace\n");
		{
			optix::prime::Query query = renderer.m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
			query->setRays(in_queue_size, Ray::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.rays);
			query->setHits(in_queue_size, Hit::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.hits);
			query->execute(0);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
		}

		//fprintf(stderr, "    shade hits(%u)\n", context.in_bounce);

		// perform lighting at this bounce
		//
		bpt::process_secondary_eye_vertices(in_queue_size, n_eye_paths, n_light_paths, sample_sink, primary_coords, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("process eye vertices"));

		if (lazy_shadows == false)
		{
			// trace & accumulate occlusion queries
			solve_shadows(sample_sink, context, renderer, renderer_view);
		}

		//fprintf(stderr, "    finish pass\n");

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);
	}

	if (lazy_shadows)
	{
		// trace & accumulate occlusion queries
		solve_shadows(sample_sink, context, renderer, renderer_view);
	}
}

template <typename TEyePrimaryCoordinates, typename TLightPrimaryCoordinates, typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void sample_paths(
	const uint32				n_eye_paths,
	const uint32				n_light_paths,
	TEyePrimaryCoordinates		eye_primary_coords,
	TLightPrimaryCoordinates	light_primary_coords,
	TSampleSink					sample_sink,
	TBPTContext&				context,
	const TBPTConfig&			config,
	Renderer&					renderer,
	RendererView&				renderer_view,
	const bool					lazy_shadows = false)
{
	sample_light_subpaths(
		n_light_paths,
		light_primary_coords,
		context,
		config,
		renderer,
		renderer_view);

	sample_eye_subpaths(
		n_eye_paths,
		n_light_paths,
		eye_primary_coords,
		sample_sink,
		context,
		config,
		renderer,
		renderer_view,
		lazy_shadows);
}

template <typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void light_tracing(
	const uint32		n_light_paths,
	TSampleSink			sample_sink,
	TBPTContext&		context,
	const TBPTConfig&	config,
	Renderer&			renderer,
	RendererView&		renderer_view)
{
	// solve pure light tracing occlusions
	if (context.options.light_tracing)
	{
		//fprintf(stderr, "  light tracing : started\n");

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));

		bpt::light_tracing(n_light_paths, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("light tracing"));

		bpt::solve_shadows(sample_sink, context, renderer, renderer_view);
		//fprintf(stderr, "  light tracing : done\n");
	}
}

///@} BPTLib
///@} Fermat

} // namespace bpt