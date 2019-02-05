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

#include <hellopt.h>
#include <hellopt_impl.h>


void HelloPT::init(int argc, char** argv, RenderingContext& renderer)
{
	// parse the options
	m_options.parse( argc, argv );

	// calculate the number of pixels in the target framebuffer
	const uint2 res = renderer.res();

	const uint32 n_pixels = res.x * res.y;

	// pre-alloc some queue storage
	{
		// keep track of how much storage we'll need
		cugar::memory_arena arena;

		PTRayQueue input_queue;
		PTRayQueue scatter_queue;
		PTRayQueue shadow_queue;
			
		alloc_queues( n_pixels, input_queue, scatter_queue, shadow_queue, arena );
	
		fprintf(stderr, "  allocating queue storage: %.1f MB\n", float(arena.size) / (1024*1024));
		m_memory_pool.alloc(arena.size);
	}

	// build the set of samples assuming 6 random numbers per path vertex and a tile size of 256 pixels
	const uint32 n_dimensions = 6 * (m_options.max_path_length + 1);
	const uint32 tile_size    = 256;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);
	m_sequence.setup(n_dimensions, tile_size);

	fprintf(stderr, "  creating mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.get_mesh_lights().init( n_pixels, renderer );

	fprintf(stderr, "  creating mesh lights... done\n");
}

void HelloPT::render(const uint32 instance, RenderingContext& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.rescale_frame( instance );

	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// carve an arena out of our pre-allocated memory pool
	cugar::memory_arena arena( m_memory_pool.ptr() );

	// alloc all the queues
	PTRayQueue in_queue;
	PTRayQueue scatter_queue;
	PTRayQueue shadow_queue;

	alloc_queues(
		n_pixels,
		in_queue,
		scatter_queue,
		shadow_queue,
		arena );

	// fetch a view of the renderer
	RenderingContextView renderer_view = renderer.view(instance);

	// fetch the ray tracing context
	RTContext* rt_context = renderer.get_rt_context();

	// setup the samples for this frame
	m_sequence.set_instance(instance);

	// setup our context
	HelloPTContext context;
	context.options       = m_options;
	context.sequence      = m_sequence.view();
	context.frame_weight  = 1.0f / (instance + 1);
	context.in_queue      = in_queue;
	context.scatter_queue = scatter_queue;
	context.shadow_queue  = shadow_queue;

	// generate the primary rays
	generate_primary_rays(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	// start the path tracing loop
	for (context.in_bounce = 0;
		 context.in_bounce < context.options.max_path_length;
		 context.in_bounce++)
	{
		uint32 in_queue_size;

		// fetch the amount of tasks in the queue
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		// trace the rays generated at the previous bounce
		//
		rt_context->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// perform lighting at this bounce
		//
		shade_vertices(in_queue_size, context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("shade hits"));

		// trace & accumulate occlusion queries
		{
			// fetch the amount of tasks in the queue
			uint32 shadow_queue_size;
			cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			if (shadow_queue_size)
			{
				// trace the rays
				//
				rt_context->trace_shadow(shadow_queue_size, (MaskedRay*)context.shadow_queue.rays, context.shadow_queue.hits);

				// shade the results
				//
				resolve_occlusion(shadow_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("resolve occlusion"));
			}
		}

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);
	}
}
