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

#include <pathtracer_core.h>
#include <pathtracer_queues.h>
#include <renderer.h>
#include <rt.h>
#include <cugar/basic/memory_arena.h>

///@addtogroup Fermat
///@{

///@addtogroup PTLib
///@{

#define SHADE_HITS_BLOCKSIZE	64
#define SHADE_HITS_CTA_BLOCKS	8	// Maxwell / Volta : 16 - Turing : 8

/// A class encapsulating all the queues needed by the \ref PTLib module wavefront scheduler kernels,
/// and defining the basic \ref TPTContext trace methods
///
struct PTContextQueues
{
	PTRayQueue	in_queue;
	PTRayQueue	shadow_queue;
	PTRayQueue	scatter_queue;

	template <typename TPTVertexProcessor>
	FERMAT_DEVICE
	void trace_ray(
		TPTVertexProcessor&		_vertex_processor,
		RenderingContextView&	_renderer,
		const PixelInfo			_pixel,
		const MaskedRay			_ray,
		const cugar::Vector4f	_weight,
		const cugar::Vector2f	_cone			= cugar::Vector2f(0),
		const uint32			_vertex_info	= uint32(-1),
		const uint32			_nee_slot		= uint32(-1))
	{
		scatter_queue.warp_append( _pixel, _ray, _weight, _cone, _vertex_info, _nee_slot );
	}

	template <typename TPTVertexProcessor>
	FERMAT_DEVICE
	void trace_shadow_ray(
		TPTVertexProcessor&		_vertex_processor,
		RenderingContextView&	_renderer,
		const PixelInfo			_pixel,
		const MaskedRay			_ray,
		const cugar::Vector3f	_weight,
		const cugar::Vector3f	_weight_d,
		const cugar::Vector3f	_weight_g,
		const uint32			_vertex_info	= uint32(-1),
		const uint32			_nee_slot		= uint32(-1),
		const uint32			_nee_sample		= uint32(-1))
	{
		shadow_queue.warp_append( _pixel, _ray, cugar::Vector4f(_weight, 0.0f), cugar::Vector4f(_weight_d, 0.0f), cugar::Vector4f(_weight_g, 0.0f), _vertex_info, _nee_slot, _nee_sample );
	}
};

/// a utility function to alloc the path tracing queues
///
inline
void alloc_queues(
	PTOptions				options,
	const uint32			n_pixels,
	PTRayQueue&				input_queue,
	PTRayQueue&				scatter_queue,
	PTRayQueue&				shadow_queue,
	cugar::memory_arena&	arena)
{
	input_queue.rays		= arena.alloc<MaskedRay>(n_pixels);
	input_queue.hits		= arena.alloc<Hit>(n_pixels);
	input_queue.weights		= arena.alloc<float4>(n_pixels);
	input_queue.weights_d	= NULL;
	input_queue.weights_g	= NULL;
	input_queue.pixels		= arena.alloc<uint4>(n_pixels);
	input_queue.cones		= arena.alloc<float2>(n_pixels);
	input_queue.size		= arena.alloc<uint32>(1);

	scatter_queue.rays		= arena.alloc<MaskedRay>(n_pixels);
	scatter_queue.hits		= arena.alloc<Hit>(n_pixels);
	scatter_queue.weights	= arena.alloc<float4>(n_pixels);
	scatter_queue.weights_d	= NULL;
	scatter_queue.weights_g	= NULL;
	scatter_queue.pixels	= arena.alloc<uint4>(n_pixels);
	scatter_queue.cones		= arena.alloc<float2>(n_pixels);
	scatter_queue.size		= arena.alloc<uint32>(1);

	const uint32 n_shadow_rays = 2u;

	shadow_queue.rays		= arena.alloc<MaskedRay>(n_pixels*n_shadow_rays);
	shadow_queue.hits		= arena.alloc<Hit>(n_pixels*n_shadow_rays);
	shadow_queue.weights	= arena.alloc<float4>(n_pixels*n_shadow_rays);
	shadow_queue.weights_d	= arena.alloc<float4>(n_pixels*n_shadow_rays);
	shadow_queue.weights_g	= arena.alloc<float4>(n_pixels*n_shadow_rays);
	shadow_queue.pixels		= arena.alloc<uint4>(n_pixels*n_shadow_rays);
	shadow_queue.size		= arena.alloc<uint32>(1);
}

//------------------------------------------------------------------------------
/// a kernel to generate primary rays
///
/// \tparam TPTContext				A path tracing context
///
template <typename TPTContext>
__global__ void generate_primary_rays_kernel(TPTContext context, RenderingContextView renderer, cugar::Vector3f U, cugar::Vector3f V, cugar::Vector3f W, const float W_len, const float square_pixel_focal_length)
{
	const uint2 pixel = make_uint2(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y );

	if (pixel.x >= renderer.res_x || pixel.y >= renderer.res_y)
		return;

	const int idx = pixel.x + pixel.y*renderer.res_x;

	const MaskedRay ray = generate_primary_ray( context, renderer, pixel, U, V, W );

	reinterpret_cast<float4*>(context.in_queue.rays)[2 * idx + 0] = make_float4(ray.origin.x, ray.origin.y, ray.origin.z, __uint_as_float(ray.mask)); // origin, tmin
	reinterpret_cast<float4*>(context.in_queue.rays)[2 * idx + 1] = make_float4(ray.dir.x, ray.dir.y, ray.dir.z, ray.tmax); // dir, tmax

	// write the filter weight
	context.in_queue.weights[idx] = cugar::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

	const float out_p = camera_direction_pdf(U, V, W, W_len, square_pixel_focal_length, ray.dir, false);

	// write the pixel index
	context.in_queue.pixels[idx] = make_uint4( idx, uint32(-1), uint32(-1), uint32(-1) );

	// write the ray cone
	context.in_queue.cones[idx] = make_float2( 0, out_p );

	if (idx == 0)
		*context.in_queue.size = renderer.res_x * renderer.res_y;
}

//------------------------------------------------------------------------------
/// dispatch the kernel to generate primary rays
///
/// \tparam TPTContext				A path tracing context
///
template <typename TPTContext>
void generate_primary_rays(TPTContext context, const RenderingContextView renderer)
{
	cugar::Vector3f U, V, W;
	camera_frame(renderer.camera, renderer.aspect, U, V, W);

	const float square_pixel_focal_length = renderer.camera.square_pixel_focal_length(renderer.res_x, renderer.res_y);

	dim3 blockSize(32, 16);
	dim3 gridSize(cugar::divide_ri(renderer.res_x, blockSize.x), cugar::divide_ri(renderer.res_y, blockSize.y));
	generate_primary_rays_kernel << < gridSize, blockSize >> > (context, renderer, U, V, W, length(W), square_pixel_focal_length);
}
//------------------------------------------------------------------------------

/// a kernel to shade all path tracing hits
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
template <uint32 NUM_WARPS, typename TPTContext, typename TPTVertexProcessor>
__global__
__launch_bounds__(SHADE_HITS_BLOCKSIZE, SHADE_HITS_CTA_BLOCKS)
void shade_hits_kernel(const uint32 in_queue_size, TPTContext context, TPTVertexProcessor vertex_processor, RenderingContextView renderer)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.in_queue.size
	{
		const uint4 packed_pixel = cugar::cuda::load<cugar::cuda::LOAD_CG>( &context.in_queue.pixels[thread_id] ); // make sure we use a vectorized load

		const PixelInfo		  pixel_info		= packed_pixel.x;
		const uint32		  prev_vertex_info	= packed_pixel.y;
		const uint32		  prev_nee_slot		= packed_pixel.z;
		const float2		  cone				= context.in_queue.cones[thread_id];
		const MaskedRay		  ray				= context.in_queue.rays[thread_id];
		const Hit			  hit				= context.in_queue.hits[thread_id];
		const cugar::Vector4f w					= context.in_queue.weights[thread_id];

		const uint2 pixel = make_uint2(
			pixel_info.pixel % renderer.res_x,
			pixel_info.pixel / renderer.res_x
		);

		shade_vertex(
			context,
			vertex_processor,
			renderer,
			context.in_bounce,
			pixel_info,
			pixel,
			ray,
			hit,
			w,
			prev_vertex_info,
			prev_nee_slot,
			cone );
	}
}

/// dispatch the shade hits kernel
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
template <typename TPTContext, typename TPTVertexProcessor>
void shade_hits(const uint32 in_queue_size, TPTContext context, TPTVertexProcessor vertex_processor, RenderingContextView renderer)
{
	const uint32 blockSize(SHADE_HITS_BLOCKSIZE);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));

	shade_hits_kernel<blockSize / 32><<< gridSize, blockSize >>>( in_queue_size, context, vertex_processor, renderer );
}

/// a kernel to process NEE samples using computed occlusion information
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
template <typename TPTContext, typename TPTVertexProcessor>
__global__
void solve_occlusion_kernel(const uint32 in_queue_size, TPTContext context, TPTVertexProcessor vertex_processor, RenderingContextView renderer)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.shadow_queue.size
	{
		const PixelInfo		  pixel_info	 = context.shadow_queue.pixels[thread_id].x;
		const uint32		  vertex_info	 = context.shadow_queue.pixels[thread_id].y;
		const uint32		  nee_slot		 = context.shadow_queue.pixels[thread_id].z;
		const uint32		  nee_sample	 = context.shadow_queue.pixels[thread_id].w;
		const Hit			  hit			 = context.shadow_queue.hits[thread_id];
		const cugar::Vector4f w				 = context.shadow_queue.weights[thread_id];
		const cugar::Vector4f w_d			 = context.shadow_queue.weights_d[thread_id];
		const cugar::Vector4f w_g			 = context.shadow_queue.weights_g[thread_id];

		solve_occlusion( context, vertex_processor, renderer, hit.t > 0.0f, pixel_info, w.xyz(), w_d.xyz(), w_g.xyz(), vertex_info, nee_slot, nee_sample );
	}
}

/// dispatch a kernel to process NEE samples using computed occlusion information
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
template <typename TPTContext, typename TPTVertexProcessor>
void solve_occlusion(const uint32 in_queue_size, TPTContext context, TPTVertexProcessor vertex_processor, RenderingContextView renderer)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
	solve_occlusion_kernel<<< gridSize, blockSize >>>( in_queue_size, context, vertex_processor, renderer );
}

/// Internal stats for the path_trace_loop() function
///
struct PTLoopStats
{
	/// constructor
	///
	PTLoopStats()
	{
		primary_rt_time = 0.0f;
		path_rt_time = 0.0f;
		shadow_rt_time = 0.0f;
		path_shade_time = 0.0f;
		shadow_shade_time = 0.0f;

		shade_events = 0;
	}

	float primary_rt_time;		///<	time spent for tracing primary rays
	float path_rt_time;			///<	time spent for tracing scattering rays
	float shadow_rt_time;		///<	time spent for tracing shadow rays
	float path_shade_time;		///<	time spent for shading path vertices
	float shadow_shade_time;	///<	time spent for shading shadow samples (i.e. in solve_occlusion)
	uint64 shade_events;		///<	number of path vertex shade events
};

/// main path tracing loop
///
template <typename TPTContext, typename TPTVertexProcessor>
void path_trace_loop(
	TPTContext&				context,
	TPTVertexProcessor&		vertex_processor,
	RenderingContext&		renderer,
	RenderingContextView&	renderer_view,
	PTLoopStats&			stats)
{
	generate_primary_rays(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	cudaMemset(context.device_timers, 0x00, sizeof(uint64) * 16);

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

		// update per bounce options
		compute_per_bounce_options( context, renderer_view );

		// trace the rays generated at the previous bounce
		//
		{
			FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> trace_timer(context.in_bounce == 0 ? &stats.primary_rt_time : &stats.path_rt_time));

			renderer.get_rt_context()->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace shaded"));
		}

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// perform lighting at this bounce
		//
		{
			FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> shade_timer(&stats.path_shade_time));

			shade_hits(in_queue_size, context, vertex_processor, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("shade hits"));

			stats.shade_events += in_queue_size;
		}

		// trace & accumulate occlusion queries
		{
			uint32 shadow_queue_size;
			cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// trace the rays
			//
			if (shadow_queue_size)
			{
				FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> trace_timer(&stats.shadow_rt_time));

				renderer.get_rt_context()->trace_shadow(shadow_queue_size, (MaskedRay*)context.shadow_queue.rays, context.shadow_queue.hits);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("trace occlusion"));
			}

			// shade the results
			//
			if (shadow_queue_size)
			{
				FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> shade_timer(&stats.shadow_shade_time));

				solve_occlusion(shadow_queue_size, context, vertex_processor, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));
			}
		}

		std::swap(context.in_queue, context.scatter_queue);
	}
}

inline void print_timer_stats(const uint64* device_timers, const PTLoopStats& stats)
{
	uint64 h_device_timers[16];
	cudaMemcpy(&h_device_timers, device_timers, sizeof(uint64) * 16, cudaMemcpyDeviceToHost);

	const uint64 shade_events = stats.shade_events;

	//const uint32 warp_size = 32;
	//const uint64 warp_shade_events = (shade_events + warp_size-1) / warp_size;

	const float setup_time					= float(h_device_timers[SETUP_TIME])				/ float(shade_events);
	const float brdf_eval_time				= float(h_device_timers[BRDF_EVAL_TIME])			/ float(shade_events);
	const float dirlight_sample_time		= float(h_device_timers[DIRLIGHT_SAMPLE_TIME])		/ float(shade_events);
	const float dirlight_eval_time			= float(h_device_timers[DIRLIGHT_EVAL_TIME])		/ float(shade_events);
	const float lights_preprocess_time		= float(h_device_timers[LIGHTS_PREPROCESS_TIME])	/ float(shade_events);
	const float lights_sample_time			= float(h_device_timers[LIGHTS_SAMPLE_TIME])		/ float(shade_events);
	const float lights_eval_time			= float(h_device_timers[LIGHTS_EVAL_TIME])			/ float(shade_events);
	const float lights_mapping_time			= float(h_device_timers[LIGHTS_MAPPING_TIME])		/ float(shade_events);
	const float lights_update_time			= float(h_device_timers[LIGHTS_UPDATE_TIME])		/ float(shade_events);
	const float brdf_sample_time			= float(h_device_timers[BRDF_SAMPLE_TIME])			/ float(shade_events);
	const float trace_shadow_time			= float(h_device_timers[TRACE_SHADOW_TIME])			/ float(shade_events);
	const float trace_shaded_time			= float(h_device_timers[TRACE_SHADED_TIME])			/ float(shade_events);
	const float vertex_preprocess_time		= float(h_device_timers[PREPROCESS_VERTEX_TIME])	/ float(shade_events);
	const float nee_weights_time			= float(h_device_timers[NEE_WEIGHTS_TIME])			/ float(shade_events);
	const float scattering_weights_time		= float(h_device_timers[SCATTERING_WEIGHTS_TIME])	/ float(shade_events);
	const float fbuffer_writes_time			= float(h_device_timers[FBUFFER_WRITES_TIME])		/ float(shade_events);

	const float total_time =
			setup_time
		+ brdf_eval_time
		+ dirlight_sample_time
		+ dirlight_eval_time
		+ lights_sample_time
		+ lights_eval_time
		+ lights_mapping_time
		+ lights_update_time
		+ brdf_sample_time
		+ trace_shadow_time
		+ trace_shaded_time
		+ vertex_preprocess_time
		+ nee_weights_time
		+ scattering_weights_time
		+ fbuffer_writes_time;

	fprintf(stderr, "\n  device timing:    %f clks\n", total_time);
	fprintf(stderr, "    setup           : %4.1f %%, %f clks\n", 100.0 * setup_time / total_time, setup_time);
	fprintf(stderr, "    dirlight sample : %4.1f %%, %f clks\n", 100.0 * dirlight_sample_time / total_time, dirlight_sample_time);
	fprintf(stderr, "    dirlight eval   : %4.1f %%, %f clks\n", 100.0 * dirlight_eval_time / total_time, dirlight_eval_time);
	fprintf(stderr, "    lights preproc  : %4.1f %%, %f clks\n", 100.0 * lights_preprocess_time / total_time, lights_preprocess_time);
	fprintf(stderr, "    lights sample   : %4.1f %%, %f clks\n", 100.0 * lights_sample_time / total_time, lights_sample_time);
	fprintf(stderr, "    lights eval     : %4.1f %%, %f clks\n", 100.0 * lights_eval_time / total_time, lights_eval_time);
	fprintf(stderr, "    lights map      : %4.1f %%, %f clks\n", 100.0 * lights_mapping_time / total_time, lights_mapping_time);
	fprintf(stderr, "    lights update   : %4.1f %%, %f clks\n", 100.0 * lights_update_time / total_time, lights_update_time);
	fprintf(stderr, "    brdf eval       : %4.1f %%, %f clks\n", 100.0 * brdf_eval_time / total_time, brdf_eval_time);
	fprintf(stderr, "    brdf sample     : %4.1f %%, %f clks\n", 100.0 * brdf_sample_time / total_time, brdf_sample_time);
	fprintf(stderr, "    trace shadow    : %4.1f %%, %f clks\n", 100.0 * trace_shadow_time / total_time, trace_shadow_time);
	fprintf(stderr, "    trace shaded    : %4.1f %%, %f clks\n", 100.0 * trace_shaded_time / total_time, trace_shaded_time);
	fprintf(stderr, "    preprocess      : %4.1f %%, %f clks\n", 100.0 * vertex_preprocess_time / total_time, vertex_preprocess_time);
	fprintf(stderr, "    nee weights     : %4.1f %%, %f clks\n", 100.0 * nee_weights_time / total_time, nee_weights_time);
	fprintf(stderr, "    scatter weights : %4.1f %%, %f clks\n", 100.0 * scattering_weights_time / total_time, scattering_weights_time);
	fprintf(stderr, "    fbuffer writes  : %4.1f %%, %f clks\n", 100.0 * fbuffer_writes_time / total_time, fbuffer_writes_time);
}

///@} PTLib
///@} Fermat
