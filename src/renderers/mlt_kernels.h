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

#include <mlt.h>
#include <mlt_core.h>

namespace { // anonymous namespace

	//------------------------------------------------------------------------------

	__global__
	void perturb_primary_light_vertices_mlt_kernel(MLTContext context, RenderingContextView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		const uint32 chain_id = thread_id;

		if (chain_id < context.n_chains)
			perturb_primary_light_vertex( context, renderer, context.scatter_queue, chain_id );
	}

	void perturb_primary_light_vertices_mlt(MLTContext context, RenderingContextView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize));
		perturb_primary_light_vertices_mlt_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	//------------------------------------------------------------------------------

	template <uint32 NUM_WARPS>
	__global__
	void perturb_secondary_light_vertices_mlt_kernel(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.in_queue.size
			perturb_secondary_light_vertex( context, renderer, context.in_queue, context.scatter_queue, thread_id );
	}

	void perturb_secondary_light_vertices_mlt(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		perturb_secondary_light_vertices_mlt_kernel<blockSize / 32> << < gridSize, blockSize >> > (in_queue_size, context, renderer);
	}

	//------------------------------------------------------------------------------
	__global__ void perturb_primary_rays_mlt_kernel(MLTContext context, RenderingContextView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		const uint32 chain_id = thread_id;

		if (chain_id >= context.n_chains)
			return;

		perturb_primary_eye_vertex( context, renderer, context.in_queue, chain_id, chain_id );

		// write out the total number of queue entries
		if (chain_id == 0)
			*context.in_queue.size = context.n_chains;
	}

	void perturb_primary_rays_mlt(MLTContext context, const RenderingContextView renderer)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize.x));
		perturb_primary_rays_mlt_kernel << < gridSize, blockSize >> > (context, renderer);
	}
	//------------------------------------------------------------------------------

	template <uint32 NUM_WARPS>
	__global__
	void perturb_secondary_eye_vertices_mlt_kernel(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.in_queue.size
			perturb_secondary_eye_vertex( context, renderer, context.in_queue, context.scatter_queue, thread_id );
	}

	void perturb_secondary_eye_vertices_mlt(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		perturb_secondary_eye_vertices_mlt_kernel<blockSize / 32> << < gridSize, blockSize >> > (in_queue_size, context, renderer);
	}

	//------------------------------------------------------------------------------

	__global__
	void solve_occlusion_mlt_kernel(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.shadow_queue.size
			solve_occlusion_mlt( context, renderer, thread_id );
	}

	void solve_occlusion_mlt(const uint32 in_queue_size, MLTContext context, RenderingContextView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		solve_occlusion_mlt_kernel << < gridSize, blockSize >> > (in_queue_size, context, renderer);
	}

	//------------------------------------------------------------------------------

	__global__
	void sample_seeds_kernel(const uint32 n_connections, const float* connections_cdf, const uint4* connections_index, const uint32 n_seeds, uint32* seeds, uint32* st_counters, const uint32 max_path_length)
	{
		const uint32 seed_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (seed_id < n_seeds)
		{
			// pick a stratified sample
			const float r = (seed_id + cugar::randfloat( seed_id, 0 )) / float(n_seeds);

			const uint32 seed = cugar::upper_bound_index( r * connections_cdf[n_connections-1], connections_cdf, n_connections);

			const uint4 connection = connections_index[seed];

			seeds[seed_id] = seed | (connection.z << 24) | (connection.w << 28);

			// keep stats
			atomicAdd(st_counters + connection.z + connection.w*(max_path_length + 2), 1u);
		}
	}

	void sample_seeds(const uint32 n_connections, const float* connections_cdf, const uint4* connections_index, const uint32 n_seeds, uint32* seeds, uint32* st_counters, const uint32 max_path_length)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		sample_seeds_kernel<<< gridSize, blockSize >>>(n_connections, connections_cdf, connections_index, n_seeds, seeds, st_counters, max_path_length);
	}

	//------------------------------------------------------------------------------

	__global__
	void recover_seed_paths_kernel(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, MLTContext context, RenderingContextView renderer)
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

			// fetch the compact path vertices
			Path path(s + t, context.vertices + seed_id, n_seeds);

			for (uint32 i = 0; i < s; ++i)
				path.v_L(i) = context.bpt_light_vertices[light_idx + i * context.n_light_paths];

			for (uint32 i = 0; i < t; ++i)
				path.v_E(i) = context.bpt_eye_vertices[eye_idx + i * context.n_eye_paths];

			// write the st info
			context.st[seed_id] = make_char4(s, t, s, t);
		}
	}
	void recover_seed_paths(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, MLTContext context, RenderingContextView renderer)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		recover_seed_paths_kernel <<< gridSize, blockSize >>>(n_seeds, seeds, n_lights, context, renderer);
	}

} // anonymous namespace
