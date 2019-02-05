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

#include <ray_queues.h>
#include <buffers.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

/// BPT queues view
///
struct BPTQueuesView
{
	RayQueue in_queue;
	RayQueue scatter_queue;
	RayQueue shadow_queue;
};

/// BPT queues storage
///
struct BPTQueuesStorage
{
	void alloc(const uint32 n_entries)
	{
		m_rays.alloc(n_entries);
		m_hits.alloc(n_entries);
		m_weights.alloc(n_entries);
		m_probs.alloc(n_entries);
		m_path_weights.alloc(n_entries);
		m_pixels.alloc(n_entries);
		m_counters.alloc(4);
	}

	void alloc(const uint32 n_eye_paths, const uint32 n_light_paths, const uint32 max_path_length)
	{
		const uint32 queue_size = cugar::max(n_eye_paths, n_light_paths) * 2 + n_eye_paths * (max_path_length + 1);

		alloc(queue_size);
	}

	BPTQueuesView view(const uint32 n_eye_paths, const uint32 n_light_paths)
	{
		const uint32 n_rays = cugar::max(n_eye_paths, n_light_paths);

		BPTQueuesView r;

		r.in_queue.rays			= m_rays.ptr();
		r.in_queue.hits			= m_hits.ptr();
		r.in_queue.weights		= m_weights.ptr();
		r.in_queue.probs		= m_probs.ptr();
		r.in_queue.pixels		= m_pixels.ptr();
		r.in_queue.path_weights = m_path_weights.ptr();
		r.in_queue.size			= m_counters.ptr();

		r.scatter_queue.rays			= r.in_queue.rays + n_rays;
		r.scatter_queue.hits			= r.in_queue.hits + n_rays;
		r.scatter_queue.weights			= r.in_queue.weights + n_rays;
		r.scatter_queue.probs			= r.in_queue.probs + n_rays;
		r.scatter_queue.pixels			= r.in_queue.pixels + n_rays;
		r.scatter_queue.path_weights	= r.in_queue.path_weights + n_rays;
		r.scatter_queue.size			= r.in_queue.size + 1;

		r.shadow_queue.rays			= r.scatter_queue.rays + n_rays;
		r.shadow_queue.hits			= r.scatter_queue.hits + n_rays;
		r.shadow_queue.weights		= r.scatter_queue.weights + n_rays;
		r.shadow_queue.probs		= r.scatter_queue.probs + n_rays;
		r.shadow_queue.pixels		= r.scatter_queue.pixels + n_rays;
		r.shadow_queue.path_weights = r.scatter_queue.path_weights + n_rays;
		r.shadow_queue.size			= r.scatter_queue.size + 1;

		return r;
	}

	DomainBuffer<CUDA_BUFFER, Ray>		m_rays;
	DomainBuffer<CUDA_BUFFER, Hit>		m_hits;
	DomainBuffer<CUDA_BUFFER, float4>	m_weights;
	DomainBuffer<CUDA_BUFFER, float>	m_probs;
	DomainBuffer<CUDA_BUFFER, float4>	m_path_weights;		// see 'TempPathWeights' struct
	DomainBuffer<CUDA_BUFFER, uint32>	m_pixels;
	DomainBuffer<CUDA_BUFFER, uint32>	m_counters;
};

///@} BPTLib
///@} Fermat
