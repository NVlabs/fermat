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

// ------------------------------------------------------------------------- //
//
// Declaration of utility ray queues
//
// ------------------------------------------------------------------------- //

#include <types.h>
#include <ray.h>
#include <cugar/basic/atomics.h>
#include <cugar/basic/cuda/warp_atomics.h>
#include <cugar/linalg/vector.h>

union PixelInfo
{
	FERMAT_HOST_DEVICE PixelInfo() {}
	FERMAT_HOST_DEVICE PixelInfo(const uint32 _packed) : packed(_packed) {}
	FERMAT_HOST_DEVICE PixelInfo(const uint32 _pixel, const uint32 _channel) : pixel(_pixel), channel(_channel) {}

	uint32	packed;
	struct
	{
		uint32 pixel : 28;
		uint32 channel : 4;
	};
};

struct RayQueue
{
	Ray*		  rays;
	Hit*		  hits;
	float4*		  weights;
	union {
		float*		probs;
		uint32*		light_path_id;
	};
	uint32*		  pixels;
	float4*		  path_weights;
	uint32*		  size;

	// construct a copy of this queue with all addressed shifted by constant
	//
	RayQueue offset(const uint32 _count, uint32* _size) const
	{
		RayQueue r;
		r.rays			= rays			? rays			+ _count : NULL;
		r.hits			= hits			? hits			+ _count : NULL;
		r.weights		= weights		? weights		+ _count : NULL;
		r.probs			= probs			? probs			+ _count : NULL;
		r.pixels		= pixels		? pixels		+ _count : NULL;
		r.path_weights	= path_weights	? path_weights	+ _count : NULL;
		r.size			= _size;
		return r;
	}

	FERMAT_HOST_DEVICE
	uint32 append_slot() const { return cugar::atomic_add(size, 1u); }

	FERMAT_DEVICE
	uint32 warp_append_slot() const { return cugar::cuda::warp_increment(size); }

	FERMAT_HOST_DEVICE
	void append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float p)
	{
		uint32 slot = append_slot();

		rays[slot]		= ray;
		weights[slot]	= weight;
		probs[slot]		= p;
		pixels[slot]	= pixel.packed;
	}
	FERMAT_HOST_DEVICE
	void append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float p, float4 path_w)
	{
		uint32 slot = append_slot();

		rays[slot]			= ray;
		weights[slot]		= weight;
		probs[slot]			= p;
		pixels[slot]		= pixel.packed;
		path_weights[slot]	= path_w;
	}
	FERMAT_HOST_DEVICE
	void append(const PixelInfo pixel, const Ray& ray, const float4 weight, const uint32 path_id, float4 path_w)
	{
		uint32 slot = append_slot();

		rays[slot]			= ray;
		weights[slot]		= weight;
		light_path_id[slot] = path_id;
		pixels[slot]		= pixel.packed;
		path_weights[slot]	= path_w;
	}

	FERMAT_DEVICE
	void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float p, float4 path_w)
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]			= ray;
		weights[slot]		= weight;
		probs[slot]			= p;
		pixels[slot]		= pixel.packed;
		path_weights[slot]	= path_w;
	}
	FERMAT_DEVICE
	void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const uint32 path_id, float4 path_w)
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]			= ray;
		weights[slot]		= weight;
		light_path_id[slot]	= path_id;
		pixels[slot]		= pixel.packed;
		path_weights[slot]	= path_w;
	}
	FERMAT_DEVICE
	void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float p)
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]		= ray;
		weights[slot]	= weight;
		probs[slot]		= p;
		pixels[slot]	= pixel.packed;
	}
	FERMAT_DEVICE
	void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const uint32 path_id)
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]			= ray;
		weights[slot]		= weight;
		light_path_id[slot] = path_id;
		pixels[slot]		= pixel.packed;
	}
};