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
#include <ray.h>
#include <cugar/basic/cuda/warp_atomics.h>


///@addtogroup Fermat
///@{

///@addtogroup PTLib
///@{

// A class encapsulating the path tracing queues
//
struct PTRayQueue
{
	MaskedRay*	rays;		// rays
	Hit*		hits;		// ray hits
	float4*		weights;	// path weight
	float4*		weights_d;	// diffuse path weight
	float4*		weights_g;	// glossy path weight
	uint4*		pixels;		// path pixel info
	float2*		cones;		// ray cones
	uint32*		size;		// queue size

	FERMAT_HOST
	PTRayQueue() :
		rays(NULL),
		hits(NULL),
		weights(NULL),
		weights_d(NULL),
		weights_g(NULL),
		pixels(NULL),
		cones(NULL),
		size(NULL) {}

	FERMAT_DEVICE
	void warp_append(const uint32 pixel_info, const MaskedRay& ray, const float4 weight, const cugar::Vector2f cone = cugar::Vector2f(0), const uint32 vertex_info = uint32(-1), const uint32 nee_slot = uint32(-1))
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]		= ray;
		weights[slot]	= weight;
		pixels[slot]	= make_uint4( pixel_info, vertex_info, nee_slot, uint32(-1) );
		if (cones)
			cones[slot]		= cone;
	}

	FERMAT_DEVICE
	void warp_append(const uint32 pixel_info, const MaskedRay& ray, const float4 weight, const float4 weight_d, const float4 weight_g, const uint32 vertex_info = uint32(-1), const uint32 nee_slot = uint32(-1), const uint32 nee_cluster = uint32(-1))
	{
		const uint32 slot = cugar::cuda::warp_increment(size);

		rays[slot]		= ray;
		weights[slot]	= weight;

		if (weights_d)
			weights_d[slot]	= weight_d;
		if (weights_g)
			weights_g[slot]	= weight_g;

		pixels[slot]	= make_uint4( pixel_info, vertex_info, nee_slot, nee_cluster );
	}
};

///@} PTLib
///@} Fermat
