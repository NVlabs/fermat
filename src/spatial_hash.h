/*
 * Fermat
 *
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <types.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>

FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
uint64 spatial_hash(
	const cugar::Vector3f	P,
	const cugar::Vector3f   N,
	const cugar::Bbox3f		bbox,
	const float				samples[3],
	const float				cone_radius,
	const float             filter_scale,
	const uint32			normal_bits = 8)
{
	// find the nearest power of 2 to represent (bbox[1] - bbox[0]) / (filter_scale * 2.0f * cone_radius)
	const float world_extent = cugar::max_comp(bbox[1] - bbox[0]);
	const float  float_grid_size = cugar::max( world_extent / (filter_scale * 2.0f * cone_radius), 1.0f );
	const uint32       grid_size = cugar::next_power_of_two(uint32(float_grid_size));
	const uint32   log_grid_size = cugar::log2(grid_size); // in [0,17] -> need 5 bits!
	const cugar::Vector3f shading_loc = float(grid_size) * (P - bbox[0]) / world_extent;
	const uint3 shading_loc_i = make_uint3(
		uint32(shading_loc.x + samples[0] - 0.5f),
		uint32(shading_loc.y + samples[1] - 0.5f),
		uint32(shading_loc.z + samples[2] - 0.5f));
	const cugar::Vector2f normal_jitter = cugar::Vector2f(samples[3], samples[4]) / float(1u << (normal_bits/2));
	const uint32 shading_normal_i = cugar::pack_vector(cugar::uniform_sphere_to_square(N) + normal_jitter, normal_bits / 2);
	const uint64 shading_key =
		(uint64(shading_loc_i.x)  <<  0) |
		(uint64(shading_loc_i.y)  << 17) |
		(uint64(shading_loc_i.z)  << 34) |
		(uint64(log_grid_size)    << 51) |
		(uint64(shading_normal_i) << 56);
	return shading_key;
}
