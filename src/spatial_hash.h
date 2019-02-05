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

#include <types.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>

///@addtogroup Fermat
///@{

///@addtogroup SpatialHashModule
///@{

/// Perform jittered spatial hashing
///
FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
uint64 spatial_hash(
	const cugar::Vector3f	P,
	const cugar::Vector3f   N,
	const cugar::Bbox3f		bbox,
	const float				samples[6],
	const float				cone_radius,
	const uint32			normal_bits = 4)
{
	// find the nearest power of 2 to represent (bbox[1] - bbox[0]) / (2.0f * cone_radius)
	const float world_extent = cugar::max_comp(bbox[1] - bbox[0]);
	const float  float_grid_size = cugar::max( world_extent / (2.0f * cone_radius), 1.0f );
	//const uint32       grid_size = cugar::next_power_of_two(uint32(float_grid_size));
	//const uint32   log_grid_size = cugar::log2(grid_size); // in [0,17] -> need 5 bits!
	const float   flog_grid_size = log2f( float_grid_size );
	const uint32   log_grid_size = uint32( flog_grid_size ); // in [0,17] -> need 5 bits!
	const float   rlog_grid_size = flog_grid_size - log_grid_size;
	const uint32  log_grid_size_i = log_grid_size + (samples[2] < rlog_grid_size ? 1u : 0u);
	const uint32       grid_size = 1u << log_grid_size_i;
	const cugar::Vector3f shading_loc = float(grid_size) * (P - bbox[0]) / world_extent;
	const uint3 shading_loc_i = make_uint3(
		uint32(cugar::max(cugar::round(shading_loc.x + samples[0] - 0.5f), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.y + samples[1] - 0.5f), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.z + samples[2] - 0.5f), 0.0f)));
	const cugar::Vector2f normal_jitter = cugar::Vector2f(samples[3], samples[4]) / float(1u << (normal_bits/2));
	cugar::Vector2f N_uv = cugar::uniform_sphere_to_square(N);
					N_uv.x = cugar::mod( N_uv.x + normal_jitter.x, 1.0f );
					N_uv.y = cugar::min( N_uv.y + normal_jitter.y, 1.0f );
	const uint32 comp_mask = (1u << 17) - 1u;
	const uint32 shading_normal_i = cugar::pack_vector(N_uv, normal_bits / 2);
	const uint64 shading_key =
		(uint64(shading_loc_i.x & comp_mask)  <<  0) |
		(uint64(shading_loc_i.y & comp_mask)  << 17) |
		(uint64(shading_loc_i.z & comp_mask)  << 34) |
		(uint64(log_grid_size_i)  << 51) |
		(uint64(shading_normal_i) << 56);
	return shading_key;
}

/// Perform jittered spatial hashing
///
FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
uint64 spatial_hash(
	const uint32			pixel_index,
	const cugar::Vector3f	P,
	const cugar::Vector3f   N,
	const cugar::Vector3f   T,
	const cugar::Vector3f   B,
	const cugar::Bbox3f		bbox,
	const float				samples[6],
	const float				cone_radius,
	const float             filter_radius,
	const uint32			normal_bits = 4)
{
	// find the nearest power of 2 to represent (bbox[1] - bbox[0]) / (2.0f * cone_radius)
	const float world_extent = cugar::max_comp(bbox[1] - bbox[0]);
	const float  float_grid_size = cugar::max( world_extent / (2.0f * cone_radius), 1.0f );
	//const uint32       grid_size = cugar::next_power_of_two(uint32(float_grid_size));
	//const uint32   log_grid_size = cugar::log2(grid_size); // in [0,17] -> need 5 bits!
	const float   flog_grid_size = log2f( float_grid_size );
	const uint32   log_grid_size = uint32( flog_grid_size ); // in [0,17] -> need 5 bits!
	const float   rlog_grid_size = flog_grid_size - log_grid_size;
	const uint32  log_grid_size_i = log_grid_size + (samples[5] < rlog_grid_size ? 1u : 0u);
	const uint32       grid_size  = 1u << log_grid_size_i;
#if 0
	const cugar::Vector3f shading_loc = float(grid_size) * (P - bbox[0]) / world_extent;
	const cugar::Vector2f rdisc = cugar::square_to_unit_disk( cugar::Vector2f(samples[0], samples[1]) ) * (world_extent / grid_size);
	const uint3 shading_loc_i = make_uint3(
	#if 0
		uint32(cugar::round(shading_loc.x + rdisc.x * T.x + rdisc.y * B.x)),
		uint32(cugar::round(shading_loc.y + rdisc.x * T.y + rdisc.y * B.y)),
		uint32(cugar::round(shading_loc.z + rdisc.x * T.z + rdisc.y * B.z)));
	#else
		uint32(cugar::max(cugar::round(shading_loc.x + samples[0] - 0.5f), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.y + samples[1] - 0.5f), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.z + samples[2] - 0.5f), 0.0f)));
	#endif
#else
	const cugar::Vector2f rdisc = (filter_radius * cone_radius) * cugar::square_to_unit_disk( cugar::Vector2f(samples[0], samples[1]) );
	const cugar::Vector3f shading_loc = float(grid_size) * (P + T * rdisc.x + B * rdisc.y - bbox[0]) / world_extent;
	const uint3 shading_loc_i = make_uint3(
		uint32(cugar::max(cugar::round(shading_loc.x), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.y), 0.0f)),
		uint32(cugar::max(cugar::round(shading_loc.z), 0.0f)));
#endif
#if 0
	if (pixel_index >= (1206 + 305 * 1600) &&
		pixel_index <= (1215 + 305 * 1600))
	{
		printf("\npixel %u\n  filter[%f]\n  grid_size[%f -> %u]\n  log[%f=(%u + %f) -> %u]\n  floc: %f, %f, %f\n  iloc: %u, %u, %u\n\n",
			pixel_index - (1206 + 305 * 1600),
			(filter_radius * 2.0f * cone_radius),
			float_grid_size,
			grid_size,
			flog_grid_size,
			log_grid_size,
			rlog_grid_size,
			log_grid_size_i,
			shading_loc.x,
			shading_loc.y,
			shading_loc.z,
			shading_loc_i.x,
			shading_loc_i.y,
			shading_loc_i.z);
	}
#endif
	const cugar::Vector2f normal_jitter = cugar::Vector2f(samples[3], samples[4]) / float(1u << (normal_bits/2));
	cugar::Vector2f N_uv = cugar::uniform_sphere_to_square(N);
					N_uv.x = cugar::mod( N_uv.x + normal_jitter.x, 1.0f );
					N_uv.y = cugar::min( N_uv.y + normal_jitter.y, 1.0f );
	const uint32 shading_normal_i = cugar::pack_vector(N_uv, normal_bits / 2);
	const uint32 comp_mask = (1u << 17) - 1u;
	const uint64 shading_key =
		(uint64(shading_loc_i.x & comp_mask)  <<  0) |
		(uint64(shading_loc_i.y & comp_mask)  << 17) |
		(uint64(shading_loc_i.z & comp_mask)  << 34) |
		(uint64(log_grid_size_i)  << 51) |
		(uint64(shading_normal_i) << 56);
	return shading_key;
}

/// Compute the LOD level used for jittered spatial hashing
///
FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
float spatial_hash_lod(
	const cugar::Vector3f	P,
	const cugar::Bbox3f		bbox,
	const float				cone_radius)
{
	// find the nearest power of 2 to represent (bbox[1] - bbox[0]) / (2.0f * cone_radius)
	const float world_extent = cugar::max_comp(bbox[1] - bbox[0]);
	const float  float_grid_size = cugar::max( world_extent / (2.0f * cone_radius), 1.0f );
	return log2f( float_grid_size );
}

///@} SpatialHashModule
///@} Fermat
