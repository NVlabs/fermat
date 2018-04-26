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

#include "MeshStorage.h"
#include <cugar/basic/vector.h>
#include <cugar/linalg/matrix.h>

__global__
void translate_group_kernel(
	MeshView 			mesh,
	const uint32		group_id,
	const float3		delta,
	uint32*				set)
{
	const uint32 begin	= mesh.group_offsets[group_id];
	const uint32 end	= mesh.group_offsets[group_id + 1];

	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	const uint32 tri_id = begin + thread_id;

	if (tri_id >= end)
		return;

	for (uint32 i = 0; i < 3; ++i)
	{
		const uint32 vertex_id = mesh.vertex_indices[tri_id * 3 + i];

		const uint32 word_id = vertex_id / 32u;
		const uint32 bit_id  = vertex_id & 31u;
		const uint32 bit_mask = 1u << bit_id;

		if ((cugar::atomic_or( set + word_id, bit_mask ) & bit_mask) == 0u)
		{
			float3* v = reinterpret_cast<float3*>(mesh.vertex_data) + vertex_id;
			v->x += delta.x;
			v->y += delta.y;
			v->z += delta.z;
		}
	}
}

// translate a given group
//
SUTILAPI void translate_group(
	DeviceMeshStorage&	mesh,
	const uint32		group_id,
	const float3		delta)
{
	// NOTE: device vector reads!
	const uint32 begin	= mesh.m_group_offsets[group_id];
	const uint32 end	= mesh.m_group_offsets[group_id + 1];

	const uint32 n_entries = end - begin;

	const uint32 block_dim = 128;
	const uint32 grid_dim  = cugar::divide_ri(n_entries,block_dim);

	cugar::caching_device_vector<uint32> set(cugar::divide_ri(mesh.getNumVertices(),32u));

	translate_group_kernel<<<grid_dim,block_dim>>>( mesh.view(), group_id, delta, cugar::raw_pointer(set) );
}
