/*
 * Fermat
 *
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cugar/basic/cuda/pointers.h>

/// given a hashing key, return the corresponding cell slot
///
FERMAT_DEVICE FERMAT_FORCEINLINE
uint32 ClusteredRLView::find_slot(const uint64 key)
{
	uint32 slot;
	hashmap.insert(key, cugar::hash(key/HashMap::BUCKET_SIZE), &slot);
	return slot;
}

// given a cell and a random number, sample an item
//
FERMAT_DEVICE FERMAT_FORCEINLINE
uint32 ClusteredRLView::sample(const uint32 cell_slot, const float z, float* pdf, uint32* out_cluster_idx) const
{
	const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

	//typedef const float*											float_pointer;
	//typedef const uint32*											uint_pointer;
	typedef cugar::cuda::load_pointer<float,cugar::cuda::LOAD_LDG>	float_pointer;
	typedef cugar::cuda::load_pointer<uint32,cugar::cuda::LOAD_LDG>	uint_pointer;

	// 1. sample a cluster according to the CDF
	float_pointer cdf		 = cdfs + cell_slot * cluster_count;
	const uint32 cluster_idx = cugar::upper_bound_index( cugar::min( z, one ) * cdf[cluster_count-1], cdf, cluster_count );
	const float  cdf_begin	 = cluster_idx ? cdf[cluster_idx - 1] : 0.0f;
	const float  cdf_end	 = cdf[cluster_idx];
	const float  cluster_pdf = cdf_end - cdf_begin;

	// 2. select a VTL uniformly within the cluster and sample that uniformly
	const float  cluster_z		= (z - cdf_begin) / cluster_pdf;
	const uint32 cluster_offset = uint_pointer(cluster_offsets)[cluster_idx];
	const uint32 cluster_size   = uint_pointer(cluster_offsets)[cluster_idx+1] - cluster_offset;
	const uint32 index			= cluster_offset + cugar::quantize( cugar::min( cluster_z, one ), cluster_size );

	// 3. compute the pdf
	*pdf = cluster_pdf / float(cluster_size);

	*out_cluster_idx = cluster_idx;

	return index;
}

// given a cell and an item's index, return the sampling pdf of that item
//
FERMAT_DEVICE FERMAT_FORCEINLINE
float ClusteredRLView::pdf(const uint32 cell_slot, const uint32 index) const
{
	// 1. find the cluster containing this index
	const float* cdf		 = cdfs + cell_slot * cluster_count;
	const uint32 cluster_idx = cugar::upper_bound_index( index, cluster_offsets + 1, cluster_count );
	const float  cdf_begin	 = cluster_idx ? cdf[cluster_idx - 1] : 0.0f;
	const float  cdf_end	 = cdf[cluster_idx];
	const float  cluster_pdf = cdf_end - cdf_begin;

	// 2. compute the cluster size
	const uint32 cluster_offset = cluster_offsets[cluster_idx];
	const uint32 cluster_size   = cluster_offsets[cluster_idx+1] - cluster_offset;

	// 3. compute the pdf
	return cluster_pdf / float(cluster_size);
}

// update the value corresponding to sampled cluster
//
FERMAT_DEVICE FERMAT_FORCEINLINE
void ClusteredRLView::update(const uint32 cell_slot, const uint32 cluster_idx, const float update_val, const float alpha)
{
	const float* pdf = pdfs + cell_slot * cluster_count;

	//while (1)
	for (uint32 i = 0; i < 5; ++i) // perform 5 attempts only!
	{
		const float cur = cugar::cuda::load<cugar::cuda::LOAD_VOLATILE>(pdf + cluster_idx);
		const float val = cur * (1.0f - alpha) + update_val * alpha;

		if (atomicCAS((uint32*)pdf + cluster_idx, cugar::binary_cast<uint32>(cur), cugar::binary_cast<uint32>(val)) == cugar::binary_cast<uint32>(cur))
			break;
	}
}

/// given a hashing key, return the corresponding cell slot
///
FERMAT_DEVICE FERMAT_FORCEINLINE
uint32 AdaptiveClusteredRLView::find_slot(const uint64 key)
{
	uint32 slot;
	hashmap.insert(key, cugar::hash(key/HashMap::BUCKET_SIZE), &slot);
	return slot;
}

// given a cell and a random number, sample an item
//
FERMAT_DEVICE FERMAT_FORCEINLINE
uint32 AdaptiveClusteredRLView::sample(const uint32 cell_slot, const float z, float* pdf, uint32* out_cluster_idx) const
{
	const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

	// find the actual cluster count
	const uint32 cluster_count = cluster_counts[cell_slot];
	const uint32* ends = cluster_ends + cell_slot * init_cluster_count;

	// 1. sample a cluster according to the CDF
	const float* cdf		 = cdfs + cell_slot * init_cluster_count;
	const uint32 cluster_idx = cugar::upper_bound_index( cugar::min( z, one ) * cdf[cluster_count-1], cdf, cluster_count );
	const float  cdf_begin	 = cluster_idx ? cdf[cluster_idx - 1] : 0.0f;
	const float  cdf_end	 = cdf[cluster_idx];
	const float  cluster_pdf = cdf_end - cdf_begin;

	// 2. select a VTL uniformly within the cluster and sample that uniformly
	const float  cluster_z		= (z - cdf_begin) / cluster_pdf;
	const uint32 cluster_offset = cluster_idx ? ends[cluster_idx-1] : 0u;
	const uint32 cluster_size   = ends[cluster_idx] - cluster_offset;
	const uint32 index			= cluster_offset + cugar::quantize( cugar::min( cluster_z, one ), cluster_size );

	// 3. compute the pdf
	*pdf = cluster_pdf / float(cluster_size);

	*out_cluster_idx = cluster_idx;

	return index;
}

// given a cell and an item's index, return the sampling pdf of that item
//
FERMAT_DEVICE FERMAT_FORCEINLINE
float AdaptiveClusteredRLView::pdf(const uint32 cell_slot, const uint32 index) const
{
	// find the actual cluster count
	const uint32 cluster_count = cluster_counts[cell_slot];
	const uint32* ends = cluster_ends + cell_slot * init_cluster_count;

	// 1. find the cluster containing this index
	const float* cdf		 = cdfs + cell_slot * init_cluster_count;
	const uint32 cluster_idx = cugar::upper_bound_index( index, ends, cluster_count );
	const float  cdf_begin	 = cluster_idx ? cdf[cluster_idx - 1] : 0.0f;
	const float  cdf_end	 = cdf[cluster_idx];
	const float  cluster_pdf = cdf_end - cdf_begin;

	// 2. compute the cluster size
	const uint32 cluster_offset = cluster_idx ? ends[cluster_idx-1] : 0u;
	const uint32 cluster_size   = ends[cluster_idx] - cluster_offset;

	// 3. compute the pdf
	return cluster_pdf / float(cluster_size);
}

// update the value corresponding to sampled cluster
//
FERMAT_DEVICE FERMAT_FORCEINLINE
void AdaptiveClusteredRLView::update(const uint32 cell_slot, const uint32 cluster_idx, const float update_val, const float alpha)
{
	const float* pdf = pdfs + cell_slot * init_cluster_count;

	//while (1)
	for (uint32 i = 0; i < 5; ++i) // perform 5 attempts only!
	{
		const float cur = cugar::cuda::load<cugar::cuda::LOAD_VOLATILE>(pdf + cluster_idx);
		const float val = cur * (1.0f - alpha) + update_val * alpha;

		if (atomicCAS((uint32*)pdf + cluster_idx, cugar::binary_cast<uint32>(cur), cugar::binary_cast<uint32>(val)) == cugar::binary_cast<uint32>(cur))
			break;
	}
}

