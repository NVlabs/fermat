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

#include <hashmap.h>
#include <cugar/bvh/bvh_node.h>
#include <cugar/basic/algorithms.h>
#include <cugar/basic/cuda/hash.h>

///@addtogroup Fermat
///@{

///@defgroup ClusteredRLModule	Clustered RL
///\par
/// This module implements novel methods for sampling from a large discrete set using adaptive clustering and  <a href="https://en.wikipedia.org/wiki/Reinforcement_learning">Reinforcement Learning</a>.
/// It implements two such samplers:
///\par
///  - a simple Clustered Reinforcement Learning sampler, using a predefined set of clusters over the discrete set
///  - an Adaptively Clustered Reinforcement Learning sampler, using an adaptively refined clustering over the discrete set
///@{

/// Device-side view class for the Clustered Reinforcement Learning sampler (\ref ClusteredRLModule).
///
struct ClusteredRLView
{
	typedef cugar::cuda::SyncFreeHashMap<uint64, uint32, 0xFFFFFFFFFFFFFFFFllu> HashMap;

	FERMAT_HOST_DEVICE
	ClusteredRLView() {}
	
	/// given a hashing key, return the corresponding cell slot
	///
	FERMAT_DEVICE
	uint32 find_slot(const uint64 key);

	/// given a cell and a random number, sample an item
	///
	FERMAT_DEVICE
	uint32 sample(const uint32 cell_slot, const float z, float* pdf, uint32* cluster_idx) const;

	/// given a cell and an item's index, return the sampling pdf of that item
	///
	FERMAT_DEVICE
	float pdf(const uint32 cell_slot, const uint32 index) const;

	/// update the value corresponding to sampled cluster
	///
	FERMAT_DEVICE
	void update(const uint32 cell_slot, const uint32 cluster_idx, const float val, const float alpha = 0.05f);

	HashMap			hashmap;
	float*			pdfs;
	float*			cdfs;
	const uint32*	cluster_offsets;
	uint32			cluster_count;
	uint32			hash_size;
};

/// Host-side class to handle the storage of a Clustered Reinforcement Learning sampler (\ref ClusteredRLModule).
///
struct ClusteredRLStorage
{
	typedef ClusteredRLView	view_type;

	ClusteredRLStorage() : hash_size(0), n_clusters(0), old_hash_size(0) {}

	uint64 needed_bytes(const uint32 _hash_size, const uint32 _n_clusters) const;

	void init(
		const uint32				_hash_size,
		const uint32				_n_clusters,
		const uint32*				_cluster_offsets);

	void resize(
		const uint32				_hash_size,
		const uint32				_n_clusters,
		const uint32*				_cluster_offsets);

	void update(const float bias = 0.1f);

	void clear();

	uint32 size() const;

	DeviceHashTable							hashmap;
	DomainBuffer<CUDA_BUFFER, float>		values;
	uint32									hash_size;
	uint32									n_clusters;
	const uint32* 							cluster_offsets;
	uint32									old_hash_size;
};

ClusteredRLView view(ClusteredRLStorage& storage);

/// Device-side view class for the Adaptively Clustered Reinforcement Learning sampler (\ref ClusteredRLModule).
///
struct AdaptiveClusteredRLView
{
	typedef cugar::cuda::SyncFreeHashMap<uint64, uint32, 0xFFFFFFFFFFFFFFFFllu> HashMap;

	FERMAT_HOST_DEVICE
	AdaptiveClusteredRLView() {}
	
	/// given a hashing key, return the corresponding cell slot
	///
	FERMAT_DEVICE
	uint32 find_slot(const uint64 key);

	/// given a cell and a random number, sample an item
	///
	FERMAT_DEVICE
	uint32 sample(const uint32 cell_slot, const float z, float* pdf, uint32* cluster_idx) const;

	/// given a cell and an item's index, return the sampling pdf of that item
	///
	FERMAT_DEVICE
	float pdf(const uint32 cell_slot, const uint32 index) const;

	/// update the value corresponding to sampled cluster
	///
	FERMAT_DEVICE
	void update(const uint32 cell_slot, const uint32 cluster_idx, const float val, const float alpha = 0.05f);

	HashMap			hashmap;
	float*			pdfs;
	float*			cdfs;
	const uint32*	cluster_ends;
	const uint32*	cluster_counts;
	uint32			init_cluster_count;
	uint32			hash_size;
};

/// Host-side class to handle the storage of an Adaptively Clustered Reinforcement Learning sampler (\ref ClusteredRLModule).
///
struct AdaptiveClusteredRLStorage
{
	typedef AdaptiveClusteredRLView	view_type;

	AdaptiveClusteredRLStorage() {}

	uint64 needed_bytes(const uint32 _hash_size, const uint32 _n_clusters) const;

	void init(
		const uint32				_hash_size,
		const cugar::Bvh_node_3d*	_nodes,
		const uint32*				_parents,
		const uint2*				_ranges,
		const uint32				_n_clusters,
		const uint32*				_cluster_indices,
		const uint32*				_cluster_offsets);

	void update(bool adaptive = true);

	void clear();

	uint32 size() const;

	DeviceHashTable							hashmap;
	DomainBuffer<CUDA_BUFFER, float>		values;
	const cugar::Bvh_node_3d*				nodes;
	const uint32*							parents;
	const uint2*							ranges;
	uint32									hash_size;
	DomainBuffer<CUDA_BUFFER, uint32>		cluster_counts;
	DomainBuffer<CUDA_BUFFER, uint32>		cluster_indices;
	DomainBuffer<CUDA_BUFFER, uint32>		cluster_ends;
	uint32									init_cluster_count;
	const uint32* 							init_cluster_indices;
	const uint32* 							init_cluster_offsets;
};

AdaptiveClusteredRLView view(AdaptiveClusteredRLStorage& storage);

///@} ClusteredRLModule
///@} Fermat

#include <clustered_rl_inline.h>
