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

#include <clustered_rl.h>
#include <cugar/basic/cuda/hash.h>
#include <cugar/basic/cuda/arch.h>
#include <cugar/basic/cuda/pointers.h>
#include <cugar/basic/cuda/timer.h>
#include <cub/cub.cuh>

#define BIAS 0.75f

template <uint32 BLOCKDIM>
__global__
void update_cdfs_kernel(const uint32 n_entries, const uint32 n_clusters, float* values, float* cdfs, const float bias, bool init)
{
	typedef cub::BlockScan<float, BLOCKDIM> BlockScan;

	// allocate shared memory for BlockScan
	__shared__ typename BlockScan::TempStorage temp_storage;

	const uint32 idx = blockIdx.x * n_clusters + threadIdx.x;

	if (init)
		values[idx] = 0.01f;

	// obtain a segment of consecutive items that are blocked across threads
	float thread_data = threadIdx.x < n_clusters ? values[idx] : 0.0f;
	float aggregate;

	// collectively compute the block-wide exclusive prefix sum
	BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, aggregate);

	// write out the value
	if (threadIdx.x < n_clusters)
		cdfs[idx] = (thread_data / aggregate) * (1.0f - bias) + float(threadIdx.x + 1) * bias / float(n_clusters);

	//if (threadIdx.x == n_clusters-1)
	//	printf("cdf[%u : %u] = %f -> %f\n", blockIdx.x, n_clusters-1, aggregate, cdfs[idx]);
}


template <uint32 BLOCKDIM>
__global__
void update_cdfs_kernel(const uint32 n_entries, const uint32 init_cluster_count, const uint32* cluster_counts, float* values, float* cdfs, bool init)
{
	typedef cub::BlockScan<float, BLOCKDIM> BlockScan;

	// allocate shared memory for BlockScan
	__shared__ typename BlockScan::TempStorage temp_storage;

	const uint32 slot = blockIdx.x;
	const uint32 idx  = blockIdx.x * init_cluster_count + threadIdx.x;

	const uint32 cluster_count = cluster_counts[ slot ];

	if (init)
		values[idx] = 0.01f;

	// obtain a segment of consecutive items that are blocked across threads
	float thread_data = threadIdx.x < cluster_count ? values[idx] : 0.0f;
	float aggregate;

	// collectively compute the block-wide exclusive prefix sum
	BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, aggregate);

	// write out the value
	if (threadIdx.x < cluster_count)
		cdfs[idx] = (thread_data / aggregate) * (1.0f - BIAS) + float(threadIdx.x + 1) * BIAS / float(cluster_count);
}

void update_cdfs(const uint32 n_entries, const uint32 n_clusters, float* values, float* cdfs, const float bias, bool init = false)
{
	if (n_entries)
	{
		if (n_clusters <= 128)			update_cdfs_kernel<128> <<< n_entries, 128 >>> (n_entries, n_clusters, values, cdfs, bias, init);
		else if (n_clusters <= 256)		update_cdfs_kernel<256> <<< n_entries, 256 >>> (n_entries, n_clusters, values, cdfs, bias, init);
		else if (n_clusters <= 512)		update_cdfs_kernel<512> <<< n_entries, 512 >>> (n_entries, n_clusters, values, cdfs, bias, init);
		else if (n_clusters <= 1024)	update_cdfs_kernel<1024> <<< n_entries, 1024 >>>(n_entries, n_clusters, values, cdfs, bias, init);
		else if (n_clusters <= 2048)	update_cdfs_kernel<2048> <<< n_entries, 2048 >>>(n_entries, n_clusters, values, cdfs, bias, init);
		else
		{
			fprintf(stderr, "unsupported number of vtl clusters: %u\n", n_clusters );
			exit(1);
		}
	}
	CUDA_CHECK(cugar::cuda::sync_and_check_error("ClusteredRL::update_cdfs"));
}

void update_cdfs(const uint32 n_entries, const uint32 init_cluster_count, const uint32* cluster_counts, float* values, float* cdfs, bool init = false)
{
	if (n_entries)
	{
		if (init_cluster_count <= 128)			update_cdfs_kernel<128> <<< n_entries, 128 >>> (n_entries, init_cluster_count, cluster_counts, values, cdfs, init);
		else if (init_cluster_count <= 256)		update_cdfs_kernel<256> <<< n_entries, 256 >>> (n_entries, init_cluster_count, cluster_counts, values, cdfs, init);
		else if (init_cluster_count <= 512)		update_cdfs_kernel<512> <<< n_entries, 512 >>> (n_entries, init_cluster_count, cluster_counts, values, cdfs, init);
		else if (init_cluster_count <= 1024)	update_cdfs_kernel<1024> <<< n_entries, 1024 >>>(n_entries, init_cluster_count, cluster_counts, values, cdfs, init);
		else
		{
			fprintf(stderr, "unsupported number of vtl clusters: %u\n", init_cluster_count );
			exit(1);
		}
	}
	CUDA_CHECK(cugar::cuda::sync_and_check_error("AdaptiveClusteredRL::update_cdfs"));
}

__global__
void init_clusters_kernel(
	const uint32	n_entries,
	const uint32	init_cluster_count, 
	const uint32*	init_cluster_indices,
	const uint32*	init_cluster_offsets,
	uint32*			cluster_counts,
	uint32*			cluster_indices,
	uint32*			cluster_ends)
{
	const uint32 slot = blockIdx.x;

	if (threadIdx.x == 0)
		cluster_counts[slot] = init_cluster_count;

	uint32* cta_indices = cluster_indices + slot * init_cluster_count;
	uint32* cta_ends    = cluster_ends    + slot * init_cluster_count;

	if (threadIdx.x < init_cluster_count)
	{
		cta_indices[threadIdx.x] = init_cluster_indices[threadIdx.x];
		cta_ends[threadIdx.x]    = init_cluster_offsets[threadIdx.x + 1];
	}
}

void init_clusters(
	const uint32	n_entries,
	const uint32	init_cluster_count, 
	const uint32*	init_cluster_indices,
	const uint32*	init_cluster_offsets,
	uint32*			cluster_counts,
	uint32*			cluster_indices,
	uint32*			cluster_ends)
{
	if (n_entries)
	{
		const uint32 block_dim = cugar::next_power_of_two(init_cluster_count);
		init_clusters_kernel <<< n_entries, block_dim >>> (n_entries, init_cluster_count, init_cluster_indices, init_cluster_offsets, cluster_counts, cluster_indices, cluster_ends);
	}
	CUDA_CHECK(cugar::cuda::sync_and_check_error("AdaptiveClusteredRL::init_clusters"));
}

void ClusteredRLStorage::init(const uint32 _hash_size, const uint32 _n_clusters, const uint32* _cluster_offsets)
{
	resize( _hash_size, _n_clusters, _cluster_offsets );
}

void ClusteredRLStorage::resize(const uint32 _hash_size, const uint32 _n_clusters, const uint32* _cluster_offsets)
{
	if (_hash_size != hash_size)
		hashmap.resize( _hash_size );

	if (_hash_size * _n_clusters > hash_size * n_clusters)
		values.resize( _hash_size * _n_clusters * 2 );

	hash_size  = _hash_size;
	n_clusters = _n_clusters;

	cluster_offsets = _cluster_offsets;

	clear();
}

uint64 ClusteredRLStorage::needed_bytes(const uint32 _hash_size, const uint32 _n_clusters) const
{
	return hashmap.needed_bytes(_hash_size ) + sizeof(float) * _hash_size * _n_clusters * 2;
}

void ClusteredRLStorage::clear()
{
	// initialize the DL cache
	hashmap.clear();

	// clear the pdf values
	//cudaMemset( values.ptr(), 0x00, sizeof(float) * hash_size * n_clusters );
	update_cdfs( hash_size, n_clusters, values.ptr(), values.ptr() + hash_size * n_clusters, 0.1f, true );

	old_hash_size = 0;
}

void ClusteredRLStorage::update(const float bias)
{
	const uint32 new_hash_size = hashmap.size();

	update_cdfs( new_hash_size, n_clusters, values.ptr(), values.ptr() + hash_size * n_clusters, bias, false );

	old_hash_size = new_hash_size;
}

uint32 ClusteredRLStorage::size() const
{
	return hashmap.size();
}

ClusteredRLView view(ClusteredRLStorage& storage)
{
	ClusteredRLView r;
	r.hash_size = storage.hash_size;
	r.hashmap = ClusteredRLView::HashMap(
		storage.hash_size,
		storage.hashmap.m_keys.ptr(),
		storage.hashmap.m_unique.ptr(),
		storage.hashmap.m_slots.ptr(),
		storage.hashmap.m_size.ptr()
	);
	r.cluster_count		= storage.n_clusters;
	r.cluster_offsets	= storage.cluster_offsets;
	r.pdfs = storage.values.ptr();
	r.cdfs = storage.values.ptr() + storage.hash_size * storage.n_clusters;
	return r;
}



struct ClusterParams
{
	uint32						cluster_count;
	uint32*						cluster_indices;
	uint32*						cluster_ends;
	float*						cluster_powers;

	const cugar::Bvh_node_3d*	bvh_nodes;
	const uint2*				bvh_ranges;
	const uint32*				bvh_parents;
};

//
// Given a global BVH (e.g. over a set of lights, although this algorithm is completely agnostic to the actual content),
// and a cut into it expressed by the ordered list of its terminal nodes, here called "clusters", each with an associated score
// (e.g. representing the "power" of the lights in the cluster), assign a CTA to the problem of splitting the cluster
// with the highest score, and collapsing the internal node with the lowest one.
// The split/collapse pair is only performed if the lowest internal node's score is lower than the highest cluster score.
//
template <uint32 BLOCK_DIM>
__device__
void cta_split_and_collapse(
	const ClusterParams&	in,
	uint32&					out_cluster_count,
	uint32*					out_cluster_indices,
	uint32*					out_cluster_ends,
	float*					out_cluster_powers)
{
	// collect the parents, deduplicating them with a hash-map
	const uint32 HASH_SIZE = BLOCK_DIM * 2;
	typedef cugar::cuda::BlockHashMap<uint32,uint32,BLOCK_DIM,HASH_SIZE> HashMap;
	typedef cub::BlockReduce<float, BLOCK_DIM>	BlockReduce;
	typedef cub::BlockScan<uint32, BLOCK_DIM>	BlockScan;

	__shared__ 
	union
	{
		typename HashMap::TempStorage		hashmap;
		typename BlockReduce::TempStorage	reduce;
		typename BlockScan::TempStorage		scan;
	} temp_storage;

	__shared__ float parents_vals[BLOCK_DIM];
	
	HashMap hashmap( temp_storage.hashmap );

	// fetch this thread's leaf
	uint32 cluster_index = threadIdx.x < in.cluster_count ? in.cluster_indices[threadIdx.x] : uint32(-1);
	float  cluster_power = threadIdx.x < in.cluster_count ? in.cluster_powers[threadIdx.x]  : float(0.0f);

	// check if this cluster is an actual bvh leaf
	bool is_bvh_leaf = threadIdx.x < in.cluster_count ? in.bvh_nodes[cluster_index].is_leaf() : true;

	// assign zero power to the actual bvh leaves, so as to make sure they are not candidates for splitting
	float non_leaf_cluster_power = is_bvh_leaf ? 0.0f : cluster_power;

	if (threadIdx.x < in.cluster_count)
	{
		const uint32 parent = cugar::cuda::load<cugar::cuda::LOAD_LDG>(&in.bvh_parents[ cluster_index ]);
		hashmap.insert( parent, cugar::hash(parent) );
	}

	// clear the values bound to parents, as it will be needed later on...
	parents_vals[threadIdx.x] = 0.0f;

	// synchronize all threads
	__syncthreads();

	const uint32 parent_count = hashmap.size();
	const uint32* parents = hashmap.unique;

	//if (threadIdx.x == 0)
	//	printf("%u clusters --> %u parents\n", in.cluster_count, parent_count);

	// accumulate the values of the parents using shared memory atomics
	if (threadIdx.x < in.cluster_count)
	{
		uint32 parent = cugar::cuda::load<cugar::cuda::LOAD_LDG>(&in.bvh_parents[ cluster_index ]);

		while (parent != uint32(-1))
		{
			const uint32 slot = hashmap.find(parent, cugar::hash(parent)); // for 10k CTAs, this costs 0.8ms out of 1.73ms
			if (slot != 0xFFFFFFFFu)
				cugar::atomic_add( parents_vals + slot, cluster_power );

			parent = cugar::cuda::load<cugar::cuda::LOAD_LDG>(&in.bvh_parents[ parent ]);
		}
	}

	// synchronize all threads
	__syncthreads();

	uint32 parent_index = threadIdx.x < parent_count ? parents[ threadIdx.x ]      : uint32(-1);
	float  parent_power = threadIdx.x < parent_count ? parents_vals[ threadIdx.x ] : float(1.0e16f);
	//if (blockIdx.x == 0 && parent_index != uint32(-1))
	//	printf("parent[%u] = %f\n", parent_index, parent_power);

	// NOTE:
	// after this point the 'parents array IS NO LONGER USABLE - as it is aliased with the temporary storage for reduce/scan

	__shared__ float min_parent_power;
	__shared__ float max_cluster_power;
	__shared__ uint32 min_parent_index;
	__shared__ uint32 max_cluster_index;

	// find the minimum parent's power (note that the result is only valid for thread0)
	{
		float result = BlockReduce(temp_storage.reduce).Reduce(parent_power, cub::Min(), parent_count);
		if (threadIdx.x == 0)
			min_parent_power = result;
	}

	__syncthreads();

	// find the maximum leaf's power (note that the result is only valid for thread0)
	{
		float result = BlockReduce(temp_storage.reduce).Reduce(non_leaf_cluster_power, cub::Max(), in.cluster_count);
		if (threadIdx.x == 0)
			max_cluster_power = result;
	}

	__syncthreads();

	if (min_parent_power == parent_power)
		min_parent_index  = parent_index;

	if (max_cluster_power == non_leaf_cluster_power)
		max_cluster_index  = cluster_index;

	__syncthreads();

	__shared__ uint2 collapsed_range;

	if (threadIdx.x == 0)
	{
		// if the minimum parent's power is less than the maximum leaf's power,
		// we'll collapse the parent, and expand the leaf
		if (min_parent_power < max_cluster_power)
		{
			// fetch the range of primitives of the parent we want to collapse
			collapsed_range = in.bvh_ranges[min_parent_index];
		}
		else
			collapsed_range = make_uint2(0,0);
	}

	__syncthreads();

	if (collapsed_range.x != collapsed_range.y) // this branch is uniform across the entire CTA
	{
		const uint2 cluster_range = threadIdx.x < in.cluster_count ? in.bvh_ranges[cluster_index] : make_uint2(0xFFFFFFFFu,0xFFFFFFFFu);

		bool is_max_cluster		  = max_cluster_index == cluster_index;
		bool keep_cluster		  = cluster_range.x != 0xFFFFFFFFu && (cluster_range.x < collapsed_range.x || cluster_range.x >= collapsed_range.y);
		bool last_collapsed_leaf  = cluster_range.y == collapsed_range.y;

		// output only the leaves whose range is not contained in the collapsed range, excluding the one we want to expand
		uint32 out_counter =
			is_max_cluster		?	2u :
			keep_cluster		?	1u :
			last_collapsed_leaf ?	1u :
									0u;

		// Collectively compute the block-wide exclusive prefix sum
	    BlockScan(temp_storage.scan).ExclusiveSum(out_counter, out_counter, out_cluster_count);

		cugar::cuda::store_pointer<uint32,cugar::cuda::STORE_CS> out_indices(out_cluster_indices);
		cugar::cuda::store_pointer<uint32,cugar::cuda::STORE_CS> out_ends(out_cluster_ends);
		cugar::cuda::store_pointer<float, cugar::cuda::STORE_CS> out_powers(out_cluster_powers);

		if (is_max_cluster)
		{
			// split this leaf
			const uint32 child_offset = in.bvh_nodes[cluster_index].get_child(0u);

			const uint2 child_range = in.bvh_ranges[child_offset + 0u];

			out_indices[out_counter + 0u]	= child_offset + 0u;
			out_indices[out_counter + 1u]	= child_offset + 1u;
			out_ends[out_counter + 0u]		= child_range.y;	// assume the second child starts where the first ends
			out_ends[out_counter + 1u]		= cluster_range.y;
			out_powers[out_counter + 0u]	= cluster_power * 0.5f;
			out_powers[out_counter + 1u]	= cluster_power * 0.5f;
		}
		else if (keep_cluster || last_collapsed_leaf)
		{
			// keep this leaf as is or replace it with the collapsed node
			out_indices[out_counter]		= last_collapsed_leaf ? min_parent_index : cluster_index;
			out_ends[out_counter]			= cluster_range.y;
			out_powers[out_counter]			= last_collapsed_leaf ? min_parent_power : cluster_power;
		}
	}
	else
	{
		// NOTE: there's actually nothing to do, as we are modifying the input in place...
	#if 0
		// copy the input onto the output
		out_cluster_count = in.cluster_count;

		if (threadIdx.x < in.cluster_count)
		{
			out_cluster_indices[threadIdx.x]	= cluster_index;
			out_cluster_ends[threadIdx.x]		= in.bvh_ranges[cluster_index].y;
			out_cluster_powers[threadIdx.x]		= cluster_power;
		}
	#endif
	}
}

struct SplitKernelParams
{
	uint32						n_entries;
	uint32						init_cluster_count;

	uint32*						cluster_counts;
	uint32*						cluster_indices;
	uint32*						cluster_ends;
	float*						cluster_powers;

	const cugar::Bvh_node_3d*	bvh_nodes;
	const uint2*				bvh_ranges;
	const uint32*				bvh_parents;
};

template <uint32 BLOCK_DIM>
__global__
void split_and_collapse_kernel(
	const SplitKernelParams params)
{
	const uint32 slot = blockIdx.x;

	ClusterParams cta_params;
	cta_params.cluster_count		= params.cluster_counts[slot];
	cta_params.cluster_indices		= params.cluster_indices + slot * params.init_cluster_count;
	cta_params.cluster_ends			= params.cluster_ends + slot * params.init_cluster_count; 
	cta_params.cluster_powers		= params.cluster_powers + slot * params.init_cluster_count;
	cta_params.bvh_nodes			= params.bvh_nodes;
	cta_params.bvh_ranges			= params.bvh_ranges;
	cta_params.bvh_parents			= params.bvh_parents;

	cta_split_and_collapse<BLOCK_DIM>(
		cta_params,
		params.cluster_counts[slot],								// NOTE: this is overwritten in place!
		cta_params.cluster_indices,									// NOTE: this is overwritten in place!
		cta_params.cluster_ends,									// NOTE: this is overwritten in place!
		cta_params.cluster_powers									// NOTE: this is overwritten in place!
	);
}

void split_and_collapse(
	const SplitKernelParams params)
{
  //#define TIMING
  #ifdef TIMING
	cugar::cuda::Timer timer;
	timer.start();
	const uint32 n_tests = 10;
	for (uint32 i = 0; i < n_tests; ++i)
  #endif
	{
		if (params.n_entries)
		{
			if (params.init_cluster_count <= 128)		split_and_collapse_kernel<128> <<< params.n_entries, 128 >>> (params);
			else if (params.init_cluster_count <= 256)	split_and_collapse_kernel<256> <<< params.n_entries, 256 >>> (params);
			else if (params.init_cluster_count <= 512)	split_and_collapse_kernel<512> <<< params.n_entries, 512 >>> (params);
			else if (params.init_cluster_count <= 1024)	split_and_collapse_kernel<1024> <<< params.n_entries, 1024 >>> (params);
			else
			{
				fprintf(stderr, "unsupported number of vtl clusters: %u\n", params.init_cluster_count );
				exit(1);
			}
		}
	}
	CUDA_CHECK(cugar::cuda::sync_and_check_error("AdaptiveClusteredRL::split_and_collapse"));
  #ifdef TIMING
	timer.stop();
	fprintf(stderr, "\nsplit %u: %fms\n", params.n_entries, 1000.0f * timer.seconds()/n_tests);
  #endif
}

uint64 AdaptiveClusteredRLStorage::needed_bytes(const uint32 _hash_size, const uint32 _n_clusters) const
{
	return
		hashmap.needed_bytes(_hash_size ) +
		sizeof(float) * _hash_size * _n_clusters * 2 +
		sizeof(uint32) * (_hash_size + _hash_size * _n_clusters * 2);
}

void AdaptiveClusteredRLStorage::init(
		const uint32				_hash_size,
		const cugar::Bvh_node_3d*	_nodes,
		const uint32*				_parents,
		const uint2*				_ranges,
		const uint32				_n_clusters,
		const uint32*				_cluster_indices,
		const uint32*				_cluster_offsets)
{
	nodes   = _nodes;
	parents = _parents;
	ranges  = _ranges;

	hash_size  = _hash_size;
	init_cluster_count   = _n_clusters;
	init_cluster_offsets = _cluster_offsets;
	init_cluster_indices = _cluster_indices;

	hashmap.resize( hash_size );
	values.resize( hash_size * init_cluster_count * 2 );
	cluster_counts.resize( hash_size );
	cluster_indices.resize( hash_size * init_cluster_count );
	cluster_ends.resize( hash_size * init_cluster_count );

	// initialize the DL cache
	hashmap.clear();

	// initialize the cluster counts and offsets
	init_clusters(hash_size, init_cluster_count, init_cluster_indices, init_cluster_offsets, cluster_counts.ptr(), cluster_indices.ptr(), cluster_ends.ptr());

	update_cdfs( hash_size, init_cluster_count, cluster_counts.ptr(), values.ptr(), values.ptr() + hash_size * init_cluster_count, true );
	//update_cdfs( hash_size, init_cluster_count, values.ptr(), values.ptr() + hash_size * init_cluster_count, true );
}

void AdaptiveClusteredRLStorage::update(bool adaptive)
{
	const uint32 size = hashmap.m_size[0];
	if (adaptive)
	{
		SplitKernelParams params;

		params.n_entries			= size;
		params.init_cluster_count	= init_cluster_count;
		params.cluster_counts		= cluster_counts.ptr();
		params.cluster_indices		= cluster_indices.ptr();
		params.cluster_ends			= cluster_ends.ptr();
		params.cluster_powers		= values.ptr();
		params.bvh_nodes			= nodes;
		params.bvh_ranges			= ranges;
		params.bvh_parents			= parents;

		split_and_collapse( params );
	}
	update_cdfs( size, init_cluster_count, cluster_counts.ptr(), values.ptr(), values.ptr() + hash_size * init_cluster_count, false );
}

void AdaptiveClusteredRLStorage::clear()
{
	// initialize the DL cache
	hashmap.clear();

	// initialize the cluster counts and offsets
	init_clusters(hash_size, init_cluster_count, init_cluster_indices, init_cluster_offsets, cluster_counts.ptr(), cluster_indices.ptr(), cluster_ends.ptr());

	update_cdfs( hash_size, init_cluster_count, cluster_counts.ptr(), values.ptr(), values.ptr() + hash_size * init_cluster_count, true );
}

uint32 AdaptiveClusteredRLStorage::size() const
{
	return hashmap.size();
}

AdaptiveClusteredRLView view(AdaptiveClusteredRLStorage& storage)
{
	AdaptiveClusteredRLView r;
	r.hash_size = storage.hash_size;
	r.hashmap = ClusteredRLView::HashMap(
		storage.hash_size,
		storage.hashmap.m_keys.ptr(),
		storage.hashmap.m_unique.ptr(),
		storage.hashmap.m_slots.ptr(),
		storage.hashmap.m_size.ptr()
	);
	r.init_cluster_count	= storage.init_cluster_count;
	r.cluster_counts		= storage.cluster_counts.ptr();
	r.cluster_ends			= storage.cluster_ends.ptr();
	r.pdfs = storage.values.ptr();
	r.cdfs = storage.values.ptr() + storage.hash_size * storage.init_cluster_count;
	return r;
}