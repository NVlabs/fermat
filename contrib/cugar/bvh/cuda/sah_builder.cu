/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nih/bvh/cuda/sah_builder.h>

namespace nih {
namespace cuda {
namespace sah {

void inclusive_segmented_scan(
    uint2* dest,
    uint2* src,
    uint32* flags,
    const uint32 n_objects);

void exclusive_segmented_scan(
    uint2* dest,
    uint2* src,
    uint32* flags,
    const uint32 n_objects);

void radix_sort(
    const uint32    n_elements,
    uint16*         keys,
    uint32*         values,
    uint16*         keys_tmp,
    uint32*         values_tmp);

void radix_sort(
    const uint32    n_elements,
    uint32*         keys,
    uint32*         values,
    uint32*         keys_tmp,
    uint32*         values_tmp);

void radix_sort(
    const uint32    n_elements,
    uint64*         keys,
    uint32*         values,
    uint64*         keys_tmp,
    uint32*         values_tmp);

template <typename Integer>
struct invalid_key
{
    FORCE_INLINE NIH_HOST_DEVICE bool operator() (const Integer key_centroid) const
    {
        const uint32 key = uint32( key_centroid >> 10 );
        return key == uint32(-1);
    }
};
template <>
struct invalid_key<uint32>
{
    FORCE_INLINE NIH_HOST_DEVICE bool operator() (const uint32 key_centroid) const
    {
        const uint32 mask = ~1023u;
        return (key_centroid & mask) == mask;
    }
};
template <>
struct invalid_key<uint16>
{
    FORCE_INLINE NIH_HOST_DEVICE bool operator() (const uint16 key_centroid) const
    {
        const uint16 mask = ~uint16(1023);
        return (key_centroid & mask) == mask;
    }
};
template <typename Integer>
struct extract_key
{
    typedef uint32 result_type;

    FORCE_INLINE NIH_HOST_DEVICE uint32 operator() (const Integer key_centroid) const
    {
        return uint32( key_centroid >> 10 );
    }
};

struct different_keys
{
    typedef uint32 result_type;

    FORCE_INLINE NIH_HOST_DEVICE uint32 operator() (const thrust::tuple<uint32,uint32> op) const
    {
        return thrust::get<0>( op ) != thrust::get<1>( op ) ? 1u : 0u;
    }
};

/// utility function to compute the area of a compressed bbox
FORCE_INLINE NIH_HOST_DEVICE float area(const uint2 bbox)
{
    uint32 x = (((bbox.y & (1023u <<  0)) - (bbox.x & (1023u <<  0))) >>  0) + 1;
    uint32 y = (((bbox.y & (1023u << 10)) - (bbox.x & (1023u << 10))) >> 10) + 1;
    uint32 z = (((bbox.y & (1023u << 20)) - (bbox.x & (1023u << 20))) >> 20) + 1;
    return float(x*y + x*z + y*z) / float(1u << 30);
}

///
/// Functor to pack an object's key and its compressed centroid along the
/// longest axis of the parent's bbox.
///
template <typename Integer>
struct Key_centroid_packer
{
    typedef thrust::tuple<uint32,uint2>    argument_type;
    typedef Integer                        result_type;

    FORCE_INLINE NIH_HOST_DEVICE Integer operator() (const argument_type op) const
    {
        const uint32 key = thrust::get<0>( op );
        if (key == uint32(-1))
            return Integer(key) << 10;

        const uint2 bbox = thrust::get<1>( op );

        const uint32 axis = bbox.x >> 30;

        const uint32 centroid =
            axis == 0 ? ((bbox.x & (1023u <<  0)) + (bbox.y & (1023u <<  0))) >> 1:
            axis == 1 ? ((bbox.x & (1023u << 10)) + (bbox.y & (1023u << 10))) >> 11:
                        ((bbox.x & (1023u << 20)) + (bbox.y & (1023u << 20))) >> 21;

        return (Integer(key) << 10) | Integer(centroid);
    }
};

///
/// Functor to compute the cost of each possible split
///
struct Cost_functor
{
    typedef uint32 argument_type;
    typedef thrust::tuple<float,uint32> result_type;

    Cost_functor(
        const uint32*        keys,
        const uint32*        heads,
        const uint2*         bounds_l,
        const uint2*         bounds_r) :
        m_keys( keys ),
        m_heads( heads ),
        m_bounds_l( bounds_l ),
        m_bounds_r( bounds_r ) {}

    FORCE_INLINE NIH_DEVICE result_type operator() (const uint32 i) const
    {
        const uint32 key   = m_keys[i] - 1;
        const uint32 begin = m_heads[key];
        const uint32 end   = m_heads[key+1];

        const uint32 size_l = i - begin + 1;
        const uint32 size_r = end - begin - size_l;

        const float cost_l = size_l * area( m_bounds_l[i] );
        const float cost_r = size_r * area( m_bounds_r[i] );
        const float cost = (size_l == 0 || size_r == 0) ?
            1.0e9f : cost_l + cost_r;

        return thrust::make_tuple( cost, size_l );
    }

    const uint32* m_keys;
    const uint32* m_heads;
    const uint2*  m_bounds_l;
    const uint2*  m_bounds_r;
};

///
/// Binary functor to merge a pair of quantized bboxes
///
struct merge
{
    typedef uint2 result_type;

    FORCE_INLINE NIH_HOST_DEVICE uint2 operator() (const uint2 op1, const uint2 op2) const
    {
        return make_uint2(
            nih::min( op1.x & (1023u <<  0), op2.x & (1023u <<  0) ) |
            nih::min( op1.x & (1023u << 10), op2.x & (1023u << 10) ) |
            nih::min( op1.x & (1023u << 20), op2.x & (1023u << 20) ),
            nih::max( op1.y & (1023u <<  0), op2.y & (1023u <<  0) ) |
            nih::max( op1.y & (1023u << 10), op2.y & (1023u << 10) ) |
            nih::max( op1.y & (1023u << 20), op2.y & (1023u << 20) ) );
    }

    // identity operator
	FORCE_INLINE NIH_HOST_DEVICE uint2 operator()() { return make_uint2( 0xFFFFFFFFu, 0 ); }
};

///
/// Binary functor to select the cheapest of two split candidates
///
struct Best_split_functor
{
    typedef thrust::tuple<float,uint32> argument_type;
    typedef thrust::tuple<float,uint32> result_type;

    FORCE_INLINE NIH_DEVICE result_type operator() (
        const argument_type op1,
        const argument_type op2)
    {
        const float cost1 = thrust::get<0>( op1 );
        const float cost2 = thrust::get<0>( op2 );

        return (cost1 < cost2) ?
            thrust::make_tuple( cost1, thrust::get<1>( op1 ) ) :
            thrust::make_tuple( cost2, thrust::get<1>( op2 ) );
    }
};

/// CUDA kernel:
/// evaluate the new splits, choosing which ones to keep and which ones to discard.
__global__ void eval_splits_kernel(
    const uint32    n_nodes,
          uint32*   split_planes,
    const float*    split_costs,
    const uint32*   segment_heads,
    uint32*         out_splits,
    const uint32    max_leaf_size,
    const float     max_cost)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_nodes;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_nodes)
            return;

        const uint32 node_begin  = segment_heads[index];
        const uint32 node_end    = segment_heads[index+1];
        const float  split_cost  = split_costs[index];
              uint32 split_index = split_planes[index] + node_begin;

        // place the split in the middle if its cost is too high
        if (split_cost > float(node_end - node_begin) * max_cost)
            split_index = (node_begin + node_end)/2;

        bool valid_split = (split_index > node_begin && split_index < node_end) && (node_end - node_begin > max_leaf_size);

        if (valid_split)
        {
            split_planes[ index ] = split_index;
            out_splits[ index ] = 1;
        }
        else
        {
            split_planes[ index ] = node_end;
            out_splits[ index ] = 0;
        }
    }
}

/// evaluate the new splits, choosing which ones to keep and which ones to discard.
void eval_splits(
    const uint32    n_nodes,
          uint32*   split_planes,
    const float*    split_costs,
    const uint32*   segment_heads,
    uint32*         out_splits,
    const uint32    max_leaf_size,
    const float     max_cost)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(eval_splits_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_nodes + BLOCK_SIZE-1) / BLOCK_SIZE );

    eval_splits_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_nodes,
        split_planes,
        split_costs,
        segment_heads,
        out_splits,
        max_leaf_size,
        max_cost );

    cudaThreadSynchronize();
}

/// CUDA kernel:
/// assign objects to their new nodes
__global__ void assign_objects_kernel(
    const uint32    n_objects,
    const uint32    n_leaves,
    const uint32*   order,
    const uint32*   segment_keys,
    const uint32*   split_index,
    const uint32*   allocation_map,
    uint32*         segment_ids,
    uint32*         leaf_ids)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_objects)
            return;

        // get the segment this object belongs to
        const uint32 key = segment_keys[ index ]-1;

        const uint32 address      = allocation_map[key];
        const uint32 prev_address = key ? allocation_map[key-1] : 0u;

        const bool is_split    = address > prev_address;
        const bool right_child = is_split && index >= split_index[ key ];

        // assign the object to its new segment
        segment_ids[ order[ index ] ] = is_split ? 1 + prev_address*2 + (right_child ? 1 : 0) : uint32(-1);

        // check whether the segment was split: if not, the node assignment doesn't change
        if (is_split == false)
        {
            // assign the object to its new leaf
            leaf_ids[ order[ index ] ] = n_leaves + (key - address);
            continue;
        }
    }
}

/// assign objects to their new nodes
void assign_objects(
    const uint32    n_objects,
    const uint32    n_leaves,
    const uint32*   order,
    const uint32*   segment_keys,
    const uint32*   split_index,
    const uint32*   allocation_map,
    uint32*         segment_ids,
    uint32*         leaf_ids)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(assign_objects_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    assign_objects_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        n_leaves,
        order,
        segment_keys,
        split_index,
        allocation_map,
        segment_ids,
        leaf_ids );

    cudaThreadSynchronize();
}

FORCE_INLINE NIH_HOST_DEVICE void decompress(Bin& bin, const float4 bmin, const float3 delta, const uint2 bounds, const uint32 skip_node)
{
    const float norm = 1.0f / 1024.0f;

    const uint3 l = make_uint3(
        (bounds.x & (1023u <<  0)) >> 0,
        (bounds.x & (1023u << 10)) >> 10,
        (bounds.x & (1023u << 20)) >> 20 );

    bin.bmin = make_float4(
        ((float(l.x) * norm) * delta.x) + bmin.x,
        ((float(l.y) * norm) * delta.y) + bmin.y,
        ((float(l.z) * norm) * delta.z) + bmin.z,
        0.0f );

    const uint3 r = make_uint3(
        (bounds.y & (1023u <<  0)) >> 0,
        (bounds.y & (1023u << 10)) >> 10,
        (bounds.y & (1023u << 20)) >> 20 );

    bin.bmax = make_float4(
        ((float(r.x+1) * norm) * delta.x) + bmin.x,
        ((float(r.y+1) * norm) * delta.y) + bmin.y,
        ((float(r.z+1) * norm) * delta.z) + bmin.z,
        binary_cast<float>(skip_node) );
}

/// CUDA kernel:
/// compute the bounding box of the output segments
__global__ void compute_bins_kernel(
    const uint32    n_segments,
    const uint32    n_nodes,
    const uint32    n_leaves,
    const uint32    input_node_offset,
    const uint32*   split_index,
    const uint32*   allocation_map,
    const Bin*      in_bins,
    const uint2*    bounds_l,
    const uint2*    bounds_r,
    Bin*            out_bins,
    Bvh_node*       bvh_nodes)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_segments;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_segments)
            return;

        const uint32 address      = allocation_map[ index ];
        const uint32 prev_address = index ? allocation_map[ index-1 ] : 0u;

        const bool is_split = address > prev_address;

        const Bin&   bin  = in_bins[ index ];
        const float4 bmax = bin.bmax;

        if (is_split)
        {
            const float4 bmin = bin.bmin;
            const float3 delta = make_float3( bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z );

            decompress( out_bins[ prev_address*2     ], bmin, delta, bounds_l[ split_index[index] ], binary_cast<uint32>(n_nodes + prev_address*2 + 1) );
            decompress( out_bins[ prev_address*2 + 1 ], bmin, delta, bounds_r[ split_index[index] ], binary_cast<uint32>(bmax.w) );

            // write the parent node
            bvh_nodes[ input_node_offset + index ] = Bvh_node( Bvh_node::kInternal, n_nodes + prev_address*2, binary_cast<uint32>(bmax.w) );
        }
        else
        {
            // write the leaf node
            bvh_nodes[ input_node_offset + index ] = Bvh_node( Bvh_node::kLeaf, n_leaves + index - prev_address, binary_cast<uint32>(bmax.w) );
        }
    }
}

/// compute the bounding box of the output segments
void compute_bins(
    const uint32    n_segments,
    const uint32    n_nodes,
    const uint32    n_leaves,
    const uint32    input_node_offset,
    const uint32*   split_index,
    const uint32*   allocation_map,
    const Bin*      in_bins,
    const uint2*    bounds_l,
    const uint2*    bounds_r,
    Bin*            out_bins,
    Bvh_node*       bvh_nodes)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(compute_bins_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_segments + BLOCK_SIZE-1) / BLOCK_SIZE );

    compute_bins_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_segments,
        n_nodes,
        n_leaves,
        input_node_offset,
        split_index,
        allocation_map,
        in_bins,
        bounds_l,
        bounds_r,
        out_bins,
        bvh_nodes );

    cudaThreadSynchronize();
}

/// build a set of head pointers starting from the segment flags & keys
__global__ void build_head_pointers_kernel(
    const uint32    n_objects,
    const uint32*   segment_keys,
    uint32*         segment_heads)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_objects)
            return;

        // get the segment this object belongs to
        const uint32 key      = segment_keys[ index ]-1;
        const uint32 prev_key = index ? segment_keys[ index-1 ]-1 : uint32(-1);

        // check whether this is a head pointer
        if (key == prev_key)
            continue;

        // store the new head
        segment_heads[ key ] = index;
    }
}

/// build a set of head pointers starting from the segment flags & keys
void build_head_pointers(
    const uint32    n_objects,
    const uint32*   segment_keys,
    uint32*         segment_heads)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(build_head_pointers_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    build_head_pointers_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        segment_keys,
        segment_heads );

    cudaThreadSynchronize();
}


///
/// CUDA kernel: setup the leaf array
///
__global__ void setup_leaves_kernel(
    const uint32    n_objects,
    const uint32*   leaf_ids,
    uint2*          leaves)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 id = threadIdx.x + base_idx;

        if (id >= n_objects)
            continue;

        const uint32      leaf_id = leaf_ids[ id ];
        const uint32 prev_leaf_id = id == 0           ? uint32(-1) : leaf_ids[ id-1 ];
        const uint32 next_leaf_id = id == n_objects-1 ? uint32(-1) : leaf_ids[ id+1 ];

        if (prev_leaf_id != leaf_id)
            leaves[ leaf_id ].x = id;

        if (next_leaf_id != leaf_id)
            leaves[ leaf_id ].y = id+1;
    }
}
///
/// Setup the leaf array
///
void setup_leaves(
    const uint32    n_objects,
    const uint32*   leaf_ids,
    uint2*          leaves)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(setup_leaves_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    setup_leaves_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        leaf_ids,
        leaves );

    cudaThreadSynchronize();
}

/// CUDA kernel:
/// find the best splits - first part
__global__ void init_splits_kernel(
    const uint32    n_segments,
    float*          split_costs,
    uint32*         split_index)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_segments;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_segments)
            return;

        split_costs[index] = 1.0e9f;
        split_index[index] = uint32(-1);
    }
}
/// CUDA kernel:
/// find the best splits - first part
__global__ void find_best_splits_kernel(
    const uint32    n_objects,
    const uint32*   segment_keys,
    Cost_functor    cost_functor,
    float*          split_costs,
    uint32*         split_index)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    const uint32 BLOCK_SIZE = 256;
    const uint32 WARP_COUNT = BLOCK_SIZE >> CUDA_config::log_WARP_SIZE;

    const uint32 warp_tid = threadIdx.x & (CUDA_config::WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> CUDA_config::log_WARP_SIZE;

    __shared__ volatile float sm_tile[BLOCK_SIZE];
    __shared__ volatile uint32 sm_begin_key[ WARP_COUNT ], sm_end_key[ WARP_COUNT ];

    volatile float*  tile      = sm_tile + warp_id * CUDA_config::WARP_SIZE;
    volatile uint32& begin_key = sm_begin_key[ warp_id ];
    volatile uint32& end_key   = sm_end_key[ warp_id ];

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;

        // load this tile and compute its min and max keys
        bool valid = index < n_objects;

        // get the segment this object belongs to
        const uint32 key = valid ? segment_keys[ index ]-1 : segment_keys[ n_objects-1 ]-1;

        if (warp_tid == 0)
            begin_key = key;
        if (warp_tid == CUDA_config::WARP_SIZE-1)
            end_key = key+1;

        // load the values currently bound to this tile of keys in smem
        if (warp_tid < end_key - begin_key)
            tile[ warp_tid ] = split_costs[ warp_tid + begin_key ];

        // perform a min-reduction in the smem tile
        if (valid)
        {
            const thrust::tuple<float,uint32> split = cost_functor( index );

            const float old_cost = tile[ key - begin_key ];
            const float new_cost = thrust::get<0>( split );
            if (new_cost < old_cost)
                atomicMin( (int32*)&tile[ key - begin_key ], binary_cast<int32>( new_cost ) );
        }

        // merge this tile of per-key values back in gmem
        if (warp_tid < end_key - begin_key)
        {
            const float old_cost = split_costs[ warp_tid + begin_key ];
            const float new_cost = tile[ warp_tid ];
            if (new_cost < old_cost)
                atomicMin( (int32*)&split_costs[ warp_tid + begin_key ], binary_cast<int32>( new_cost ) );
        }
    }
}
/// CUDA kernel:
/// find the best splits - second part
__global__ void assign_best_splits_kernel(
    const uint32    n_objects,
    const uint32*   segment_keys,
    Cost_functor    cost_functor,
    float*          split_costs,
    uint32*         split_index)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_objects)
            return;

        // get the segment this object belongs to
        const uint32 key = segment_keys[ index ]-1;

        const thrust::tuple<float,uint32> split = cost_functor( index );

        const float old_cost = split_costs[ key ];
        const float new_cost = thrust::get<0>( split );
        //if (new_cost == old_cost)
        //    split_index[ key ] = thrust::get<1>( split );
        const uint32 old_index = split_index[ key ];
        const uint32 new_index = thrust::get<1>( split );
        if (new_cost == old_cost && new_index < old_index)
            atomicMin( &split_index[ key ], new_index );
    }
}

/// assign objects to their new nodes
void find_best_splits(
    const uint32    n_objects,
    const uint32    n_segments,
    const uint32*   segment_keys,
    Cost_functor    cost_functor,
    float*          split_costs,
    uint32*         split_index)
{
    const uint32 BLOCK_SIZE = 256;
    const size_t max_blocks = nih::min(
        thrust::detail::backend::cuda::arch::max_active_blocks(find_best_splits_kernel, BLOCK_SIZE, 0),
        thrust::detail::backend::cuda::arch::max_active_blocks(assign_best_splits_kernel, BLOCK_SIZE, 0) );
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    init_splits_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_segments,
        split_costs,
        split_index );

    cudaThreadSynchronize();

    find_best_splits_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        segment_keys,
        cost_functor,
        split_costs,
        split_index );

    cudaThreadSynchronize();

    assign_best_splits_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        segment_keys,
        cost_functor,
        split_costs,
        split_index );

    cudaThreadSynchronize();
}

template <typename Integer>
void sort(
    const uint32                                n_objects,
    thrust::device_vector<uint32>::iterator     segment_ids,
    thrust::device_vector<uint32>::iterator     segment_keys,
    thrust::device_vector<uint2>::iterator      in_bounds,
    thrust::device_vector<uint2>::iterator      bounds,
    thrust::device_vector<uint2>::iterator      bounds_tmp,
    thrust::device_vector<uint32>::iterator     order,
    uint32&                                     n_active_objects,
    uint32&                                     n_segments)
{
    thrust::device_ptr<Integer> key_centroids( (Integer*)thrust::raw_pointer_cast( &*bounds ) );
    thrust::device_ptr<Integer> key_centroids_tmp( (Integer*)thrust::raw_pointer_cast( &*bounds_tmp ) );
    thrust::device_ptr<uint32>  order_tmp( (uint32*)thrust::raw_pointer_cast( &*segment_keys ) );

    thrust::transform(
        thrust::make_zip_iterator( thrust::make_tuple( segment_ids, in_bounds ) ),
        thrust::make_zip_iterator( thrust::make_tuple( segment_ids, in_bounds ) ) + n_objects,
        key_centroids,
        sah::Key_centroid_packer<Integer>() );

    thrust::copy(
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(0u) + n_objects,
        order );

    radix_sort(
        n_objects,
        thrust::raw_pointer_cast( &*key_centroids ),
        thrust::raw_pointer_cast( &*order ),
        thrust::raw_pointer_cast( &*key_centroids_tmp ),
        thrust::raw_pointer_cast( &*order_tmp ) );

    n_active_objects = thrust::find_if(
        key_centroids,
        key_centroids + n_active_objects,
        sah::invalid_key<Integer>() ) - key_centroids;

    thrust::transform(
        key_centroids,
        key_centroids + n_active_objects,
        segment_keys,
        sah::extract_key<Integer>() );

    n_segments = segment_keys[ n_active_objects-1 ];

    // gather the compressed bounds in the proper order
    thrust::gather(
        order,
        order + n_active_objects,
        in_bounds,
        bounds );
}

} // namespace sah

void Sah_builder::sort(
    const uint32                            n_objects,
    thrust::device_vector<uint32>::iterator segment_ids,
    thrust::device_vector<uint32>::iterator segment_keys,
    thrust::device_vector<uint2>::iterator  in_bounds,
    thrust::device_vector<uint2>::iterator  bounds,
    thrust::device_vector<uint2>::iterator  bounds_tmp,
    thrust::device_vector<uint32>::iterator order,
    uint32&                                 n_active_objects,
    uint32&                                 n_segments)
{
    // check whether we can fit the segment keys in 32 - 10 bits (where
    // 10 is the number of bits used for the centroids)
    if (n_segments < (1u << 6) - 1u)
    {
        sah::sort<uint16>(
            n_objects,
            segment_ids,
            segment_keys,
            in_bounds,
            bounds,
            bounds_tmp,
            order,
            n_active_objects,
            n_segments );
    }
    else if (n_segments < (1u << 22) - 1u)
    {
        sah::sort<uint32>(
            n_objects,
            segment_ids,
            segment_keys,
            in_bounds,
            bounds,
            bounds_tmp,
            order,
            n_active_objects,
            n_segments );
    }
    else
    {
        sah::sort<uint64>(
            n_objects,
            segment_ids,
            segment_keys,
            in_bounds,
            bounds,
            bounds_tmp,
            order,
            n_active_objects,
            n_segments );
    }
}

void Sah_builder::eval_split_costs(
    const uint32                            n_active_objects,
    const uint32                            n_segments,
    thrust::device_vector<uint32>::iterator segment_keys,
    thrust::device_vector<uint32>::iterator segment_heads,
    thrust::device_vector<uint2>::iterator  bounds,
    thrust::device_vector<uint2>::iterator  bounds_l,
    thrust::device_vector<uint2>::iterator  bounds_r,
    thrust::device_vector<float>::iterator  split_costs,
    thrust::device_vector<uint32>::iterator split_index)
{
#if 1
    // compute segment flags
    segment_heads[0] = 1u;

    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                segment_keys,
                segment_keys+1 ) ),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                segment_keys,
                segment_keys+1 ) ) + n_active_objects-1,
        segment_heads + 1,
        sah::different_keys() );

    sah::inclusive_segmented_scan(
        thrust::raw_pointer_cast( &*bounds_l ),
        thrust::raw_pointer_cast( &*bounds ),
        thrust::raw_pointer_cast( &*segment_heads ),
        int(n_active_objects) );

    // compute reverse segment flags
    thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_reverse_iterator( segment_keys + n_active_objects ),
                thrust::make_reverse_iterator( segment_keys + n_active_objects ) + 1 ) ),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_reverse_iterator( segment_keys + n_active_objects ),
                thrust::make_reverse_iterator( segment_keys + n_active_objects ) + 1 ) ) + n_active_objects-1,
        segment_heads + 1,
        sah::different_keys() );

    // reverse the bounds sequence
    thrust::copy(
        thrust::make_reverse_iterator( bounds + n_active_objects ),
        thrust::make_reverse_iterator( bounds + n_active_objects ) + n_active_objects,
        bounds_r );

    sah::exclusive_segmented_scan(
        thrust::raw_pointer_cast( &*bounds ),
        thrust::raw_pointer_cast( &*bounds_r ),
        thrust::raw_pointer_cast( &*segment_heads ),
        int(n_active_objects) );

    // reverse the bounds_r sequence
    thrust::copy(
        thrust::make_reverse_iterator( bounds + n_active_objects ),
        thrust::make_reverse_iterator( bounds + n_active_objects ) + n_active_objects,
        bounds_r );
#else
    // scan the bounds from the left
    thrust::inclusive_scan_by_key(
        segment_keys,
        segment_keys + n_active_objects,
        bounds,
        bounds_l,
        thrust::equal_to<uint32>(),
        sah::merge() );

    // scan the bounds from the right
    thrust::exclusive_scan_by_key(
        thrust::make_reverse_iterator( segment_keys + n_active_objects ),
        thrust::make_reverse_iterator( segment_keys + n_active_objects ) + n_active_objects,
        thrust::make_reverse_iterator( bounds + n_active_objects ),
        thrust::make_reverse_iterator( bounds_r + n_active_objects ),
        make_uint2( 0xFFFFFFFFu, 0u ),
        thrust::equal_to<uint32>(),
        sah::merge() );
#endif
    // build head pointers
    {
        sah::build_head_pointers(
            n_active_objects,
            thrust::raw_pointer_cast( &*segment_keys ),
            thrust::raw_pointer_cast( &*segment_heads ) );

        segment_heads[ n_segments ] = n_active_objects;
    }
#if 1
    sah::find_best_splits(
        n_active_objects,
        n_segments,
        thrust::raw_pointer_cast( &*segment_keys ),
        sah::Cost_functor(
            thrust::raw_pointer_cast( &*segment_keys ),
            thrust::raw_pointer_cast( &*segment_heads ),
            thrust::raw_pointer_cast( &*bounds_l ),
            thrust::raw_pointer_cast( &*bounds_r ) ),
        thrust::raw_pointer_cast( &*split_costs ),
        thrust::raw_pointer_cast( &*split_index )  );
#else
    // find the best split plane for each node
    thrust::reduce_by_key(
        segment_keys,
        segment_keys + n_active_objects,
        thrust::make_transform_iterator(
            thrust::make_counting_iterator(0u),
            sah::Cost_functor(
                thrust::raw_pointer_cast( &segment_keys.front() ),
                thrust::raw_pointer_cast( &segment_heads.front() ),
                thrust::raw_pointer_cast( &*bounds_l ),
                thrust::raw_pointer_cast( &*bounds_r ) ) ),
            tmp,
        thrust::make_zip_iterator( thrust::make_tuple( split_costs, split_index ) ),
        thrust::equal_to<uint32>(),
        sah::Best_split_functor() );
#endif
}

} // namespace cuda
} // namespace nih
