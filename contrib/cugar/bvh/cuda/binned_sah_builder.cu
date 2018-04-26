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

#include <nih/bvh/cuda/binned_sah_builder.h>
#include <nih/basic/cuda_config.h>

#define ACCESS_BINS(a,id,axis,index,stride) a[id + (axis*BINS + index)*stride]

namespace nih {
namespace cuda {
namespace binned_sah {

FORCE_INLINE NIH_DEVICE Bin operator+ (const Bin bin1, const Bin bin2)
{
    Bin r;

    r.bmin = make_float4(
        fminf( bin1.bmin.x, bin2.bmin.x ),
        fminf( bin1.bmin.y, bin2.bmin.y ),
        fminf( bin1.bmin.z, bin2.bmin.z ),
        __int_as_float(
            __float_as_int( bin1.bmin.w ) +
            __float_as_int( bin2.bmin.w )) );

    r.bmax = make_float4(
        fmaxf( bin1.bmax.x, bin2.bmax.x ),
        fmaxf( bin1.bmax.y, bin2.bmax.y ),
        fmaxf( bin1.bmax.z, bin2.bmax.z ),
        0.0f );

    return r;
}
FORCE_INLINE NIH_DEVICE Bin merge(const float3 bmin, const float3 bmax, const int32 size)
{
    Bin r;

    r.bmin = make_float4(
        bmin.x,
        bmin.y,
        bmin.z,
        __int_as_float(size) );

    r.bmax = make_float4(
        bmax.x,
        bmax.y,
        bmax.z,
        0.0f );

    return r;
}
FORCE_INLINE NIH_DEVICE Bin bin(const uint32 BINS, const Bins bins, const int id, const int axis, const int index, const int stride)
{
    return merge(
        ACCESS_BINS(bins.bmin,id,axis,index,stride),
        ACCESS_BINS(bins.bmax,id,axis,index,stride),
        ACCESS_BINS(bins.size,id,axis,index,stride) );
}


__global__ void init_bins_kernel(const uint32 BINS, const uint32 n_nodes, Bins bins)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_nodes*BINS*3;
                base_idx += grid_size)
    {
        const uint32 id = threadIdx.x + base_idx;
        if (id >= n_nodes*BINS*3)
            continue;

        const float HUGE = 1.0e8f;

        bins.bmin[id] = make_float3(  HUGE,  HUGE,  HUGE );
        bins.bmax[id] = make_float3( -HUGE, -HUGE, -HUGE );
        bins.size[id] = 0;
    }
}

void init_bins(const uint32 BINS, const uint32 n_nodes, Bins bins)
{
    const uint32 BLOCK_SIZE = SAH_SINGLE_WARP ? 32 : 256;
    const size_t max_blocks = SAH_SINGLE_WARP ? 1 : thrust::detail::backend::cuda::arch::max_active_blocks(init_bins_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_nodes*BINS*3 + BLOCK_SIZE-1) / BLOCK_SIZE );

    init_bins_kernel<<<n_blocks,BLOCK_SIZE>>>( BINS, n_nodes, bins );

    cudaThreadSynchronize();
}

/// evaluate the area of a bin
FORCE_INLINE NIH_HOST_DEVICE float area(const Bin bin)
{
    const float3 edge = make_float3(
        bin.bmax.x - bin.bmin.x,
        bin.bmax.y - bin.bmin.y,
        bin.bmax.z - bin.bmin.z );

    return edge.x*edge.y + edge.x*edge.z + edge.y*edge.z;
}

/// evaluate the SAH cost of a given division in 2 bins
FORCE_INLINE NIH_DEVICE float sah_cost(
    const Bin bin1,
    const Bin bin2)
{
    return area( bin1 ) * __float_as_int(bin1.bmin.w) +
           area( bin2 ) * __float_as_int(bin2.bmin.w);
}

///
/// CUDA kernel: find the best SAH split plane for each node in the input task queue,
/// and generate child tasks.
///
__global__ void sah_split_kernel(
    const uint32        BINS,
    Bins                bins,
    Queue               qin,
    const int           input_node_offset,
    Queue               qout,
    uint32*             n_output,
    int                 output_node_offset,
    Bvh_node*           nodes,
    uint32*             n_leaves,
    const uint32        max_leaf_size,
    const float         max_cost)
{
    typedef Bin Bin;

    const uint32 grid_size = gridDim.x * blockDim.x;

    __shared__ uint32 warp_broadcast[32];

    const uint32 warp_tid = threadIdx.x & (CUDA_config::WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> CUDA_config::log_WARP_SIZE;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < qin.size;
                base_idx += grid_size)
    {
        const uint32 id = threadIdx.x + base_idx;

        if (id >= qin.size)
            continue;

        int best_split = -1;
        Bin bestL;
        Bin bestR;

        const Bin    bbox      = qin.bins[id];
        const int    node_size = __float_as_int(bbox.bmin.w);
        const uint32 skip_node = binary_cast<uint32>(bbox.bmax.w);
        const int    node_id   = input_node_offset + id;

        // mark this node tentatively as a leaf node
        bool split = false;

        // and try to split it if necessary
	    if (node_size > max_leaf_size)
	    {
            float best_cost = max_cost * area( bbox ) * node_size;

            Bin bboxesR[ SAH_MAX_BINS ];

		    // perform a serial SAH evaluation (fast for small arrays)
		    for (int axis = 0; axis < 3; axis++) 
		    {
                // right scan
			    bboxesR[BINS - 1] = bin( BINS, bins, id, axis, BINS-1, qin.size );
                for (int i = BINS - 2; i >= 0; i--)
				    bboxesR[i] = bboxesR[i + 1] + bin( BINS, bins, id, axis, i, qin.size );

			    // left scan
			    Bin bboxesL = bin( BINS, bins, id, axis, 0, qin.size );

			    for (int i = 0; i < BINS - 1; i++)
			    {
                    // skip invalid splits
                    if (__float_as_int(bboxesL.bmin.w) != 0 &&
                        __float_as_int(bboxesR[i+1].bmin.w) != 0)
                    {
                        const float cost = sah_cost( bboxesL, bboxesR[i+1] );
				        if(cost < best_cost)
                        {
					        best_cost = cost;
					        best_split = axis * BINS + i;
					        bestL = bboxesL;
					        bestR = bboxesR[i+1];
					        split = true;
				        }
                    }

                    const Bin next_bin = bin( BINS, bins, id, axis, i+1, qin.size );

                    bboxesL = bboxesL + next_bin;
			    }
            }

            // TODO: check whether the split failed and mark for
            // random-order middle split.
        }

	    if(split)
	    {
            // allocate 2 child tasks and their corresponding nodes
        	const int new_offset = alloc<2>( split, n_output, warp_tid, warp_broadcast + warp_id );
		    const int new_split  = new_offset;
		    const int new_node   = output_node_offset + new_offset;

            qin.splits[ id ] = Split( new_split, best_split );

            // pack skip nodes in bmax.w
            bestL.bmax.w = binary_cast<float>(new_node+1);
            bestR.bmax.w = binary_cast<float>(skip_node);

            qout.bins[ new_split+0 ] = bestL;
            qout.bins[ new_split+1 ] = bestR;

            // set this node as a split
            nodes[ node_id ] = Bvh_node( Bvh_node::kInternal, new_node, skip_node );

            #ifdef SAH_CHECKS
            if (__float_as_int(bestL.bmin.w) + __float_as_int(bestR.bmin.w) != __float_as_int(bbox.bmin.w))
            {
                printf("split bbox :\n" \
                    "  [%f, %f, %f], [%f, %f, %f] - %d\n" \
                    "  [%f, %f, %f], [%f, %f, %f] - %d\n" \
                    "  [%f, %f, %f], [%f, %f, %f] - %d\n  %d\n", \
                    bbox.bmin.x,bbox.bmin.y,bbox.bmin.z,
                    bbox.bmax.x,bbox.bmax.y,bbox.bmax.z,
                    __float_as_int(bbox.bmin.w),
                    bestL.bmin.x,bestL.bmin.y,bestL.bmin.z,
                    bestL.bmax.x,bestL.bmax.y,bestL.bmax.z,
                    __float_as_int(bestL.bmin.w),
                    bestR.bmin.x,bestR.bmin.y,bestR.bmin.z,
                    bestR.bmax.x,bestR.bmax.y,bestR.bmax.z,
                    __float_as_int(bestR.bmin.w),
                    new_split );
            }
            #endif
        }
	    else
        {
            // allocate a leaf
            const int leaf_id = alloc<1>( true, n_leaves, warp_tid, warp_broadcast + warp_id );

            qin.splits[ id ] = Split( -1, leaf_id );

            // set this node as a leaf
            nodes[ node_id ] = Bvh_node( Bvh_node::kLeaf, leaf_id, skip_node );
        }
    }
}

///
/// Find the best SAH split plane for each node in the input task queue, and generate child tasks
///
void sah_split(
    const uint32    BINS,
    Bins            bins,
    Queue           qin,
    const int       input_node_offset,
    Queue           qout,
    uint32*         n_output,
    int             output_node_offset,
    Bvh_node*       nodes,
    uint32*         n_leaves,
    const uint32    max_leaf_size,
    const float     max_cost)
{
    const uint32 BLOCK_SIZE = SAH_SINGLE_WARP ? 32 : 256;
    const size_t max_blocks = SAH_SINGLE_WARP ? 1 : thrust::detail::backend::cuda::arch::max_active_blocks(sah_split_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (qin.size + BLOCK_SIZE-1) / BLOCK_SIZE );

    sah_split_kernel<<<n_blocks,BLOCK_SIZE>>> (
        BINS,
        bins,
        qin,
        input_node_offset,
        qout,
        n_output,
        output_node_offset,
        nodes,
        n_leaves,
        max_leaf_size,
        max_cost );

    cudaThreadSynchronize();
}

///
/// CUDA kernel: assign the objects to their new task, or to a node if there was no split
///
__global__ void distribute_objects_kernel(
    const uint32    BINS,
    Objects         objects,
    const int       n_objects,
    Queue           queue,
    const int       input_node_offset,
    Bins            bins)
{
    typedef Split Split;

    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 idx = threadIdx.x + base_idx;

        if (idx >= n_objects)
            return;

        const uint32 id = idx;

        // check if the object has already been assigned to a node
	    const int node_id = objects.node_ids[id];
    	if (node_id > -1)
            continue;

	    const int4  bin_id    = objects.bin_ids[idx];
	    const int   split_id  = objects.split_ids[id];
	    const Split split     = queue.splits[split_id];

        const int32 new_split_id = split.task_id;

        // if the node has not been split, we have to assign its objects
	    if(new_split_id == -1)
        {
            const int32 leaf_id = split.best_plane;
            objects.node_ids[id] = leaf_id;
		    continue;
	    }

        // assign the object to its new task
	    const int32 best_split   = split.best_plane;
	    const int32 selected_bin = best_split < BINS ? bin_id.x : (best_split < 2 * BINS ? bin_id.y : bin_id.z); // select the axis&bin of the best split
        objects.split_ids[id] = selected_bin <= (best_split & (BINS-1)) ? new_split_id : new_split_id + 1;
    }
}
///
/// Assign the objects to their new task, or to a node if there was no split
///
void distribute_objects(
    const uint32    BINS,
    Objects         objects,
    const int       n_objects,
    Queue           queue,
    const int       input_node_offset,
    Bins            bins)
{
    const uint32 BLOCK_SIZE = SAH_SINGLE_WARP ? 32 : 256;
    const size_t max_blocks = SAH_SINGLE_WARP ? 1  : thrust::detail::backend::cuda::arch::max_active_blocks(distribute_objects_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    distribute_objects_kernel<<<n_blocks,BLOCK_SIZE>>> (
        BINS,
        objects,
        n_objects,
        queue,
        input_node_offset,
        bins );

    cudaThreadSynchronize();
}

///
/// CUDA kernel: setup the leaf array
///
__global__ void setup_leaves_kernel(
    const int       n_objects,
    const int32*    leaf_ids,
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

        const int32      leaf_id = leaf_ids[ id ];
        const int32 prev_leaf_id = id == 0           ? -1 : leaf_ids[ id-1 ];
        const int32 next_leaf_id = id == n_objects-1 ? -1 : leaf_ids[ id+1 ];

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
    const int       n_objects,
    const int32*    leaf_ids,
    uint2*          leaves)
{
    const uint32 BLOCK_SIZE = SAH_SINGLE_WARP ? 32 : 256;
    const size_t max_blocks = SAH_SINGLE_WARP ? 1  : thrust::detail::backend::cuda::arch::max_active_blocks(setup_leaves_kernel, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    setup_leaves_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_objects,
        leaf_ids,
        leaves );

    cudaThreadSynchronize();
}

} // namespace binned_sah
} // namespace cuda
} // namespace nih