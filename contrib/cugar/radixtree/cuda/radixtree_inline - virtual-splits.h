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

#include <cugar/basic/cuda/arch.h>
#include <cugar/basic/cuda/pointers.h>
#include <cugar/basic/functors.h>
#include <cugar/basic/algorithms.h>
#include <cugar/basic/cuda/scan.h>
#include <cugar/basic/utils.h>

namespace cugar {
namespace cuda {
namespace bintree {

typedef Radixtree_context::Split_task Split_task;

// find the most significant bit smaller than start by which code0 and code1 differ
template <typename Integer>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE int32 find_leading_bit_difference(
    const  int32  start_level,
    const Integer code0,
    const Integer code1)
{
    int32 level = start_level;

    while (level >= 0)
    {
        const Integer mask = Integer(1u) << level;

        if ((code0 & mask) !=
            (code1 & mask))
            break;

        --level;
    }
    return level;
}

#define RADIX_TREE_USE_VOLATILE             1
#define RADIX_TREE_USE_FENCE                1
#define RADIX_TREE_USE_ATOMIC_RELEASE       1

#if defined(RADIX_TREE_USE_VOLATILE)
  #define RADIX_TREE_UNCACHED_LOAD(x)    load<LOAD_VOLATILE>(x)
  #define RADIX_TREE_UNCACHED_STORE(x,v) store<STORE_VOLATILE>(x,v)
#else
  #define RADIX_TREE_UNCACHED_LOAD(x)    load<LOAD_CG>(x)
  #define RADIX_TREE_UNCACHED_STORE(x,v) store<STORE_CG>(x,v)
#endif

#if defined(RADIX_TREE_USE_FENCE)
  #define RADIX_TREE_RELEASE_FENCE()     __threadfence()
#else
  #define RADIX_TREE_RELEASE_FENCE()
#endif

#if defined(RADIX_TREE_USE_ATOMIC_RELEASE)
  #define RADIX_TREE_RELEASE(x,v)        atomicExch(x,v)
#else
  #define RADIX_TREE_RELEASE(x,v)        RADIX_TREE_UNCACHED_STORE(x,v)
#endif

// do a single kd-split for all nodes in the input task queue, and generate
// a corresponding list of output tasks
template <uint32 BLOCK_SIZE, typename Tree, typename Integer>
__global__ void split_kernel(
    const uint32        grid_size,
    Tree                tree,
    const uint32        max_leaf_size,
    const uint32        n_nodes,
    const uint32        n_codes,
    const Integer*      codes,
    int32*              flags,
    Split_task*         tasks,
    uint32*             skip_nodes,
    uint32*             out_node_count,
    uint32*             out_leaf_count,
    uint32*             work_counter)
{
    const uint32 LOG_WARP_SIZE = 5;
    const uint32 WARP_SIZE = 1u << LOG_WARP_SIZE;

    volatile __shared__ uint32 warp_offset[ BLOCK_SIZE >> LOG_WARP_SIZE ];

    const uint32 warp_tid = threadIdx.x & (WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> LOG_WARP_SIZE;

    uint32 node         = 0;
    uint32 begin        = 0;
    uint32 end          = 0;
    uint32 level        = uint32(-1);
    uint32 parent       = uint32(-1);
    uint32 skip_node    = uint32(-1);
    uint32 split_index  = 0;
     int32 node_flag    = -1;
     bool  proper_split = false;
     bool  terminated   = false;

    // keep the entire warp looping until there's some work to do
    while (__any(!terminated))
    {
        // fetch new tasks for inactive lanes which are done processing their node (i.e. node_flag != 0)
        const uint32 new_node = cuda::alloc<1>( node_flag != 0, work_counter, warp_tid, warp_offset + warp_id );

        if (node_flag != 0) // check if we are done processing the current node
        {
            // check whether we are done processing all nodes
            if (new_node >= n_nodes)
                terminated = true;

            // reset the node
            node         = new_node;
            begin        = 0;
            end          = 0;
            level        = uint32(-1);
            parent       = uint32(-1);
            skip_node    = uint32(-1);
            split_index  = 0;
            node_flag    = 0;
            proper_split = false;
        }

        //node_flag = terminated ? 0 : *(volatile int32*)(flags + node);
        node_flag = terminated ? 0 : RADIX_TREE_UNCACHED_LOAD(flags + node);

        if (node_flag != 0)
        {
            // fetch this node's description
            const Split_task in_task = RADIX_TREE_UNCACHED_LOAD( (uint4*)tasks + node );

            parent      = in_task.m_parent;
            begin       = in_task.m_begin;
            end         = in_task.m_end;
            level       = in_task.m_level;
            split_index = (begin + end)/2;

            skip_node = RADIX_TREE_UNCACHED_LOAD( skip_nodes + node );

            // check whether this is a valid node, needing a proper split
            if (node_flag == 1)
            {
                // check whether the input node really needs to be split
                if (end - begin > max_leaf_size)
                {
                    if (level != uint32(-1))
                    {
                        // adjust the splitting level so as to make sure the split will produce either 2 or 0 children
                        level = find_leading_bit_difference(
                            level,
                            codes[begin],
                            codes[end-1] );
                    }

                    // check again if there is any chance to make a split, after the level has been adjusted
                    if (level != uint32(-1))
                    {
                        // find the "partitioning pivot" using a binary search
                        split_index = find_pivot(
                            codes + begin,
                            end - begin,
                            mask_and<Integer>( Integer(1u) << level ) ) - codes;

                        // this shouldn't be needed, but... force a good split
                        if (split_index == begin || split_index == end)
                            split_index = (begin + end)/2;
                    }

                    // mark this as a proper split
                    proper_split = true;
                }
            }
        }

        #define RADIX_TREE_WRITE_NODE( OUTPUT_INDEX, PARENT, BEGIN, END, LEVEL, SKIP, RELEASE_VALUE )                   \
        do {                                                                                                            \
            RADIX_TREE_UNCACHED_STORE( (uint4*)tasks + OUTPUT_INDEX, make_uint4( PARENT, BEGIN, END, LEVEL ) );         \
            RADIX_TREE_UNCACHED_STORE( skip_nodes + OUTPUT_INDEX, SKIP );                                               \
            RADIX_TREE_RELEASE_FENCE();                                                                                 \
            RADIX_TREE_RELEASE( flags + OUTPUT_INDEX, RELEASE_VALUE );                                                  \
        } while (0)

        // split the node if it contains more than one code
        const uint32 child_count = (node_flag != 0) && (end - begin > 1u) ? 2u : 0u;

        // alloc the actual children
        //const uint32 node_offset = cuda::alloc<2>( child_count, out_node_count, warp_tid, warp_offset + warp_id );
        const uint32 node_offset = atomicAdd( out_node_count, child_count );

        // write the them out
        if (child_count == 2)
        {
            RADIX_TREE_WRITE_NODE( node_offset+0, node, begin, split_index, level-1, node_offset+1, proper_split ? 1 : -1 ); // mark nodes produced by "virtual" splits with a -1
            RADIX_TREE_WRITE_NODE( node_offset+1, node, split_index, end,   level-1, skip_node,     proper_split ? 1 : -1 ); // mark nodes produced by "virtual" splits with a -1
        }

        const bool generate_leaf = (node_flag == 1) && (proper_split == false);

        // count how many leaves we need to generate
        //const uint32 leaf_index = cuda::alloc<1>( generate_leaf, out_leaf_count, warp_tid, warp_offset + warp_id );
        const uint32 leaf_index = atomicAdd( out_leaf_count, generate_leaf ? 1u : 0u );

        // write out the current node if it's not virtual
        if (node_flag == 1)
        {
            tree.write_node(
                node,
                parent,
                generate_leaf ? false : true,
                generate_leaf ? false : true,
                generate_leaf ? leaf_index : node_offset,
                skip_node,
                level,
                begin,
                end,
                generate_leaf ? uint32(-1) : split_index );

            // make a leaf if necessary
            if (generate_leaf)
                tree.write_leaf( leaf_index, node, begin, end );
        }
    }
}

// do a single kd-split for all nodes in the input task queue, and generate
// a corresponding list of output tasks
template <typename Tree, typename Integer>
void split(
    Tree                tree,
    const uint32        max_leaf_size,
    const uint32        n_nodes,
    const uint32        n_codes,
    const Integer*      codes,
    int32*              flags,
    Split_task*         tasks,
    uint32*             skip_nodes,
    uint32*             out_node_count,
    uint32*             out_leaf_count,
    uint32*             work_counter)
{
    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = cuda::max_active_blocks(split_kernel<BLOCK_SIZE,Tree,Integer>, BLOCK_SIZE, 0);
    const size_t n_blocks   = cugar::min( max_blocks, size_t(n_nodes + BLOCK_SIZE-1) / BLOCK_SIZE );
    const size_t grid_size  = n_blocks * BLOCK_SIZE;

    split_kernel<BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        grid_size,
        tree,
        max_leaf_size,
        n_nodes,
        n_codes,
        codes,
        flags,
        tasks,
        skip_nodes,
        out_node_count,
        out_leaf_count,
        work_counter );

    //cudaDeviceSynchronize();
}

} // namespace bintree

template <typename Tree, typename Integer>
void generate_radix_tree(
    Radixtree_context& context,
    const uint32    n_codes,
    const Integer*  codes,
    const uint32    bits,
    const uint32    max_leaf_size,
    const bool      keep_singletons,
    Tree&           tree)
{
    const uint32 max_nodes = cugar::max( n_codes * 2u - 1u, 1u );

    tree.reserve_nodes( max_nodes );
    tree.reserve_leaves( cugar::max( n_codes, 1u ) );

    // reserve storage for internal queues
    need_space( context.m_task_queues, max_nodes );
    need_space( context.m_skip_nodes,  max_nodes );

    context.m_counters.resize( 3 );
    context.m_counters[0] = 1; // nodes counter
    context.m_counters[1] = 0; // leaf counter
    context.m_counters[2] = 0; // work counter

    context.m_task_queues[0] = Radixtree_context::Split_task( uint32(-1), 0, n_codes, bits-1 );
    context.m_skip_nodes[0]  = uint32(-1);

    caching_device_vector<int32> flags( max_nodes, 0u );

    flags[0] = 1u; // mark the root node as ready to be split

    // build the radix tree in a single pass
    bintree::split(
        tree.get_context(),
        max_leaf_size,
        max_nodes,
        n_codes,
        codes,
        raw_pointer( flags ),
        raw_pointer( context.m_task_queues ),
        raw_pointer( context.m_skip_nodes ),
        raw_pointer( context.m_counters ),
        raw_pointer( context.m_counters ) + 1,
        raw_pointer( context.m_counters ) + 2 );

    context.m_nodes  = context.m_counters[0];
    context.m_leaves = context.m_counters[1];
}

template <typename Tree_writer, typename Integer>
void generate_radix_tree(
    const uint32            n_codes,
    const Integer*          codes,
    const uint32            bits,
    const uint32            max_leaf_size,
    const bool              keep_singletons,
    Tree_writer&            tree)
{
    Radixtree_context context;

    generate_radix_tree( context, n_codes, codes, bits, max_leaf_size, keep_singletons, tree );
}

} // namespace cuda
} // namespace cugar
