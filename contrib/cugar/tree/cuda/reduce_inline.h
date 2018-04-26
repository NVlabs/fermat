/*
 * Copyright (c) 2010-2018, NVIDIA Corporation
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

#include <cugar/basic/types.h>
#include <cugar/basic/cuda/arch.h>
#include <thrust/version.h>

namespace cugar {
namespace cuda {

namespace treereduce {

// reduce leaf values
template <uint32 BLOCK_SIZE, typename Tree, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
__global__ void reduce_from_leaves_kernel(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const Value_type        def_value,
    uint32*                 visited)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    const uint32 n_leaves = tree.get_leaf_count();

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
        base_idx < n_leaves;
        base_idx += grid_size)
    {
        const uint32 leaf_id = threadIdx.x + base_idx;

        if (leaf_id < n_leaves)
        {
                  uint32 node_id = tree.get_leaf_node( leaf_id );
            const uint2  leaf    = tree.get_leaf_range( node_id );
            const uint32 begin   = leaf.x;
            const uint32 end     = leaf.y;

            Value_type value = def_value;
            for (uint32 i = begin; i < end; ++i)
                value = op( value, in_values[i] );

            out_values[ node_id ] = value;

            __threadfence();

            // go up the tree
            while (1)
            {
                node_id = tree.get_parent( node_id );
                if (node_id == uint32(-1))
                    break;

                const uint32 child_count = tree.get_child_count( node_id );

                // check whether this is the last child of the current node to reach this point
                if (atomicAdd( &visited[node_id], 1u ) != child_count-1)
                    break;

                Value_type value = out_values[tree.get_child( node_id, 0 )];
                for (uint32 i = 1; i < child_count; ++i)
                    value = op( value, out_values[tree.get_child( node_id, i )] );

                out_values[ node_id ] = value; // this write must be uncached to work

                __threadfence(); // make sure this write is visible to all threads
            }
        }
    }
}

// reduce leaf values
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
void reduce_with_leaf_pointers(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const Value_type        def_value)
{
    const uint32 n_leaves = tree.get_leaf_count();
    const uint32 n_nodes  = tree.get_node_count();

    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = cuda::max_active_blocks(reduce_from_leaves_kernel<BLOCK_SIZE, Tree, Input_iterator, Output_iterator, Operator, Value_type>, BLOCK_SIZE, 0);
    const size_t n_blocks   = cugar::min( max_blocks, size_t(n_leaves + BLOCK_SIZE-1) / BLOCK_SIZE );

    caching_device_vector<uint32> visited( n_nodes, 0u );

    reduce_from_leaves_kernel<BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        tree,
        in_values,
        out_values,
        op,
        def_value,
        raw_pointer( visited ) );
}

// reduce node values
template <uint32 BLOCK_SIZE, bool LEAF_REDUCTION, typename Tree, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
__global__ void reduce_from_nodes_kernel(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const Value_type        def_value,
    uint32*                 visited)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    const uint32 n_nodes = tree.get_node_count();

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
        base_idx < n_nodes;
        base_idx += grid_size)
    {
        uint32 node_id = threadIdx.x + base_idx;

        if (node_id < n_nodes)
        {
            // for the reduction, we only care about threads starting at a leaf node: i.e. we would really
            // like to spawn one thread per leaf, the only problem being that leaves can be sparse...
            if (tree.is_leaf( node_id ))
            {
                if (LEAF_REDUCTION)
                {
                    const uint2 leaf = tree.get_leaf_range( node_id );
                    const uint32 begin = leaf.x;
                    const uint32 end   = leaf.y;

                    Value_type value = def_value;
                    for (uint32 i = begin; i < end; ++i)
                        value = op( value, in_values[i] );

                    out_values[ node_id ] = value; // this write must be uncached to work

                    __threadfence(); // make sure this write is visible to all threads
                }

                // go up the tree
                while (1)
                {
                    node_id = tree.get_parent( node_id );
                    if (node_id == uint32(-1))
                        break;

                    const uint32 child_count = tree.get_child_count( node_id );

                    // check whether this is the last child of the current node to reach this point
                    if (atomicAdd( &visited[node_id], 1u ) != child_count-1)
                        break;

                    Value_type value = out_values[tree.get_child( node_id, 0 )];
                    for (uint32 i = 1; i < child_count; ++i)
                        value = op( value, out_values[tree.get_child( node_id, i )] );

                    out_values[ node_id ] = value; // this write must be uncached to work

                    __threadfence(); // make sure this write is visible to all threads
                }
            }
        }
    }
}

// reduce node values
template <bool LEAF_REDUCTION, typename Tree, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
void reduce_without_leaf_pointers(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const Value_type        def_value)
{
    const uint32 n_nodes = tree.get_node_count();

    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = cuda::max_active_blocks(reduce_from_nodes_kernel<BLOCK_SIZE, LEAF_REDUCTION, Tree, Input_iterator, Output_iterator, Operator, Value_type>, BLOCK_SIZE, 0);
    const size_t n_blocks   = cugar::min( max_blocks, size_t(n_nodes + BLOCK_SIZE-1) / BLOCK_SIZE );

    caching_device_vector<uint32> visited( n_nodes, 0u );

    reduce_from_nodes_kernel<BLOCK_SIZE, LEAF_REDUCTION> <<<n_blocks,BLOCK_SIZE>>> (
        tree,
        in_values,
        out_values,
        op,
        def_value,
        raw_pointer( visited ) );
}

} // namespace treereduce

//
// Reduce a bunch of values attached to the primitives of a tree.
//
template <typename Tree_visitor, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
void tree_reduce(
    const Tree_visitor      tree,
    const Input_iterator    in_values,
    Output_iterator         node_values,
    const Operator          op,
    const Value_type        def_value)
{
    // check whether there is some work
    if (tree.get_node_count() == 0)
        return;

    // check which specialization to use
    if (tree.has_leaf_pointers())
        treereduce::reduce_with_leaf_pointers( tree, in_values, node_values, op, def_value );
    else
        treereduce::reduce_without_leaf_pointers<true>( tree, in_values, node_values, op, def_value );
}

//
// Reduce a bunch of values attached to the leaves of a tree, with a simple bottom-up propagation.
//
template <typename Tree_visitor, typename Value_iterator, typename Operator>
void tree_reduce(
    const Tree_visitor      tree,
    Value_iterator          values,
    const Operator          op)
{
    // check whether there is some work
    if (tree.get_node_count() == 0)
        return;

    treereduce::reduce_without_leaf_pointers<false>( tree, values, values, op, def_value );
}

} // namespace cuda
} // namespace cugar
