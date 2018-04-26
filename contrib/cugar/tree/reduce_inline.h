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

namespace cugar {

namespace treereduce {

//
// Reduce a bunch of values attached to the primitives of a tree.
//
template <typename Tree_visitor, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
void tree_reduce(
    const uint32            node_id,
    const Tree_visitor      tree,
    const Input_iterator    in_values,
    Output_iterator         out_values,
    const Operator          op,
    const Value_type        def_value)
)
{
    if (tree.is_leaf(node_id))
    {
        const uint2 leaf = tree.get_leaf_range( node_id );
        const uint32 begin = leaf.x;
        const uint32 end   = leaf.y;

        Value_type value = def_value;
        for (uint32 i = begin; i < end; ++i)
            value = op( value, in_values[i] );

        out_values[ node_id ] = value; // this write must be uncached to work
    }
    else
    {
        const uint32 child_count = tree.get_child_count( node_id );
        for (uint32 i = 0; i < child_count; ++i)
        {
            tree_reduce(
                tree.get_child( node_id, i ),
                tree,
                in_values,
                out_values,
                op,
                def_value );
        }

        Value_type value = out_values[tree.get_child( node_id, 0 )];
        for (uint32 i = 1; i < child_count; ++i)
                value = op( value, out_values[tree.get_child( node_id, i )] );

        out_values[ node_id ] = value;
    }
}

//
// Reduce a bunch of values attached to the leaf nodes of a tree.
//
template <typename Tree_visitor, typename Output_iterator, typename Operator>
void tree_reduce(
    const uint32            node_id,
    const Tree_visitor      tree,
    Output_iterator         out_values,
    const Operator          op)
)
{
    if (!tree.is_leaf(node_id))
    {
        const uint32 child_count = tree.get_child_count( node_id );
        for (uint32 i = 0; i < child_count; ++i)
        {
            tree_reduce(
                tree.get_child( node_id, i ),
                tree,
                out_values
                op );
        }

        Value_type value = out_values[tree.get_child( node_id, 0 )];
        for (uint32 i = 1; i < child_count; ++i)
                value = op( value, out_values[tree.get_child( node_id, i )] );

        out_values[ node_id ] = value;
    }
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
    treereduce::tree_reduce(
        0u,
        tree,
        in_values,
        node_values,
        op,
        def_value );
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
    treereduce::tree_reduce(
        0u,
        tree,
        node_values,
        op );
}

} // namespace cugar
