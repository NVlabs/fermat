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

/*! \file reduce.h
 *   \brief Defines utility function to reduce a set of values attached
 *          to the elements in the leaves of a tree.
 */

#pragma once

#include <cugar/basic/types.h>
#include <iterator>

namespace cugar {
namespace cuda {

///@addtogroup TreesModule
///@{

///
/// Reduce a bunch of values attached to the elemens in the leaves of a tree.
/// The Tree_visitor template type has to provide the following interface:
///
/// \code
///
/// struct Tree_visitor
/// {
///     // get node count
///     //
///     uint32 get_node_count() const;
/// 
///     // get leaf count
///     //
///     uint32 get_leaf_count() const;
/// 
///     // get child count
///     //
///     // \param node    node index
///     uint32 get_child_count(const uint32 node) const;
/// 
///     // get i-th child (among the active ones)
///     //
///     // \param node    node index
///     // \param i        child index
///     uint32 get_child(const uint32 node, const uint32 i) const;
/// 
///     // get parent
///     //
///     // \param node    node index
///     uint32 get_parent(const uint32 node) const;
/// 
///     // get leaf range
///     //
///     // \param node    node index
///     uint2 get_leaf_range(const uint32 node) const;
/// 
///     // get primitive range size
///     //
///     // \param node    node index
///     uint2 get_range_size(const uint32 node) const;
/// 
///     // return whether it's possible to locate leaf nodes
///     bool has_leaf_pointers() const;
/// 
///     // return the index of the i-th leaf node
///     uint32 get_leaf_node(const uint32 i) const;
/// };
///
/// \endcode
///
/// The following code snippet illustrates an example usage:
///
/// \code
///
/// #include <cugar/tree/cuda/tree_reduce.h>
/// #include <cugar/tree/model.h>
///
/// struct merge_op
/// {
///     CUGAR_HOST_DEVICE Bbox4f operator() (
///         const Bbox4f op1,
///         const Bbox4f op2) const { return Bbox4f( op1, op2 ); }
/// };
///
/// // compute the bboxes of a tree
/// void compute_bboxes(
///     uint32      node_count,     // input tree nodes
///     uint32      leaf_count,     // input tree leaves
///     Bvh_node*   nodes,          // input tree nodes, device pointer
///     uint32*     parents,        // input tree node parents, device pointer
///     Bbox4f*     prim_bboxes,    // input primitive bboxes, device pointer
///     Bbox4f*     node_bboxes)    // output node bboxes, device pointer
/// {
///     // instantiate a breadth-first tree view
///     Bintree_visitor<Bvh_node> bvh(
///         node_count,
///         leaf_count,
///         nodes,
///         parents,
///         NULL,
///         NULL );
///
///     // compute a tree reduction
///     cuda::tree_reduce(
///         bvh,
///         prim_bboxes,
///         node_bboxes,
///         merge_op(),
///         Bbox4f() );
/// }
///
/// \endcode
///
template <typename Tree_visitor, typename Input_iterator, typename Output_iterator, typename Operator, typename Value_type>
void tree_reduce(
    const Tree_visitor      tree,
    const Input_iterator    in_values,
    Output_iterator         node_values,
    const Operator          op,
    const Value_type        def_value);

///
/// Reduce a bunch of values attached to the leaves of a tree, with a simple bottom-up propagation.
///
template <typename Tree_visitor, typename Value_iterator, typename Operator>
void tree_reduce(
    const Tree_visitor      tree,
    Value_iterator          values,
    const Operator          op);

///@} TreeModule

} // namespace cuda
} // namespace cugar

#include <cugar/tree/cuda/reduce_inline.h>

