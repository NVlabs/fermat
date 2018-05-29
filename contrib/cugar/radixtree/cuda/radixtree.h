/*
 * Copyright (c) 2010-2016, NVIDIA Corporation
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

/*! \file radixtree.h
 *   \brief Defines a function to generate a binary tree from a sequence of
 *          sorted integers.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/radixtree/cuda/radixtree_context.h>
#include <thrust/device_vector.h>

namespace cugar {
namespace cuda {

///@addtogroup bintree
///@{

///@addtogroup radixtree
///@{

///
/// Generate a binary radix tree from a set of sorted integers,
/// splitting the set top-down at each occurrence of a bit
/// set to 1.
/// In practice, if the integers are seen as Morton codes,
/// this algorithm generates a middle-split k-d tree.
///
/// \param context          the generation context
/// \param n_codes          number of entries in the input set of codes
/// \param codes            input set of codes
/// \param bits             number of bits per code
/// \param max_leaf_size    maximum target number of entries per leaf
/// \param keep_singletons  mark whether to keep or suppress singleton nodes
///                         in the output tree
/// \param tree             output tree
///
/// The Tree_writer template parameter has to provide the following interface:
/// \anchor TreeWriterAnchor
///
/// \code
/// struct Tree_writer
/// {
///    void reserve_nodes(const uint32 n);  // reserve space for n nodes
///    void reserve_leaves(const uint32 n); // reserve space for n leaves
///
///    context_type get_context();          // get a context to write nodes/leaves
///
///    struct context_type
///    {
///        void write_node(
///           const uint32 node,          // node to write
///           const uint32 parent,        // node parent
///           const bool   left_child,    // specify whether the node has a left child
///           const bool   right_child,   // specify whether the node has a right child
///           const uint32 offset,        // child offset
///           const uint32 skip_node,     // skip node
///           const uint32 level,         // split level
///           const uint32 begin,         // node range begin
///           const uint32 end,           // node range end
///           const uint32 split_index);  // split index
///
///        void write_leaf(
///           const uint32 leaf_index,    // leaf to write
///           const uint32 node_index,    // node containing this leaf
///           const uint32 begin,         // leaf range begin
///           const uint32 end);          // leaf range end
///    };
/// };
/// \endcode
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <cugar/bintree/cuda/bintree_gen.h>
/// #include <cugar/bintree/cuda/bintree_context.h>
/// #include <cugar/bits/morton.h>
///
/// typedef Bintree_node<leaf_index_tag> node_type;
///
/// const uint32 n_points = 1000000;
/// cugar::vector<device_tag,Vecto3f> points( n_points );
/// ... // generate a bunch of points here
///
/// // compute their Morton codes
/// cugar::vector<device_tag,uint32> codes( n_points );
/// thrust::transform(
///     points.begin(),
///     points.begin() + n_points,
///     codes.begin(),
///     morton_functor<uint32,3>() );
///
/// // sort them
/// thrust::sort( codes.begin(), codes.end() );
///
/// // allocate storage for a binary tree...
/// cugar::vector<device_tag,node_type> nodes;
/// cugar::vector<device_tag,uint2>     leaves;
///
/// Bintree_writer<node_type, device_tag> tree_writer( nodes, leaves );
///
/// // ...and generate it!
/// cuda::generate_radix_tree(
///     n_points,
///     thrust::raw_pointer_cast( &codes.front() ),
///     30u,
///     16u,
///     false,
///     tree_writer );
/// 
///  \endcode
///
template <typename Tree_writer, typename Integer>
void generate_radix_tree(
    const uint32        n_codes,
    const Integer*      codes,
    const uint32        bits,
    const uint32        max_leaf_size,
    const bool          keep_singletons,
	const bool			middle_splits,
    Tree_writer&        tree);

///
/// Generate a binary tree from a set of sorted integers,
/// splitting the set top-down at each occurrence of a bit
/// set to 1.
/// In practice, if the integers are seen as Morton codes,
/// this algorithm generates a middle-split k-d tree.
///
/// \param context          the generation context
/// \param n_codes          number of entries in the input set of codes
/// \param codes            input set of codes
/// \param bits             number of bits per code
/// \param max_leaf_size    maximum target number of entries per leaf
/// \param keep_singletons  mark whether to keep or suppress singleton nodes
///                         in the output tree
/// \param middle_splits	mark whether to allow pure middle splits once a
///							group of integers are all equal. NOTE that if this
///							flag is set to false, the maximum leaf size cannot
///							be guaranteed
/// \param tree             output tree
///
template <typename Tree_writer, typename Integer>
void generate_radix_tree(
    Radixtree_context&  context,
    const uint32        n_codes,
    const Integer*      codes,
    const uint32        bits,
    const uint32        max_leaf_size,
    const bool          keep_singletons,
	const bool			middle_splits,
    Tree_writer&        tree);

///@} radixtree
///@} bintree

} // namespace cuda
} // namespace cugar

#include <cugar/radixtree/cuda/radixtree_inline.h>
