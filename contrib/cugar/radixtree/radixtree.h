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

	///
	/// \page radixtree_page Radix Trees Module
	///
	/// This \ref radixtree "module" provides functions to build radix trees on top
	/// of sorted integer sequences.
	/// In practice, if the integers are seen as Morton codes of spatial points, the
	/// algorithms generate a middle-split k-d tree.
	///
	/// The following code snippet shows an example of how to use such builders:
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
	/// // build a tree writer
	/// Bintree_writer<node_type, device_tag> tree_writer( nodes, leaves );
	///
	/// // ...and generate it!
	/// cuda::generate_radix_tree(
	///     n_points,
	///     thrust::raw_pointer_cast( &codes.front() ),
	///     30u,
	///     16u,
	///     false,
	///		true,
	///     tree_writer );
	/// 
	///  \endcode

///@addtogroup bintree
///@{
	
///@defgroup radixtree Radix Trees
/// This module defines functions to generate binary Radix Trees on top of sorted integer sequences
///@{

///
/// Generate a binary radix tree from a set of sorted integers,
/// splitting the set top-down at each occurrence of a bit
/// set to 1.
/// In practice, if the integers are seen as Morton codes,
/// this algorithm generates a middle-split k-d tree.
///
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
    const uint32            n_codes,
    const Integer*          codes,
    const uint32            bits,
    const uint32            max_leaf_size,
    const bool              keep_singletons,
	const bool				middle_splits,
    Tree_writer&            tree);

///@} radixtree
///@} bintree

} // namespace cugar

#include <cugar/radixtree/radixtree_inline.h>
