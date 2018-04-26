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

/*! \file kd_node.h
 *   \brief Define basic k-d tree node structure.
 */

#pragma once

#include <cugar/basic/types.h>

namespace cugar {

/// \page kd_page K-d Trees Module
///\par
/// This \ref kdtree "module" implements data-structures and functions to store, build and manipulate K-d trees.
///
/// - Kd_node
/// - cugar::cuda::Kd_builder
///
///\par
/// as well as a submodule for K-NN lookups: \ref knn.
///\par
/// As an example, consider the following code to create a K-d tree on a set of points, in parallel, on the device:
/// \code
///
/// #include <cugar/kd/cuda/kd_builder.h>
/// #include <cugar/kd/cuda/kd_context.h>
///
/// cugar::vector<device_tag,Vector3f> kd_points;
/// ... // code to fill the input vector of points
///
/// cugar::vector<device_tag,Kd_node>  kd_nodes;
/// cugar::vector<device_tag,uint2>    kd_leaves;
/// cugar::vector<device_tag,uint32>   kd_index;
/// cugar::vector<device_tag,uint2>    kd_ranges;
///
/// cugar::cuda::Kd_builder<uint64> builder( kd_index );
/// cugar::cuda::Kd_context kd_tree( &kd_nodes, &kd_leaves, &kd_ranges );
/// builder.build(
///     kd_tree,                                    // output tree
///     kd_index,                                   // output index
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     kd_points.begin(),                          // begin iterator
///     kd_points.end(),                            // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///\par
/// The following k-NN example shows how to perform k-NN lookups on such a tree:
/// \code
///
/// #include <cugar/kd/cuda/kd_knn.h>
///
/// cugar::vector<device_tag,Vector4f> query_points;
/// ... // code to build the k-d tree and query points here...
///     // NOTE: even though we're doing 3-dimensional queries,
///     // we can use Vector4f arrays for better coalescing
///
/// cugar::vector<device_tag,cuda::Kd_knn<3>::Result> results( query_points.size() );
///
/// cugar::cuda::Kd_knn<3> knn;
/// knn.run(
///     query_points.begin(),
///     query_points.end(),
///     raw_pointer(kd_nodes),
///     raw_pointer(kd_ranges),
///     raw_pointer(kd_leaves),
///     raw_pointer(kd_points),
///     raw_pointer(results) );
/// 
///  \endcode
///

/*! \addtogroup kdtree k-d Trees
 *  \{
 */

///
/// A k-d tree node.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, its children
/// will be consecutive in memory.
/// Supports up to 7 dimensions.
///
struct Kd_node
{
    static const uint32 kInvalid = uint32(-1);

    /// empty constructor
    ///
    CUGAR_HOST_DEVICE Kd_node() {}

    /// leaf constructor
    ///
    /// \param index    child index
    CUGAR_HOST_DEVICE Kd_node(uint32 index) :
        m_packed_info( 7u | (index << 3) ) {}

    /// split node constructor
    ///
    /// \param split_dim    splitting dimension
    /// \param split_plane  splitting plane
    /// \param index        child index
    CUGAR_HOST_DEVICE Kd_node(const uint32 split_dim, const float split_plane, uint32 index) :
        m_packed_info( split_dim | (index << 3) ),
        m_split_plane( split_plane ) {}

    /// is a leaf?
    ///
    CUGAR_HOST_DEVICE uint32 is_leaf() const
    {
        return (m_packed_info & 7u) == 7u;
    }
    /// get offset of the first child
    ///
    CUGAR_HOST_DEVICE uint32 get_child_offset() const
    {
        return m_packed_info >> 3u;
    }
    /// get leaf index
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_index() const
    {
        return m_packed_info >> 3u;
    }
    /// get i-th child
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE uint32 get_child(const uint32 i) const
    {
        return get_child_offset() + i;
    }
    /// is the i-th child active?
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE bool has_child(const uint32 i) const
    {
        return is_leaf() ? false : true;
    }
    /// get left partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_left() const
    {
        return get_child_offset();
    }
    /// get right partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_right() const
    {
        return get_child_offset() + 1u;
    }

    /// get splitting dimension
    ///
    CUGAR_HOST_DEVICE uint32 get_split_dim() const { return (m_packed_info & 7u); }

    /// get splitting plane
    ///
    CUGAR_HOST_DEVICE float get_split_plane() const { return m_split_plane; }

    uint32 m_packed_info;
    float  m_split_plane;
};

/*! \}
 */

} // namespace cugar
