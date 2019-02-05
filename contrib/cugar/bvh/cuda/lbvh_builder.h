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

/*! \file lbvh_builder.h
 *   \brief Interface for a CUDA-based LBVH builder.
 */

#pragma once

#include <cugar/bvh/bvh.h>
#include <cugar/basic/vector.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>
#include <cugar/bintree/bintree_node.h>
#include <cugar/bintree/bintree_writer.h>
#include <cugar/radixtree/cuda/radixtree_context.h>
#include <cugar/basic/cuda/timer.h>
#include <thrust/device_vector.h>

namespace cugar {
namespace cuda {

/*! \addtogroup bvh Bounding Volume Hierarchies
 *  \{
 */

struct LBVH_builder_stats
{
    cuda::Timer morton_time;
    cuda::Timer sorting_time;
    cuda::Timer build_time;
};

///
/// GPU-based Linear BVH builder
///
/// This class provides the context to generate LBVHs on the GPU
/// starting from a set of unordered points.
/// The output is a set of nodes with the corresponding leaves and
/// a set of primitive indices into the input set of points.
/// The output leaves will specify contiguous ranges into this index.
///
/// \tparam integer     an integer type that determines the number
///                     of bits used to compute the points' Morton codes.
///                     Accepted values are uint32 and uint64.
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <cugar/bvh/cuda/lbvh_builder.h>
///
/// thrust::device_vector<Vector3f> points;
/// ... // code to fill the input vector of points
///
/// thrust::device_vector<Bvh_node> bvh_nodes;
/// thrust::device_vector<uint32>   bvh_index;
///
/// cugar::cuda::LBVH_builder<uint64> builder( &bvh_nodes, &bvh_index );
/// builder.build(
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     points.begin(),                             // begin iterator
///     points.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///
template <
    typename integer,
    typename bvh_node_type = Bvh_node,
    typename node_vector  = vector<device_tag,bvh_node_type>,
    typename range_vector = vector<device_tag,uint2>,
    typename index_vector = vector<device_tag,uint32> >
struct LBVH_builder
{
    typedef LBVH_builder_stats Stats;

    /// constructor
    ///
    /// \param nodes        output nodes array
    /// \param leaves       output leaf array
    /// \param index        output index array
    LBVH_builder(
        node_vector*    nodes           = NULL,
        index_vector*   index           = NULL,
        range_vector*   leaf_ranges     = NULL,
        index_vector*   leaf_pointers   = NULL,
        index_vector*   parents         = NULL,
        index_vector*   skip_nodes      = NULL,
		range_vector*   node_ranges     = NULL) :
        m_nodes( nodes ),
        m_leaf_ranges( leaf_ranges ),
        m_leaf_pointers( leaf_pointers ),
        m_parents( parents ),
        m_skip_nodes( skip_nodes ),
        m_node_ranges( node_ranges ),
        m_index( index ) {}

    void set_nodes(node_vector*             nodes)          { m_nodes = nodes; }
    void set_index(index_vector*            index)          { m_index = index; }
    void set_parents(index_vector*          parents)        { m_parents = parents; }
    void set_skip_nodes(index_vector*       skip_nodes)     { m_skip_nodes = skip_nodes; }
    void set_leaf_pointers(index_vector*    leaf_pointers)  { m_leaf_pointers = leaf_pointers; }
    void set_leaf_ranges(range_vector*      leaf_ranges)    { m_leaf_ranges = leaf_ranges; }
    void set_node_ranges(range_vector*      node_ranges)    { m_node_ranges = node_ranges; }

    /// build a bvh given a set of points that will be reordered in-place
    ///
    /// \param bbox             global bbox
    /// \param points_begin     beginning of the point sequence to sort
    /// \param points_end       end of the point sequence to sort
    /// \param max_leaf_size    maximum leaf size
    template <typename Iterator>
    void build(
        const Bbox3f    bbox,
        const Iterator  points_begin,
        const Iterator  points_end,
        const uint32    max_leaf_size,
        Stats*          stats = NULL);

    node_vector*                            m_nodes;
    range_vector*                           m_node_ranges;
    range_vector*                           m_leaf_ranges;
    index_vector*                           m_leaf_pointers;
    index_vector*                           m_parents;
    index_vector*                           m_skip_nodes;
    index_vector*                           m_index;
    caching_device_vector<integer>          m_codes;
    caching_device_vector<integer>          m_temp_codes;
    caching_device_vector<uint32>           m_temp_index;
    Bbox3f                                  m_bbox;
    uint32                                  m_node_count;
    uint32                                  m_leaf_count;
    cuda::Radixtree_context                 m_kd_context;
};

/*! \}
 */

} // namespace cuda
} // namespace cugar

#include <cugar/bvh/cuda/lbvh_builder_inline.h>