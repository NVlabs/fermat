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

/*! \file sah_builder.h
 *   \brief Interface for a CUDA-based Surface Area Heuristic BVH builder.
 */

#pragma once

#include <nih/bvh/bvh.h>
#include <nih/linalg/vector.h>
#include <nih/linalg/bbox.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

namespace sah {

    struct Bin
    {
        float4 bmin;
        float4 bmax;

        FORCE_INLINE NIH_HOST_DEVICE int32 get_size() const { return binary_cast<int32>( bmin.w ); }
        FORCE_INLINE NIH_HOST_DEVICE void  set_size(const int32 x) { bmin.w = binary_cast<float>(x); }
    };
    struct Bbox
    {
        float3 bmin;
        float3 bmax;
    };
    struct Bins
    {
        float3* bmin;
        float3* bmax;
        int32*  size;
    };

} // namespace sah

/*! \addtogroup bvh Bounding Volume Hierarchies
 *  \{
 */

///
/// GPU-based SAH BVH builder.
///
/// This builders provides the context to generate a bounding volume hierarchy
/// using the Surface Area Heuristic out of a generic set of unsorted bounding
/// boxes.
/// The output is a set of nodes with the corresponding leaves and
/// a set of primitive indices into the input set of points.
/// The output leaves will specify contiguous ranges into this index.
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <nih/bvh/cuda/sah_builder.h>
///
/// thrust::device_vector<Bbox4f>   bboxes;
/// ... // code to fill the input vector of bboxes
///
/// thrust::device_vector<Bvh_node> bvh_nodes;
/// thrust::device_vector<uint2>    bvh_leaves;
/// thrust::device_vector<uint32>   bvh_index;
///
/// nih::cuda::Sah_builder builder( bvh_nodes, bvh_leaves, bvh_index );
/// builder.build(
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     bboxes.begin(),                             // begin iterator
///     bboxes.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///
struct Sah_builder
{
    /// constructor
    ///
    /// \param nodes        output nodes array
    /// \param leaves       output leaf array
    /// \param index        output index array
    Sah_builder(
        thrust::device_vector<Bvh_node>&         nodes,
        thrust::device_vector<uint2>&            leaves,
        thrust::device_vector<uint32>&           index) :
        m_nodes( &nodes ), m_leaves( &leaves ), m_index( &index )
    {
        m_sorting_time            = 0.0f;
        m_compression_time        = 0.0f;
        m_sah_split_time          = 0.0f;
        m_distribute_objects_time = 0.0f;
        m_temp_storage = 0;
    }

    /// build a bvh given a set of bboxes.
    /// The bbox iterators must be thrust-compatible (i.e. implementing dereference()).
    ///
    /// \param bbox             global bbox
    /// \param bbox_begin       beginning of the bbox sequence
    /// \param bbox_end         end of the bbox sequence
    /// \param max_leaf_size    maximum leaf size
    /// \param max_cost         maximum cost relative to the parent
    template <typename Iterator>
    void build(
        const Bbox3f    bbox,
        const Iterator  bbox_begin,
        const Iterator  bbox_end,
        const uint32    max_leaf_size,
        const float     max_cost = 1.8f);

    thrust::device_vector<Bvh_node>*    m_nodes;
    thrust::device_vector<uint2>*       m_leaves;
    thrust::device_vector<uint32>*      m_index;
    uint32                              m_levels[128];
    uint32                              m_level_count;
    uint32                              m_node_count;
    uint32                              m_leaf_count;

    float   m_sorting_time;
    float   m_compression_time;
    float   m_sah_split_time;
    float   m_distribute_objects_time;
    uint32  m_temp_storage;

private:
    void sort(
        const uint32                            n_objects,
        thrust::device_vector<uint32>::iterator segment_ids,
        thrust::device_vector<uint32>::iterator segment_keys,
        thrust::device_vector<uint2>::iterator  in_bounds,
        thrust::device_vector<uint2>::iterator  bounds,
        thrust::device_vector<uint2>::iterator  bounds_tmp,
        thrust::device_vector<uint32>::iterator order,
        uint32&                                 n_active_objects,
        uint32&                                 n_segments);

    void eval_split_costs(
        const uint32                            n_objects,
        const uint32                            n_segments,
        thrust::device_vector<uint32>::iterator segment_keys,
        thrust::device_vector<uint32>::iterator segment_heads,
        thrust::device_vector<uint2>::iterator  bounds,
        thrust::device_vector<uint2>::iterator  bounds_l,
        thrust::device_vector<uint2>::iterator  bounds_r,
        thrust::device_vector<float>::iterator  split_costs,
        thrust::device_vector<uint32>::iterator split_index);

    typedef sah::Bin     Bin;
    typedef sah::Bbox    Bbox;
    typedef sah::Bins    Bins;

    thrust::device_vector<uint32> m_segment_heads;
    thrust::device_vector<uint32> m_segment_keys;
    thrust::device_vector<Bin>    m_queue_bins[2];
    thrust::device_vector<uint2>  m_scan_bounds;
    thrust::device_vector<float>  m_split_costs;
    thrust::device_vector<uint32> m_split_index;

    thrust::device_vector<uint32> m_segment_ids;
    thrust::device_vector<uint32> m_node_ids;
    thrust::device_vector<uint32> m_counters;
};

/*! \}
 */

} // namespace cuda
} // namespace nih

#include <nih/bvh/cuda/sah_builder_inline.h>
