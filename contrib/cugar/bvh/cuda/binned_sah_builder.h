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

#include <nih/bvh/bvh.h>
#include <nih/linalg/vector.h>
#include <nih/linalg/bbox.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

namespace binned_sah {

    struct Bin
    {
        float4 bmin;
        float4 bmax;

        FORCE_INLINE NIH_HOST_DEVICE int32 get_size() const { return binary_cast<int32>( bmin.w ); }
        FORCE_INLINE NIH_HOST_DEVICE void  set_size(const int32 x) { bmin.w = binary_cast<float>(x); }
    };
    struct Split
    {
        NIH_HOST_DEVICE Split() {}
        NIH_HOST_DEVICE Split(int32 id, int32 plane) : task_id(id), best_plane(plane) {}

        int32 task_id;
        int32 best_plane;
    };
    struct Queue
    {
        Bin*    bins;
        Split*  splits;
        uint32* offsets;
        int32   size;
    };
    struct Objects
    {
        int4*   bin_ids;
        int32*  node_ids;
        int32*  split_ids;
        uint32* index;
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
        volatile int32* size;
    };

} // namespace binned_sah

/*! \addtogroup bvh Bounding Volume Hierarchies
 *  \{
 */

///
/// GPU-based binned SAH BVH builder.
///
/// This builders provides the context to generate a bounding volume hierarchy
/// using the binned Surface Area Heuristic out of a generic set of unsorted
/// bounding boxes.
///
struct Binned_sah_builder
{
    /// constructor
    ///
    /// \param nodes        output nodes array
    /// \param leaves       output leaf array
    /// \param index        output index array
    Binned_sah_builder(
        thrust::device_vector<Bvh_node>&         nodes,
        thrust::device_vector<uint2>&            leaves,
        thrust::device_vector<uint32>&           index) :
        m_nodes( &nodes ), m_leaves( &leaves ), m_index( &index )
    {
        m_init_bins_time          = 0.0f;
        m_update_bins_time        = 0.0f;
        m_sah_split_time          = 0.0f;
        m_distribute_objects_time = 0.0f;
    }

    /// build a bvh given a set of bboxes
    ///
    /// \param BINS             number of bins to use
    /// \param bbox             global bbox
    /// \param bbox_begin       beginning of the bbox sequence
    /// \param bbox_end         end of the bbox sequence
    /// \param max_leaf_size    maximum leaf size
    /// \param max_cost         maximum cost relative to the parent
    template <typename Iterator>
    void build(
        const uint32    BINS,
        const Bbox3f    bbox,
        const Iterator  bbox_begin,
        const Iterator  bbox_end,
        const Iterator  h_bbox_begin,
        const uint32    max_leaf_size,
        const float     max_cost = 1.8f);

    thrust::device_vector<Bvh_node>*    m_nodes;
    thrust::device_vector<uint2>*       m_leaves;
    thrust::device_vector<uint32>*      m_index;
    uint32                              m_levels[128];
    uint32                              m_level_count;
    Bbox3f                              m_bbox;
    uint32                              m_node_count;
    uint32                              m_leaf_count;

    float m_init_bins_time;
    float m_update_bins_time;
    float m_sah_split_time;
    float m_distribute_objects_time;

    typedef binned_sah::Bin     Bin;
    typedef binned_sah::Bbox    Bbox;
    typedef binned_sah::Split   Split;
    typedef binned_sah::Queue   Queue;
    typedef binned_sah::Objects Objects;
    typedef binned_sah::Bins    Bins;

    thrust::device_vector<float3> m_bin_bmin;
    thrust::device_vector<float3> m_bin_bmax;
    thrust::device_vector<int32>  m_bin_size;
    thrust::device_vector<Bin>    m_queue_bins;
    thrust::device_vector<Split>  m_queue_splits;
    thrust::device_vector<uint32> m_queue_offsets;

    thrust::device_vector<int4>   m_bin_ids;
    thrust::device_vector<int32>  m_split_ids;
    thrust::device_vector<int32>  m_node_ids;
    thrust::device_vector<uint32> m_new_pos;
    thrust::device_vector<uint32> m_counters;
};

/*! \}
 */

} // namespace cuda
} // namespace nih

#include <nih/bvh/cuda/binned_sah_builder_inline.h>
