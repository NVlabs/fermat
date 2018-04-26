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

#include <nih/basic/utils.h>
#include <nih/basic/functors.h>
#include <nih/basic/cuda/scan.h>
#include <nih/basic/cuda_config.h>
#include <nih/thrust/iterator_wrapper.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace nih {
namespace cuda {

namespace sah {

inline void start_timer(const cudaEvent_t start, const cudaEvent_t stop)
{
    cudaEventRecord( start, 0 );
}
inline float stop_timer(const cudaEvent_t start, const cudaEvent_t stop)
{
    float dtime;
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &dtime, start, stop );
    return dtime;
}

FORCE_INLINE NIH_HOST_DEVICE uint32 largest_axis(const float3 edge)
{
    return edge.x > edge.y ?
        (edge.x > edge.z ? 0 : 2) :
        (edge.y > edge.z ? 1 : 2);
}

///
/// Functor to compress bboxes relative to their parent
///
struct Bbox_compressor
{
    typedef thrust::tuple<uint32,Bbox4f>    argument_type;
    typedef uint2                           result_type;

    Bbox_compressor(const Bin* bins) : m_bins( bins ) {}

    FORCE_INLINE NIH_HOST_DEVICE uint2 operator() (const argument_type op) const
    {
        const uint32 key = thrust::get<0>( op );
        if (key == uint32(-1))
            return make_uint2(0,0);

        const Bbox4f bbox = thrust::get<1>( op );

        const float4 bmin = m_bins[ key-1 ].bmin;
        const float4 bmax = m_bins[ key-1 ].bmax;

        const float3 delta = make_float3( bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z );

        const uint32 axis = largest_axis( delta );

        uint32 l;
        l  = delta.x < 1.0e-8f ? 0u : quantize( (bbox[0][0] - bmin.x) / delta.x, 1024 );
        l |= delta.y < 1.0e-8f ? 0u : quantize( (bbox[0][1] - bmin.y) / delta.y, 1024 ) << 10;
        l |= delta.z < 1.0e-8f ? 0u : quantize( (bbox[0][2] - bmin.z) / delta.z, 1024 ) << 20;
        l |= axis << 30;

        uint32 r;
        r  = delta.x < 1.0e-8f ? 0u : quantize( (bbox[1][0] - bmin.x) / delta.x, 1024 );
        r |= delta.y < 1.0e-8f ? 0u : quantize( (bbox[1][1] - bmin.y) / delta.y, 1024 ) << 10;
        r |= delta.z < 1.0e-8f ? 0u : quantize( (bbox[1][2] - bmin.z) / delta.z, 1024 ) << 20;

        return make_uint2( l, r );
    }

    const Bin* m_bins;
};

// evaluate the new splits, choosing which ones to keep and which ones to discard.
void eval_splits(
    const uint32    n_nodes,
          uint32*   split_planes,
    const float*    split_costs,
    const uint32*   segment_heads,
    uint32*         out_splits,
    const uint32    max_leaf_size,
    const float     max_cost);

// assign objects to their new nodes
void assign_objects(
    const uint32    n_objects,
    const uint32    n_leaves,
    const uint32*   order,
    const uint32*   segment_keys,
    const uint32*   split_index,
    const uint32*   allocation_map,
    uint32*         segment_ids,
    uint32*         leaf_ids);

// compute the bounding box of the output segments
void compute_bins(
    const uint32    n_segments,
    const uint32    n_nodes,
    const uint32    n_leaves,
    const uint32    input_node_offset,
    const uint32*   split_index,
    const uint32*   allocation_map,
    const Bin*      in_bins,
    const uint2*    bounds_l,
    const uint2*    bounds_r,
    Bin*            out_bins,
    Bvh_node*       bvh_nodes);

// setup the leaf array
void setup_leaves(
    const uint32    n_objects,
    const uint32*   leaf_ids,
    uint2*          leaves);

} // namespace sah

// build a bvh given a set of bboxes
template <typename Iterator>
void Sah_builder::build(
    const Bbox3f    bbox,
    const Iterator  bbox_begin,
    const Iterator  bbox_end,
    const uint32    max_leaf_size,
    const float     max_cost)
{
    const uint32 n_objects = uint32( bbox_end - bbox_begin );

    need_space( *m_nodes,  n_objects*2 );
    need_space( *m_leaves, n_objects );
    need_space( *m_index,  n_objects );

    uint32 storage = 0;

    need_space( m_segment_heads, n_objects+1 );     storage += (n_objects+1) * sizeof(uint32);
    need_space( m_segment_keys,  n_objects );       storage += n_objects * sizeof(uint32);

    need_space( m_queue_bins[0], n_objects / max_leaf_size );   storage += (n_objects / max_leaf_size) * sizeof(Bin);
    need_space( m_queue_bins[1], n_objects / max_leaf_size );   storage += (n_objects / max_leaf_size) * sizeof(Bin);

    need_space( m_segment_ids,   n_objects );       storage += n_objects * sizeof(uint32);
    need_space( m_node_ids,      n_objects );       storage += n_objects * sizeof(uint32);

    need_space( m_scan_bounds,   n_objects * 3 );   storage += n_objects*3 * sizeof(uint2);
    need_space( m_split_costs,   n_objects );       storage += n_objects   * sizeof(float);
    need_space( m_split_index,   n_objects );       storage += n_objects   * sizeof(uint32);

    // assign all objects to node 0
    thrust::fill( m_node_ids.begin(),  m_node_ids.begin()  + n_objects, 0 );

    thrust::fill( m_segment_keys.begin(),  m_segment_keys.begin()  + n_objects, 1u );
    thrust::fill( m_segment_ids.begin(),   m_segment_ids.begin()   + n_objects, 1u );

    m_segment_heads[0] = 0;
    m_segment_heads[1] = n_objects;

    thrust::device_vector<float>::iterator centroids = m_split_costs.begin();

    int in_queue  = 0;
    int out_queue = 1;

    // initialize root bounding box
    {
        Bin bin;
        bin.bmin = make_float4( bbox[0][0],bbox[0][1],bbox[0][2],binary_cast<float>(n_objects) );
        bin.bmax = make_float4( bbox[1][0],bbox[1][1],bbox[1][2],binary_cast<float>(Bvh_node::kInvalid) );
        m_queue_bins[ in_queue ][0] = bin;
    }

    m_counters.resize(2);
    m_counters[0] = 0;
    m_counters[1] = 0;

    uint32 input_node_offset = 0;
    uint32 n_leaves          = 0;
    uint32 n_nodes           = 1;
    uint32 out_nodes         = 1;

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    thrust::device_vector<uint2>::iterator bounds_l = m_scan_bounds.begin() + n_objects*0;
    thrust::device_vector<uint2>::iterator bounds_r = m_scan_bounds.begin() + n_objects*1;
    thrust::device_vector<uint2>::iterator bounds   = m_scan_bounds.begin() + n_objects*2;

    //const float HUGE = 1.0e9f;

    uint32 n_active_objects = n_objects;

    m_levels[0] = 0;
    int32 level = 0;

    while (out_nodes)
    {
        // mark the beginning of the new level
        m_levels[ level++ ] = n_nodes;

        // compress the bounds relative to their parents
        {
            sah::start_timer( start, stop );

            sah::Bbox_compressor bbox_compressor(
                thrust::raw_pointer_cast( &m_queue_bins[ in_queue ].front() ) );

            thrust::transform(
                thrust::make_zip_iterator( thrust::make_tuple( m_segment_ids.begin(), bbox_begin ) ),
                thrust::make_zip_iterator( thrust::make_tuple( m_segment_ids.begin(), bbox_begin ) ) + n_objects,
                bounds_l,
                bbox_compressor );

            m_compression_time += sah::stop_timer( start, stop );
        }

        uint32 n_segments = out_nodes;

        //
        // Build the largest axis ordering for this pass. Basically, sort the
        // objects by the segment they belong to and their centroid along the
        // the largest axis of their segment.
        //
        thrust::device_vector<uint32>::iterator order = m_index->begin();

        sah::start_timer( start, stop );

        sort(
            n_objects,
            m_segment_ids.begin(),
            m_segment_keys.begin(),
            bounds_l,
            bounds,
            bounds_r,
            order,
            n_active_objects,
            n_segments );

        m_sorting_time += sah::stop_timer( start, stop );

        sah::start_timer( start, stop );

        eval_split_costs(
            n_active_objects,
            n_segments,
            m_segment_keys.begin(),
            m_segment_heads.begin(),
            bounds,
            bounds_l,
            bounds_r,
            m_split_costs.begin(),
            m_split_index.begin() );

        thrust::device_ptr<uint32> allocation_map( (uint32*)thrust::raw_pointer_cast( &*bounds ) );

        // evaluate the new splits, choosing which ones to keep and which ones to discard
        sah::eval_splits(
            n_segments,
            thrust::raw_pointer_cast( &m_split_index.front() ),
            thrust::raw_pointer_cast( &m_split_costs.front() ),
            thrust::raw_pointer_cast( &m_segment_heads.front() ),
            thrust::raw_pointer_cast( &*allocation_map ),
            max_leaf_size,
            max_cost );

        m_sah_split_time += sah::stop_timer( start, stop );

        sah::start_timer( start, stop );

        // scan the split booleans to find out the new node offsets
        thrust::inclusive_scan( allocation_map, allocation_map + n_segments, allocation_map );

        const uint32 n_splits = allocation_map[ n_segments-1 ];
        out_nodes = n_splits*2;

        // assign the objects to their new nodes
        sah::assign_objects(
            n_active_objects,
            n_leaves,
            thrust::raw_pointer_cast( &*order ),
            thrust::raw_pointer_cast( &m_segment_keys.front() ),
            thrust::raw_pointer_cast( &m_split_index.front() ),
            thrust::raw_pointer_cast( &*allocation_map ),
            thrust::raw_pointer_cast( &m_segment_ids.front() ),
            thrust::raw_pointer_cast( &m_node_ids.front() ) );

        // realloc the output queue bins if needed
        storage -= sizeof(Bin) * m_queue_bins[ out_queue ].size();
        need_space( m_queue_bins[ out_queue ], out_nodes );
        storage += sizeof(Bin) * m_queue_bins[ out_queue ].size();

        // compute the bounding box of the output segments
        sah::compute_bins(
            n_segments,
            n_nodes,
            n_leaves,
            input_node_offset,
            thrust::raw_pointer_cast( &m_split_index.front() ),
            thrust::raw_pointer_cast( &*allocation_map ),
            thrust::raw_pointer_cast( &m_queue_bins[ in_queue ].front() ),
            thrust::raw_pointer_cast( &*bounds_l ),
            thrust::raw_pointer_cast( &*bounds_r ),
            thrust::raw_pointer_cast( &m_queue_bins[ out_queue ].front() ),
            thrust::raw_pointer_cast( &m_nodes->front() ) );

        m_distribute_objects_time += sah::stop_timer( start, stop );

        input_node_offset = n_nodes;

        n_nodes  += out_nodes;
        n_leaves += (n_segments - n_splits);

        std::swap( in_queue, out_queue );
    }

    m_level_count = level;
    for (; level < 128; ++level)
        m_levels[ level ] = n_nodes;

    // sort the objects by their leaf id
    thrust::copy( thrust::make_counting_iterator(0u), thrust::make_counting_iterator(0u) + n_objects, m_index->begin() );
    thrust::sort_by_key( m_node_ids.begin(), m_node_ids.begin() + n_objects, m_index->begin() );

    // setup leaf ranges
    sah::setup_leaves(
        n_objects,
        thrust::raw_pointer_cast( &m_node_ids.front() ),
        thrust::raw_pointer_cast( &m_leaves->front() ) );

    m_leaf_count = n_leaves;
    m_node_count = n_nodes;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    m_temp_storage = nih::max( storage, m_temp_storage );
}

} // namespace cuda
} // namespace nih
