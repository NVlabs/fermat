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

#include <cugar/radixtree/cuda/radixtree.h>
#include <cugar/bits/morton.h>
#include <cugar/basic/cuda/sort.h>
#include <thrust/transform.h>

namespace cugar {
namespace cuda {

namespace lbvh {

    template <typename integer>
    struct Morton_bits {};

	template <>
    struct Morton_bits<uint32> { static const uint32 value = 30u; };

    template <>
    struct Morton_bits<uint64> { static const uint32 value = 60u; };

};

// build a Linear BVH given a set of points
template <
    typename integer,
    typename bvh_node_type,
    typename node_vector,
    typename range_vector,
    typename index_vector>
template <typename Iterator>
void LBVH_builder<integer, bvh_node_type, node_vector, range_vector, index_vector>::build(
    const Bbox3f    bbox,
    const Iterator  points_begin,
    const Iterator  points_end,
    const uint32    max_leaf_size,
    Stats*          stats)
{ 
    const uint32 n_points = uint32( points_end - points_begin );

    m_bbox = bbox;
    need_space( m_codes,      n_points );
    need_space( *m_index,     n_points );
    need_space( m_temp_codes, n_points );
    need_space( m_temp_index, n_points );

    cuda::Timer timer;
    if (stats)
        stats->morton_time.start();

    if (n_points > 0)
    {
        // compute the Morton code for each point
        thrust::transform(
            points_begin,
            points_begin + n_points,
            m_codes.begin(),
            morton_functor<integer, 3u>( bbox ) );

        // setup the point indices, from 0 to n_points-1
        thrust::copy(
            thrust::counting_iterator<uint32>(0),
            thrust::counting_iterator<uint32>(0) + n_points,
            m_index->begin() );
    }

    if (stats)
    {
        stats->morton_time.stop();
        stats->sorting_time.start();
    }

    if (n_points > 1)
    {
        // sort the indices by Morton code
        SortBuffers<integer*,uint32*> sort_buffers;
        sort_buffers.keys[0] = raw_pointer( m_codes );
        sort_buffers.keys[1] = raw_pointer( m_temp_codes );
        sort_buffers.values[0] = raw_pointer( *m_index );
        sort_buffers.values[1] = raw_pointer( m_temp_index );

        SortEnactor sort_enactor;
        sort_enactor.sort( n_points, sort_buffers );

        // check whether we need to copy the sort results back in place
        if (sort_buffers.selector)
        {
            thrust::copy( m_temp_codes.begin(), m_temp_codes.begin() + n_points, m_codes.begin() );
            thrust::copy( m_temp_index.begin(), m_temp_index.begin() + n_points, m_index->begin() );
        }
    }

    if (stats)
    {
        stats->sorting_time.stop();
        stats->build_time.start();
    }

    // generate a kd-tree
    Bintree_writer<bvh_node_type,device_tag,node_vector,range_vector,index_vector> tree_writer;
    tree_writer.set_nodes( m_nodes );
    tree_writer.set_leaf_ranges( m_leaf_ranges );
    tree_writer.set_leaf_pointers( m_leaf_pointers );
    tree_writer.set_parents( m_parents );
    tree_writer.set_skip_nodes( m_skip_nodes );
    tree_writer.set_node_ranges( m_node_ranges );

    const uint32 bits = lbvh::Morton_bits<integer>::value;

    generate_radix_tree(
        m_kd_context,
        n_points,
        raw_pointer( m_codes ),
        bits,
        max_leaf_size,
        false,
		true,
        tree_writer );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;

    if (stats)
        stats->build_time.stop();
}

} // namespace cuda
} // namespace cugar
