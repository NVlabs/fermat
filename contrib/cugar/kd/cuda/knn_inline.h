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
 *     documentation and/or far materials provided with the distribution.
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

#include <cugar/basic/cuda/arch.h>

//#define KD_KNN_STATS
#ifdef KD_KNN_STATS
#define KD_KNN_STATS_DEF(type,name,value) type name = value;
#define KD_KNN_STATS_ADD(name,value)      name += value;
#else
#define KD_KNN_STATS_DEF(type,name,value)
#define KD_KNN_STATS_ADD(name,value)
#endif

namespace cugar {
namespace cuda {

namespace kd_knn {

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float comp(const float2 v, const uint32 i)
{
	return i == 0 ? v.x : v.y;
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
void set_comp(float2& v, const uint32 i, const float t)
{
	if (i == 0) v.x = t;
	else		v.y = t;
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float sq_length(const float2 v)
{
    return v.x*v.x + v.y*v.y;
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float comp(const float3 v, const uint32 i)
{
    return
		i == 0 ? v.x :
		i == 1 ? v.y :
				 v.z;
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
void set_comp(float3& v, const uint32 i, const float t)
{
	if (i == 0)		 v.x = t;
	else if (i == 1) v.y = t;
	else			 v.z = t;
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float sq_length(const float3 v)
{
    return v.x*v.x + v.y*v.y + v.z*v.z;
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float comp(const float4 v, const uint32 i)
{
	return i < 2 ?
		(i == 0 ? v.x : v.y) :
		(i == 2 ? v.z : v.w);
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
void set_comp(float4& v, const uint32 i, const float t)
{
	if (i == 0)		 v.x = t;
	else if (i == 1) v.y = t;
	else if (i == 2) v.z = t;
	else			 v.w = t;
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float sq_length(const float4 v)
{
    return v.x*v.x + v.y*v.y + v.z*v.z+ v.w*v.w;
}

template <typename VectorType, typename PointIterator>
__device__ void lookup_2d(
    const VectorType        query,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const PointIterator     kd_points,
    Kd_knn<2>::Result*      results)
{
    //
    // 1-st pass: find the leaf containing the query point and compute an upper bound
    // on the search distance
    //

    uint32 idx;
    float  dist2 = 1.0e16f;

    // start from the root node
    uint32 node_index = 0;

    // keep track of which leaf we visited
    uint32 first_leaf = 0;

    while (1)
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            // find the closest neighbor in this leaf
            const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
            for (uint32 i = leaf.x; i < leaf.y; ++i)
            {
                const VectorType delta = kd_points[i] - query;
                const float d2 = delta[0]*delta[0] + delta[1]*delta[1];
                if (dist2 > d2)
                {
                    dist2 = d2;
                    idx   = i;
                }
            }

            // keep track of which leaf we found
            first_leaf = node_index;
            break;
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index = node.get_child_offset() + (comp( query, split_dim ) < split_plane ? 0u : 1u);
        }
    }

    //
    // 2-nd pass: visit the tree with a stack and careful pruning
    //

    int32  stackp = 1;
    float4 stack[64];

    // place a sentinel node in the stack
    stack[0] = make_float4( 0.0f, 0.0f, 0.0f, binary_cast<float>(uint32(-1)) );

    // start from the root node
    node_index = 0;

    float2 cdist = make_float2( 0.0f, 0.0f );

    while (node_index != uint32(-1))
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            if (first_leaf != node_index)
            {
                // find the closest neighbor in this leaf
                const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
                for (uint32 i = leaf.x; i < leaf.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1];
                    if (dist2 > d2)
                    {
                        dist2 = d2;
                        idx   = i;
                    }
                }
            }

            // pop the next node from the stack
            while (stackp > 0)
            {
                const float4 stack_node = stack[ --stackp ];
                node_index = binary_cast<uint32>( stack_node.w );
                cdist      = make_float2( stack_node.x, stack_node.y );

                if (sq_length( cdist ) < dist2)
                    break;
            }
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            const float split_dist = comp( query, split_dim ) - split_plane;

            const uint32 select = split_dist <= 0.0f ? 0u : 1u;

            node_index = node.get_child_offset() + select;

            // compute the vector distance to the far node
            float2 cdist_far = cdist;
			set_comp( cdist_far, split_dim, split_dist );

            // check whether we should push the far node on the stack
            const float dist_far2 = sq_length( cdist_far );

            if (dist_far2 < dist2)
            {
                stack[ stackp++ ] = make_float4(
                    cdist_far.x,
                    cdist_far.y,
                    0,
                    binary_cast<float>( node.get_child_offset() + 1u - select ) );
            }
        }
    }

    // write the result
    results->index = idx;
    results->dist2 = dist2;
}

template <typename VectorType, typename PointIterator>
__device__ void lookup_3d(
    const VectorType        query,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const PointIterator     kd_points,
    Kd_knn<3>::Result*      results)
{
    //
    // 1-st pass: find the leaf containing the query point and compute an upper bound
    // on the search distance
    //

    uint32 idx;
    float  dist2 = 1.0e16f;

    // start from the root node
    uint32 node_index = 0;

    // keep track of which leaf we visited
    uint32 first_leaf = 0;

    while (1)
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            // find the closest neighbor in this leaf
            const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
            for (uint32 i = leaf.x; i < leaf.y; ++i)
            {
                const VectorType delta = kd_points[i] - query;
                const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if (dist2 > d2)
                {
                    dist2 = d2;
                    idx   = i;
                }
            }

            // keep track of which leaf we found
            first_leaf = node_index;
            break;
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index = node.get_child_offset() + (comp( query, split_dim ) < split_plane ? 0u : 1u);
        }
    }

    //
    // 2-nd pass: visit the tree with a stack and careful pruning
    //

    int32  stackp = 1;
    float4 stack[64];

    // place a sentinel node in the stack
    stack[0] = make_float4( 0.0f, 0.0f, 0.0f, binary_cast<float>(uint32(-1)) );

    // start from the root node
    node_index = 0;

    float3 cdist = make_float3( 0.0f, 0.0f, 0.0f );

    while (node_index != uint32(-1))
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            if (first_leaf != node_index)
            {
                // find the closest neighbor in this leaf
                const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
                for (uint32 i = leaf.x; i < leaf.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                    if (dist2 > d2)
                    {
                        dist2 = d2;
                        idx   = i;
                    }
                }
            }

            // pop the next node from the stack
            while (stackp > 0)
            {
                const float4 stack_node = stack[ --stackp ];
                node_index = binary_cast<uint32>( stack_node.w );
                cdist      = make_float3( stack_node.x, stack_node.y, stack_node.z );

                if (sq_length( cdist ) < dist2)
                    break;
            }
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            const float split_dist = comp( query, split_dim ) - split_plane;

            const uint32 select = split_dist <= 0.0f ? 0u : 1u;

            node_index = node.get_child_offset() + select;

            // compute the vector distance to the far node
            float3 cdist_far = cdist;
			set_comp( cdist_far, split_dim, split_dist );

            // check whether we should push the far node on the stack
            const float dist_far2 = sq_length( cdist_far );

            if (dist_far2 < dist2)
            {
                stack[ stackp++ ] = make_float4(
                    cdist_far.x,
                    cdist_far.y,
                    cdist_far.z,
                    binary_cast<float>( node.get_child_offset() + 1u - select ) );
            }
        }
    }

    // write the result
    results->index = idx;
    results->dist2 = dist2;
}

struct Compare
{
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE bool operator() (const float2 op1, const float2 op2) const
    {
        return (op1.y == op2.y) ?
            op1.x < op2.x :
            op1.y < op2.y;
    }
};

#if 0
FORCE_INLINE CUGAR_HOST_DEVICE float norm3(const float4 v) { return v.x*v.x + v.y*v.y + v.z*v.z; }

#define KNN_REORDER_STACK(LhsIndex, RhsIndex) \
if (norm3( stack[stackp - (LhsIndex)] ) < \
    norm3( stack[stackp - (RhsIndex)] )) \
    { \
        const float4 tmpv = stack[stackp - (LhsIndex)]; \
        stack[stackp - (LhsIndex)] = stack[stackp - (RhsIndex)]; \
        stack[stackp - (RhsIndex)] = tmpv; \
    }
#else
#define KNN_REORDER_STACK(LhsIndex, RhsIndex)
#endif

template <uint32 K>
struct knn_lookup_traits
{
    const static uint32 M = 8;
};

template <uint32 K, typename VectorType, typename PointIterator>
__device__ void lookup_2d(
    const VectorType        query,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const PointIterator     kd_points,
    Kd_knn<3>::Result*      results)
{
    //
    // 1-st pass: find the smallest node containing the query point and compute an upper bound
    // on the search distance
    //
	typedef vector_view<float2*> queue_vector_type;
	float2 queue_storage[K+1];
	queue_vector_type queue_vector(0,queue_storage);
    priority_queue<float2,queue_vector_type,Compare> queue( queue_vector );
    float max_dist2 = 1.0e16f;

    // start from the root node
    uint32 node_index = 0;

    // keep track of which node we visited
    uint32 entry_subtree = 0;

    while (1)
    {
        const Kd_node node = kd_nodes[ node_index ];
        const uint2 range = kd_ranges[ node_index ];

        if (range.y - range.x < K)
            break;

        entry_subtree = node_index;

        if (node.is_leaf())
            break;
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index = node.get_child_offset() + (comp( query, split_dim ) < split_plane ? 0u : 1u);
        }
    }

    //
    // 2-nd pass: in the entry subtree, found an upper bound on distance
    // looking at the first K neighbors encountered during a traversal
    //

    // find the closest nodes with at least M primitives
    const uint32 M = knn_lookup_traits<K>::M;

    int32  stackp = 1;
    float4 stack[64];

    // place a sentinel node in the stack
    stack[0] = make_float4( 1.0e8f, 1.0e8f, 1.0e8f, binary_cast<float>(uint32(-1)) );

    if (K <= 8)
    {
        // find the closest neighbors in this node
        const uint2 range = kd_ranges[ entry_subtree ];
        for (uint32 i = range.x; i < range.y; ++i)
        {
            const VectorType delta = kd_points[i] - query;
            const float d2 = delta[0]*delta[0] + delta[1]*delta[1];

			// check whether the queue is already full
			if (queue.size() == K && d2 < queue.top().y)
				queue.pop();

			if (queue.size() < K)
				queue.push( make_float2(binary_cast<float>(i), d2) );
        }
        // set the maximum distance bound
        max_dist2 = queue.top().y;
    }
    else
    {
        // start from the entry node
        node_index = entry_subtree;

        max_dist2 = 0.0f;
        uint32 found = 0;

        float2 cdist = make_float2( 0.0f, 0.0f );
        while (node_index != uint32(-1))
        {
            KD_KNN_STATS_ADD( node_tests, 1u );
            const Kd_node node  = kd_nodes[ node_index ];
            const uint2   range = kd_ranges[ node_index ];

            if (node.is_leaf() || range.y - range.x <= M)
            {
                // find the closest neighbors in this node
                KD_KNN_STATS_ADD( leaf_tests, 1u );
                KD_KNN_STATS_ADD( point_tests, range.y - range.x );
                for (uint32 i = range.x; i < range.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1];

                    // reset the maximum distance bound
                    max_dist2 = cugar::max( max_dist2, d2 );
                    if (++found == K)
                        break;
                }
                if (found == K)
                    break;

                // pop the next node from the stack
                while (stackp > 0)
                {
                    KD_KNN_STATS_ADD( node_pops, 1u );
                    const float4 stack_node = stack[ --stackp ];
                    node_index = binary_cast<uint32>( stack_node.w );
                    cdist      = make_float2( stack_node.x, stack_node.y );

                    if (sq_length( cdist ) < max_dist2)
                        break;
                }
            }
            else
            {
                const uint32 split_dim   = node.get_split_dim();
                const float  split_plane = node.get_split_plane();

                const float split_dist = comp( query, split_dim ) - split_plane;

                const uint32 select = split_dist <= 0.0f ? 0u : 1u;

                node_index = node.get_child_offset() + select;

                // compute the vector distance to the far node
                float2 cdist_far = cdist;
				set_comp( cdist_far, split_dim, split_dist );

                // check whether we should push the far node on the stack
                const float dist_far2 = sq_length( cdist_far );

                if (dist_far2 < max_dist2)
                {
                    KD_KNN_STATS_ADD( node_pushes, 1u );

                    #if 0
                    // partially reorder the stack
                    if (stackp >= 1)
                    {
                        const float4 last_cdist = stack[ stackp-1 ];
                        const float last_dist2 = sq_length( last_cdist );

                        const uint32 index =
                            last_dist2 < dist_far2 ? stackp-1 : stackp;

                        if (last_dist2 < dist_far2)
                            stack[ stackp ] = last_cdist;

                        stack[ index ] = make_float4(
                            cdist_far.x,
                            cdist_far.y,
                            0,
                            binary_cast<float>( node.get_child_offset() + 1u - select ) );

                        stackp++;
                    }
                    #else
                    stack[ stackp++ ] = make_float4(
                        cdist_far.x,
                        cdist_far.y,
                        0,
                        binary_cast<float>( node.get_child_offset() + 1u - select ) );

                    if (stackp >= 4)
                    {
                        KNN_REORDER_STACK(4, 3);
                        KNN_REORDER_STACK(2, 1);
                        KNN_REORDER_STACK(4, 2);
                        KNN_REORDER_STACK(3, 1);
                        KNN_REORDER_STACK(3, 2);
                    }
                    #endif
                }
            }
        }

        // remove the excluded node
        entry_subtree = uint32(-1);
    }

    //
    // 3-rd pass: restart traversal from the root node, this time pruning
    // the tree using the computed upper bound on distance
    //

    stackp = 1;

    // start from the root node
    node_index = 0;

    float2 cdist = make_float2( 0.0f, 0.0f );

    while (node_index != uint32(-1))
    {
        KD_KNN_STATS_ADD( node_tests, 1u );
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf() || entry_subtree == node_index)
        {
            if (entry_subtree != node_index)
            {
                // find the closest neighbors in this leaf
                const uint2 range = kd_leaves[ node.get_leaf_index() ];

                KD_KNN_STATS_ADD( leaf_tests, 1u );
                KD_KNN_STATS_ADD( point_tests, range.y - range.x );
                for (uint32 i = range.x; i < range.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1];
                    if (max_dist2 > d2)
                    {
                        KD_KNN_STATS_ADD( point_pushes, 1u );

						// check whether the queue is already full
						if (queue.size() == K && d2 < queue.top().y)
							queue.pop();

						if (queue.size() < K)
							queue.push( make_float2( binary_cast<float>(i), d2 ) );

						// reset the maximum distance bound
						if (queue.size() == K)
							max_dist2 = queue.top().y;
                    }
                }
            }

            // pop the next node from the stack
            while (stackp > 0)
            {
                KD_KNN_STATS_ADD( node_pops, 1u );
                const float4 stack_node = stack[ --stackp ];
                node_index = binary_cast<uint32>( stack_node.w );
                cdist      = make_float2( stack_node.x, stack_node.y );

                if (sq_length( cdist ) < max_dist2)
                    break;
            }
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            const float split_dist = comp( query, split_dim ) - split_plane;

            const uint32 select = split_dist <= 0.0f ? 0u : 1u;

            node_index = node.get_child_offset() + select;

            // compute the vector distance to the far node
            float2 cdist_far = cdist;
			set_comp( cdist_far, split_dim, split_dist );

            // check whether we should push the far node on the stack
            const float dist_far2 = sq_length( cdist_far );

            if (dist_far2 < max_dist2)
            {
                KD_KNN_STATS_ADD( node_pushes, 1u );

                #if 0
                // partially reorder the stack
                if (stackp >= 1)
                {
                    const float4 last_cdist = stack[ stackp-1 ];
                    const float last_dist2 = sq_length( last_cdist );

                    const uint32 index =
                        last_dist2 < dist_far2 ? stackp-1 : stackp;

                    if (last_dist2 < dist_far2)
                        stack[ stackp ] = last_cdist;

                    stack[ index ] = make_float4(
                        cdist_far.x,
                        cdist_far.y,
                        cdist_far.z,
                        binary_cast<float>( node.get_child_offset() + 1u - select ) );

                    stackp++;
                }
                #else
                stack[ stackp++ ] = make_float4(
                    cdist_far.x,
                    cdist_far.y,
                    0,
                    binary_cast<float>( node.get_child_offset() + 1u - select ) );

                if (stackp >= 4)
                {
                    KNN_REORDER_STACK(4, 3);
                    KNN_REORDER_STACK(2, 1);
                    KNN_REORDER_STACK(4, 2);
                    KNN_REORDER_STACK(3, 1);
                    KNN_REORDER_STACK(3, 2);
                }
                #endif
            }
        }
    }

    // write the results
    for (uint32 i = 0; i < K; ++i)
        ((float2*)results)[i] = queue[i];
}

template <uint32 K, typename VectorType, typename PointIterator>
__device__ void lookup_3d(
    const VectorType        query,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const PointIterator     kd_points,
    Kd_knn<3>::Result*      results)
{
    //
    // 1-st pass: find the smallest node containing the query point and compute an upper bound
    // on the search distance
    //
	typedef vector_view<float2*> queue_vector_type;
	float2 queue_storage[K+1];
	queue_vector_type queue_vector(0,queue_storage);
    priority_queue<float2,queue_vector_type,Compare> queue( queue_vector );
    float max_dist2 = 1.0e16f;

    // start from the root node
    uint32 node_index = 0;

    // keep track of which node we visited
    uint32 entry_subtree = 0;

    while (1)
    {
        const Kd_node node = kd_nodes[ node_index ];
        const uint2 range = kd_ranges[ node_index ];

        if (range.y - range.x < K)
            break;

        entry_subtree = node_index;

        if (node.is_leaf())
            break;
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index = node.get_child_offset() + (comp( query, split_dim ) < split_plane ? 0u : 1u);
        }
    }

    //
    // 2-nd pass: in the entry subtree, found an upper bound on distance
    // looking at the first K neighbors encountered during a traversal
    //

    // find the closest nodes with at least M primitives
    const uint32 M = knn_lookup_traits<K>::M;

    int32  stackp = 1;
    float4 stack[64];

    // place a sentinel node in the stack
    stack[0] = make_float4( 1.0e8f, 1.0e8f, 1.0e8f, binary_cast<float>(uint32(-1)) );

    if (K <= 8)
    {
        // find the closest neighbors in this node
        const uint2 range = kd_ranges[ entry_subtree ];
        for (uint32 i = range.x; i < range.y; ++i)
        {
            const VectorType delta = kd_points[i] - query;
            const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];

			// check whether the queue is already full
			if (queue.size() == K && d2 < queue.top().y)
				queue.pop();

			if (queue.size() < K)
				queue.push( make_float2(binary_cast<float>(i), d2) );
        }
        // set the maximum distance bound
        max_dist2 = queue.top().y;
    }
    else
    {
        // start from the entry node
        node_index = entry_subtree;

        max_dist2 = 0.0f;
        uint32 found = 0;

        float3 cdist = make_float3( 0.0f, 0.0f, 0.0f );
        while (node_index != uint32(-1))
        {
            KD_KNN_STATS_ADD( node_tests, 1u );
            const Kd_node node  = kd_nodes[ node_index ];
            const uint2   range = kd_ranges[ node_index ];

            if (node.is_leaf() || range.y - range.x <= M)
            {
                // find the closest neighbors in this node
                KD_KNN_STATS_ADD( leaf_tests, 1u );
                KD_KNN_STATS_ADD( point_tests, range.y - range.x );
                for (uint32 i = range.x; i < range.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];

                    // reset the maximum distance bound
                    max_dist2 = cugar::max( max_dist2, d2 );
                    if (++found == K)
                        break;
                }
                if (found == K)
                    break;

                // pop the next node from the stack
                while (stackp > 0)
                {
                    KD_KNN_STATS_ADD( node_pops, 1u );
                    const float4 stack_node = stack[ --stackp ];
                    node_index = binary_cast<uint32>( stack_node.w );
                    cdist      = make_float3( stack_node.x, stack_node.y, stack_node.z );

                    if (sq_length( cdist ) < max_dist2)
                        break;
                }
            }
            else
            {
                const uint32 split_dim   = node.get_split_dim();
                const float  split_plane = node.get_split_plane();

                const float split_dist = comp( query, split_dim ) - split_plane;

                const uint32 select = split_dist <= 0.0f ? 0u : 1u;

                node_index = node.get_child_offset() + select;

                // compute the vector distance to the far node
                float3 cdist_far = cdist;
				set_comp( cdist_far, split_dim, split_dist );

                // check whether we should push the far node on the stack
                const float dist_far2 = sq_length( cdist_far );

                if (dist_far2 < max_dist2)
                {
                    KD_KNN_STATS_ADD( node_pushes, 1u );

                    #if 0
                    // partially reorder the stack
                    if (stackp >= 1)
                    {
                        const float4 last_cdist = stack[ stackp-1 ];
                        const float last_dist2 = sq_length( last_cdist );

                        const uint32 index =
                            last_dist2 < dist_far2 ? stackp-1 : stackp;

                        if (last_dist2 < dist_far2)
                            stack[ stackp ] = last_cdist;

                        stack[ index ] = make_float4(
                            cdist_far.x,
                            cdist_far.y,
                            cdist_far.z,
                            binary_cast<float>( node.get_child_offset() + 1u - select ) );

                        stackp++;
                    }
                    #else
                    stack[ stackp++ ] = make_float4(
                        cdist_far.x,
                        cdist_far.y,
                        cdist_far.z,
                        binary_cast<float>( node.get_child_offset() + 1u - select ) );

                    if (stackp >= 4)
                    {
                        KNN_REORDER_STACK(4, 3);
                        KNN_REORDER_STACK(2, 1);
                        KNN_REORDER_STACK(4, 2);
                        KNN_REORDER_STACK(3, 1);
                        KNN_REORDER_STACK(3, 2);
                    }
                    #endif
                }
            }
        }

        // remove the excluded node
        entry_subtree = uint32(-1);
    }

    //
    // 3-rd pass: restart traversal from the root node, this time pruning
    // the tree using the computed upper bound on distance
    //

    stackp = 1;

    // start from the root node
    node_index = 0;

    float3 cdist = make_float3( 0.0f, 0.0f, 0.0f );

    while (node_index != uint32(-1))
    {
        KD_KNN_STATS_ADD( node_tests, 1u );
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf() || entry_subtree == node_index)
        {
            if (entry_subtree != node_index)
            {
                // find the closest neighbors in this leaf
                const uint2 range = kd_leaves[ node.get_leaf_index() ];

                KD_KNN_STATS_ADD( leaf_tests, 1u );
                KD_KNN_STATS_ADD( point_tests, range.y - range.x );
                for (uint32 i = range.x; i < range.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                    if (max_dist2 > d2)
                    {
                        KD_KNN_STATS_ADD( point_pushes, 1u );

						// check whether the queue is already full
						if (queue.size() == K && d2 < queue.top().y)
							queue.pop();

						if (queue.size() < K)
							queue.push( make_float2( binary_cast<float>(i), d2 ) );

						// reset the maximum distance bound
						if (queue.size() == K)
							max_dist2 = queue.top().y;
                    }
                }
            }

            // pop the next node from the stack
            while (stackp > 0)
            {
                KD_KNN_STATS_ADD( node_pops, 1u );
                const float4 stack_node = stack[ --stackp ];
                node_index = binary_cast<uint32>( stack_node.w );
                cdist      = make_float3( stack_node.x, stack_node.y, stack_node.z );

                if (sq_length( cdist ) < max_dist2)
                    break;
            }
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            const float split_dist = comp( query, split_dim ) - split_plane;

            const uint32 select = split_dist <= 0.0f ? 0u : 1u;

            node_index = node.get_child_offset() + select;

            // compute the vector distance to the far node
            float3 cdist_far = cdist;
			set_comp( cdist_far, split_dim, split_dist );

            // check whether we should push the far node on the stack
            const float dist_far2 = sq_length( cdist_far );

            if (dist_far2 < max_dist2)
            {
                KD_KNN_STATS_ADD( node_pushes, 1u );

                #if 0
                // partially reorder the stack
                if (stackp >= 1)
                {
                    const float4 last_cdist = stack[ stackp-1 ];
                    const float last_dist2 = sq_length( last_cdist );

                    const uint32 index =
                        last_dist2 < dist_far2 ? stackp-1 : stackp;

                    if (last_dist2 < dist_far2)
                        stack[ stackp ] = last_cdist;

                    stack[ index ] = make_float4(
                        cdist_far.x,
                        cdist_far.y,
                        cdist_far.z,
                        binary_cast<float>( node.get_child_offset() + 1u - select ) );

                    stackp++;
                }
                #else
                stack[ stackp++ ] = make_float4(
                    cdist_far.x,
                    cdist_far.y,
                    cdist_far.z,
                    binary_cast<float>( node.get_child_offset() + 1u - select ) );

                if (stackp >= 4)
                {
                    KNN_REORDER_STACK(4, 3);
                    KNN_REORDER_STACK(2, 1);
                    KNN_REORDER_STACK(4, 2);
                    KNN_REORDER_STACK(3, 1);
                    KNN_REORDER_STACK(3, 2);
                }
                #endif
            }
        }
    }

    // write the results
    for (uint32 i = 0; i < K; ++i)
        ((float2*)results)[i] = queue[i];
}

template <uint32 DIM>
struct lookup_dispatch {};

template <>
struct lookup_dispatch<2>
{
	template <typename VectorType, typename PointIterator>
	__device__ static void lookup(
		const VectorType        query,
		const Kd_node*          kd_nodes,
		const uint2*            kd_ranges,
		const uint2*            kd_leaves,
		const PointIterator     kd_points,
		Kd_knn<2>::Result*      results)
	{
		lookup_2d( query, kd_nodes, kd_ranges, kd_leaves, kd_points, results );
	}

	template <uint32 K, typename VectorType, typename PointIterator>
	__device__ static void lookup(
		const VectorType        query,
		const Kd_node*          kd_nodes,
		const uint2*            kd_ranges,
		const uint2*            kd_leaves,
		const PointIterator     kd_points,
		Kd_knn<2>::Result*      results)
	{
		lookup_2d<K>( query, kd_nodes, kd_ranges, kd_leaves, kd_points, results );
	}
};

template <>
struct lookup_dispatch<3>
{
	template <typename VectorType, typename PointIterator>
	__device__ static void lookup(
		const VectorType        query,
		const Kd_node*          kd_nodes,
		const uint2*            kd_ranges,
		const uint2*            kd_leaves,
		const PointIterator     kd_points,
		Kd_knn<3>::Result*      results)
	{
		lookup_3d( query, kd_nodes, kd_ranges, kd_leaves, kd_points, results );
	}

	template <uint32 K, typename VectorType, typename PointIterator>
	__device__ static void lookup(
		const VectorType        query,
		const Kd_node*          kd_nodes,
		const uint2*            kd_ranges,
		const uint2*            kd_leaves,
		const PointIterator     kd_points,
		Kd_knn<3>::Result*      results)
	{
		lookup_3d<K>( query, kd_nodes, kd_ranges, kd_leaves, kd_points, results );
	}
};

template <uint32 DIM, typename QueryIterator, typename PointIterator>
__global__ void lookup_kernel_1(
    const uint32                    n_points,
    const QueryIterator             points_begin,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Kd_knn<DIM>::Result*            results)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_points;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_points)
            return;

        lookup_dispatch<DIM>::lookup(
            points_begin[ index ],
            kd_nodes,
            kd_ranges,
            kd_leaves,
            kd_points,
            results + index );
    }
}

template <uint32 DIM, uint32 K, typename QueryIterator, typename PointIterator>
__global__ void lookup_kernel(
    const uint32                    n_points,
    const QueryIterator             points_begin,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Kd_knn<DIM>::Result*            results)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_points;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_points)
            return;

        lookup_dispatch<DIM>::lookup<K>(
            points_begin[ index ],
            kd_nodes,
            kd_ranges,
            kd_leaves,
            kd_points,
            results + index*K );
    }
}

} // namespace knn

// perform a k-nn lookup for a set of query points
//
// \param points_begin     beginning of the query point sequence
// \param points_end       end of the query point sequence
// \param kd_nodes         k-d tree nodes
// \param kd_ranges        k-d tree node ranges
// \param kd_leaves        k-d tree leaves
// \param kd_points        k-d tree points
template <uint32 DIM>
template <typename QueryIterator, typename PointIterator>
void Kd_knn<DIM>::run(
    const QueryIterator             points_begin,
    const QueryIterator             points_end,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Result*                         results)
{
    const uint32 n_points = uint32( points_end - points_begin );

    const uint32 BLOCK_SIZE = 128;
    const uint32 max_blocks = (uint32)cuda::max_active_blocks(
        kd_knn::lookup_kernel_1<DIM,QueryIterator,PointIterator>, BLOCK_SIZE, 0);
    const uint32 n_blocks   = cugar::min( max_blocks, uint32(n_points + BLOCK_SIZE-1) / BLOCK_SIZE );

    kd_knn::lookup_kernel_1<DIM> <<<n_blocks,BLOCK_SIZE>>> (
        n_points,
        points_begin,
        kd_nodes,
        kd_ranges,
        kd_leaves,
        kd_points,
        results );

    //cudaDeviceSynchronize();
}

// perform a k-nn lookup for a set of query points
//
// \param points_begin     beginning of the query point sequence
// \param points_end       end of the query point sequence
// \param kd_nodes         k-d tree nodes
// \param kd_ranges        k-d tree node ranges
// \param kd_leaves        k-d tree leaves
// \param kd_points        k-d tree points
template <uint32 DIM>
template <uint32 K, typename QueryIterator, typename PointIterator>
void Kd_knn<DIM>::run(
    const QueryIterator             points_begin,
    const QueryIterator             points_end,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Result*                         results)
{
    const uint32 n_points = uint32( points_end - points_begin );

    const uint32 BLOCK_SIZE = 64;
    const uint32 max_blocks = (uint32)cuda::max_active_blocks(
        kd_knn::lookup_kernel<DIM,K,QueryIterator,PointIterator>, BLOCK_SIZE, 0);
    const uint32 n_blocks   = cugar::min( max_blocks, uint32(n_points + BLOCK_SIZE-1) / BLOCK_SIZE );

    kd_knn::lookup_kernel<DIM,K> <<<n_blocks,BLOCK_SIZE>>> (
        n_points,
        points_begin,
        kd_nodes,
        kd_ranges,
        kd_leaves,
        kd_points,
        results );

    //cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace cugar

