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

#include <cugar/bits/morton.h>
#include <cugar/basic/cuda/sort.h>
#include <cugar/basic/utils.h>
#include <cugar/radixtree/cuda/radixtree.h>

namespace cugar {
namespace cuda {

namespace kd {

    template <typename integer, uint32 DIM>
    struct Morton_bits {};

    template <>
    struct Morton_bits<uint32, 2u>
	{
		static const uint32 value = 32u;
	
		CUGAR_HOST_DEVICE static inline float convert(float a, float b, const uint32 i)
		{
			const float x = float(i) / float(1u << 16u);
			return a + (b - a) * x;
		}
	};

    template <>
    struct Morton_bits<uint64, 2u>
	{
		static const uint32 value = 64u;

		CUGAR_HOST_DEVICE static inline float convert(float a, float b, const uint64 i)
		{
			const float x = float(i) / float(0xFFFFFFFFu);
			return a + (b - a) * x;
		}
	};

	template <>
    struct Morton_bits<uint32, 3u>
	{
		static const uint32 value = 30u;

		CUGAR_HOST_DEVICE static inline float convert(float a, float b, const uint64 i)
		{
			const float x = float(i) / float(1u << 10u);
			return a + (b - a) * x;
		}
	};

    template <>
    struct Morton_bits<uint64, 3u>
	{
		static const uint32 value = 60u;

		CUGAR_HOST_DEVICE static inline float convert(float a, float b, const uint64 i)
		{
			const float x = float(i) / float(1u << 20u);
			return a + (b - a) * x;
		}
	};

/// A simple binary tree context implementation to be used with
/// the Bvh generate() function.
template <uint32 DIM, typename BboxType, typename Integer, typename OutputTree>
struct Kd_context
{
    typedef typename OutputTree::Context BaseContext;

    /// Cuda accessor struct
    struct Context
    {
        CUGAR_HOST_DEVICE Context() {}
        CUGAR_HOST_DEVICE Context(const BaseContext context, const Integer* codes, BboxType bbox) :
            m_context( context ), m_codes( codes ), m_bbox( bbox ) {}

		/// write a new node
		///
		CUGAR_HOST_DEVICE void write_node(const uint32 node, const uint32 parent, bool p1, bool p2, const uint32 offset, const uint32 skip_node, const uint32 level, const uint32 begin, const uint32 end, const uint32 split_index)
		{
			//if (m_parents)
			//	m_parents[node] = parent;

			if (p1)
            {
                // fetch the Morton code corresponding to the split plane
                      Integer code = m_codes[ split_index ];
                const uint32  split_dim = level % DIM;

                // extract the selected coordinate
                Integer split_coord = 0;

				if (level)
				{
					code >>= level-1;
					code <<= level-1;
				}

                for (int i = 0; code; i++)
                {
	                split_coord |= (((code >> split_dim) & 1u) << i);
                    code >>= DIM;
                }

				// convert to floating point
                const float split_plane = Morton_bits<Integer,DIM>::convert( m_bbox[0][split_dim], m_bbox[1][split_dim], split_coord );

                // and output the split node
                m_context.write_node(
                    node,
                    offset,
                    skip_node,
                    begin,
                    end,
                    split_index,
                    split_dim,
                    split_plane );
            }
            else
            {
                // output a leaf node
                m_context.write_node(
                    node,
                    offset,
                    skip_node,
                    begin,
                    end );
            }
        }
		/// write a new leaf
		///
		CUGAR_HOST_DEVICE void write_leaf(const uint32 leaf_index, const uint32 node_index, const uint32 begin, const uint32 end)
		{
            m_context.write_leaf( leaf_index, begin, end );
        }

        BaseContext     m_context;
        const Integer*  m_codes;
        BboxType        m_bbox;
    };

    /// constructor
    Kd_context(
        OutputTree			context,
        const Integer*		codes,
        BboxType            bbox) :
        m_context( context ), m_codes( codes ), m_bbox( bbox ) {}

    /// reserve space for more nodes
    void reserve_nodes(const uint32 n) { m_context.reserve_nodes(n); }

    /// reserve space for more leaves
    void reserve_leaves(const uint32 n) { m_context.reserve_leaves(n); }

    /// return a cuda context
    Context get_context()
    {
        return Context(
            m_context.get_context(),
            m_codes,
            m_bbox );
    }

    OutputTree			m_context;
    const Integer*		m_codes;
    BboxType			m_bbox;
};

// a small kernel to calculate Morton codes
template <typename PointIterator, typename Integer, typename MortonFunctor>
__global__
void morton_kernel(const uint32 n_points, const PointIterator points_begin, Integer* out, const MortonFunctor morton)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < n_points)
	{
		typedef typename std::iterator_traits<PointIterator>::value_type VectorType;

		const VectorType p = points_begin[thread_id];
		out[thread_id] = morton(p);
	}
}

}; // namespace kd

// build a k-d tree given a set of points
template <typename Integer>
template <typename OutputTree, typename Iterator, typename BboxType>
void Kd_builder<Integer>::build(
    OutputTree&                     tree,
	vector<device_tag, uint32>&		index,
    const BboxType                  bbox,
    const Iterator                  points_begin,
    const Iterator                  points_end,
    const uint32                    max_leaf_size)
{
	const uint32 DIM = BboxType::vector_type::DIMENSION;
    const uint32 n_points = uint32( points_end - points_begin );

    need_space( m_codes, n_points );
    need_space( index, n_points );
	need_space( m_temp_codes, n_points );
	need_space( m_temp_index, n_points );

    // compute the Morton code for each point
  #if 1
	{
		const uint32 blockSize = (uint32)cugar::cuda::max_blocksize_with_highest_occupancy(kd::morton_kernel< Iterator,Integer,morton_functor<Integer,DIM> >, 0u);
		const dim3 gridSize(cugar::divide_ri(n_points, blockSize));
		kd::morton_kernel<<< gridSize, blockSize >>> (n_points, points_begin, raw_pointer(m_codes), morton_functor<Integer,DIM>( bbox ));
	}
	//cuda::sync_and_check_error("morton codes");
  #else
    thrust::transform(
        points_begin,
        points_begin + n_points,
        m_codes.begin(),
        morton_functor<Integer,DIM>( bbox ) );
  #endif

    // setup the point indices, from 0 to n_points-1
    thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        index.begin() );

	//cuda::sync_and_check_error("copy");

	if (n_points > 1)
	{
		// sort the indices by Morton code
		SortBuffers<Integer*, uint32*> sort_buffers;
		sort_buffers.keys[0] = raw_pointer(m_codes);
		sort_buffers.keys[1] = raw_pointer(m_temp_codes);
		sort_buffers.values[0] = raw_pointer(index);
		sort_buffers.values[1] = raw_pointer(m_temp_index);

		SortEnactor sort_enactor;
		sort_enactor.sort(n_points, sort_buffers);

		// check whether we need to copy the sort results back in place
		if (sort_buffers.selector)
		{
			thrust::copy(m_temp_codes.begin(), m_temp_codes.begin() + n_points, m_codes.begin());
			thrust::copy(m_temp_index.begin(), m_temp_index.begin() + n_points, index.begin());
		}
	}

    // generate a kd-tree
    kd::Kd_context<DIM,BboxType,Integer,OutputTree> bintree_context( tree, thrust::raw_pointer_cast( &m_codes.front() ), bbox );

    const uint32 bits = kd::Morton_bits<Integer,DIM>::value;

	generate_radix_tree(
		m_kd_context,
		n_points,
		raw_pointer(m_codes),
		bits,
		max_leaf_size,
		false,
		false,
        bintree_context );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;
}

} // namespace cuda
} // namespace cugar
