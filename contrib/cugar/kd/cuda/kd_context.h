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

/*! \file kd_context.h
 *   \brief Define basic k-d tree context structure.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/vector.h>

namespace cugar {
namespace cuda {

/*! \addtogroup kdtree k-d Trees
 *  \{
 */

///
/// A context to pass to k-d tree builders
///
struct Kd_context
{
    /// Cuda accessor struct
    struct Context
    {
        CUGAR_HOST_DEVICE Context() {}
        CUGAR_HOST_DEVICE Context(Kd_node* nodes, uint2* leaves, uint2* ranges) :
            m_nodes(nodes), m_leaves(leaves), m_ranges( ranges ) {}

        /// write a new node
        CUGAR_HOST_DEVICE void write_node(const uint32 node, const uint32 offset, const uint32 skip_node, const uint32 begin, const uint32 end, const uint32 split_index, const uint32 split_dim, const float split_plane)
        {
            m_nodes[ node ] = Kd_node( split_dim, split_plane, offset );

            if (m_ranges)
                m_ranges[ node ] = make_uint2( begin, end );
        }
        /// write a new node
        CUGAR_HOST_DEVICE void write_node(const uint32 node, const uint32 offset, const uint32 skip_node, const uint32 begin, const uint32 end)
        {
            m_nodes[ node ] = Kd_node( offset );

            if (m_ranges)
                m_ranges[ node ] = make_uint2( begin, end );
        }
        /// write a new leaf
        CUGAR_HOST_DEVICE void write_leaf(const uint32 index, const uint32 begin, const uint32 end)
        {
            m_leaves[ index ] = make_uint2( begin, end );
        }

        Kd_node*        m_nodes;    ///< node pointer
        uint2*          m_leaves;   ///< leaf pointer
        uint2*          m_ranges;   ///< range pointer
    };

    /// constructor
    ///
    /// \param nodes            output node vector
    /// \param leaves           output leaf vector
    /// \param ranges           output node range vector: if not NULL,
    ///                         the i-th item will be set with the range
    ///                         of prims corresponding to the i-th node.
    Kd_context(
        thrust::device_vector<Kd_node>* nodes,
        thrust::device_vector<uint2>*   leaves,
        thrust::device_vector<uint2>*   ranges) :
        m_nodes( nodes ), m_leaves( leaves ), m_ranges( ranges ) {}

    /// reserve space for more nodes
    ///
    void reserve_nodes(const uint32 n)
    {
        if (m_nodes->size() < n) m_nodes->resize(n);
        if (m_ranges && m_ranges->size() < n) m_ranges->resize(n);
    }

    /// reserve space for more leaves
    ///
    void reserve_leaves(const uint32 n) { if (m_leaves->size() < n) m_leaves->resize(n); }

    /// return a cuda context
    ///
    Context get_context()
    {
        return Context(
            thrust::raw_pointer_cast( &m_nodes->front() ),
            thrust::raw_pointer_cast( &m_leaves->front() ),
            m_ranges ? thrust::raw_pointer_cast( &m_ranges->front() ) : NULL );
    }

    thrust::device_vector<Kd_node>*  m_nodes;
    thrust::device_vector<uint2>*    m_leaves;
    thrust::device_vector<uint2>*    m_ranges;
};

/*! \}
 */

} // namespace cuda
} // namespace cugar
