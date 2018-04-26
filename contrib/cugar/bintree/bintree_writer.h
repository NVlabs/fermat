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

/*! \file bintree_writer.h
 *   \brief Defines a simple binary tree context implementation to be used with
 *          the generate_radix_tree() function.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/vector.h>
#include <cugar/bintree/bintree_node.h>
#include <thrust/device_vector.h>

namespace cugar {

///@addtogroup bintree
///@{

/// Binary tree writer context class
///
template <typename node_type, typename node_tag>  // leaf_index_tag | leaf_range_tag
struct Bintree_writer_context {};

/// Binary tree writer context class leaf_index_tag specialization
///
template <typename node_type>
struct Bintree_writer_context<node_type, leaf_index_tag>
{
    /// empty constructor
    ///
    CUGAR_HOST_DEVICE Bintree_writer_context() {}
    /// constructor
    ///
    CUGAR_HOST_DEVICE Bintree_writer_context(
        node_type* nodes,
        uint2*     leaf_ranges   = NULL,
        uint32*    leaf_pointers = NULL,
        uint32*    parents       = NULL,
        uint32*    skip_nodes    = NULL) :
        m_nodes(nodes), m_leaf_ranges(leaf_ranges), m_leaf_pointers(leaf_pointers), m_parents(parents), m_skip_nodes(skip_nodes) {}

    /// write a new node
    ///
    CUGAR_HOST_DEVICE void write_node(const uint32 node, const uint32 parent, bool p1, bool p2, const uint32 offset, const uint32 skip_node, const uint32 level, const uint32 begin, const uint32 end, const uint32 split_index)
    {
        m_nodes[ node ] = Bintree_node<leaf_index_tag>( p1, p2, offset );

        if (m_parents)
            m_parents[ node ] = parent;

        if (m_skip_nodes)
            m_skip_nodes[ node ] = skip_node;
    }
    /// write a new leaf
    ///
    CUGAR_HOST_DEVICE void write_leaf(const uint32 leaf_index, const uint32 node_index, const uint32 begin, const uint32 end)
    {
        if (m_leaf_ranges)
            m_leaf_ranges[ leaf_index ] = make_uint2( begin, end );

        if (m_leaf_pointers)
            m_leaf_pointers[ leaf_index ] = node_index;
    }

    node_type*  m_nodes;        ///< node pointer
    uint2*      m_leaf_ranges;  ///< leaf ranges
    uint32*     m_leaf_pointers;///< leaf pointers
    uint32*     m_parents;      ///< parent pointers
    uint32*     m_skip_nodes;   ///< skip nodes pointer
};

/// Binary tree writer context class leaf_range_tag specialization
///
template <typename node_type>
struct Bintree_writer_context<node_type, leaf_range_tag>
{
    /// empty constructor
    ///
    CUGAR_HOST_DEVICE Bintree_writer_context() {}
    /// constructor
    ///
    CUGAR_HOST_DEVICE Bintree_writer_context(
        node_type* nodes,
        uint2*     leaf_ranges   = NULL,
        uint32*    leaf_pointers = NULL,
        uint32*    parents       = NULL,
        uint32*    skip_nodes    = NULL) :
        m_nodes(nodes), m_leaf_ranges(leaf_ranges), m_leaf_pointers(leaf_pointers), m_parents(parents), m_skip_nodes(skip_nodes) {}


    /// write a new node
    ///
    CUGAR_HOST_DEVICE void write_node(const uint32 node, const uint32 parent, bool p1, bool p2, const uint32 offset, const uint32 skip_node, const uint32 level, const uint32 begin, const uint32 end, const uint32 split_index)
    {
        if (p1 || p2)
            m_nodes[ node ] = Bintree_node<leaf_range_tag>( p1, p2, offset, end - begin );
        else
            m_nodes[ node ] = Bintree_node<leaf_range_tag>( begin, end );

        if (m_parents)
            m_parents[ node ] = parent;

        if (m_skip_nodes)
            m_skip_nodes[ node ] = skip_node;
    }
    /// write a new leaf
    ///
    CUGAR_HOST_DEVICE void write_leaf(const uint32 leaf_index, const uint32 node_index, const uint32 begin, const uint32 end)
    {
        if (m_leaf_ranges)
            m_leaf_ranges[ leaf_index ] = make_uint2( begin, end );

        if (m_leaf_pointers)
            m_leaf_pointers[ leaf_index ] = node_index;
    }

    node_type*  m_nodes;        ///< node pointer
    uint2*      m_leaf_ranges;  ///< leaf ranges
    uint32*     m_leaf_pointers;///< leaf pointers
    uint32*     m_parents;      ///< parent pointers
    uint32*     m_skip_nodes;   ///< skip nodes pointer
};

/// A simple binary tree writer implementation to be used with
/// the generate_radix_tree() function.
///
/// \ref TreeWriterAnchor "Tree Writer"
///
template <
    typename node_type,
    typename system_tag,
    typename node_vector  = vector<system_tag,node_type>,
    typename range_vector = vector<system_tag,uint2>,
    typename index_vector = vector<system_tag,uint32> >  // leaf_index_tag | leaf_range_tag
struct Bintree_writer
{
    typedef typename node_type::node_tag                node_tag;
    typedef Bintree_writer_context<node_type,node_tag>  context_type;

    /// constructor
    ///
    /// \param nodes    nodes to write to
    /// \param leaves   leaves to write to
    Bintree_writer(
        node_vector*     nodes         = NULL,
        range_vector*    leaf_ranges   = NULL,
        index_vector*    leaf_pointers = NULL,
        index_vector*    parents       = NULL,
        index_vector*    skip_nodes    = NULL) :
        m_nodes( nodes ), m_leaf_ranges( m_leaf_ranges ), m_leaf_pointers( NULL ), m_parents( parents ), m_skip_nodes( skip_nodes ) {}

    void set_nodes(node_vector*             nodes)          { m_nodes = nodes; }
    void set_parents(index_vector*          parents)        { m_parents = parents; }
    void set_skip_nodes(index_vector*       skip_nodes)     { m_skip_nodes = skip_nodes; }
    void set_leaf_pointers(index_vector*    leaf_pointers)  { m_leaf_pointers = leaf_pointers; }
    void set_leaf_ranges(range_vector*      leaf_ranges)    { m_leaf_ranges = leaf_ranges; }

    /// reserve space for more nodes
    ///
    /// \param n    nodes to reserve
    void reserve_nodes(const uint32 n)
    {
        if (m_nodes->size() < n) m_nodes->resize(n);
        if (m_parents && m_parents->size() < n) m_parents->resize(n);
        if (m_skip_nodes && m_skip_nodes->size() < n) m_skip_nodes->resize(n);
    }

    /// reserve space for more leaves
    ///
    /// \param n    leaves to reserve
    void reserve_leaves(const uint32 n)
    {
        if (m_leaf_ranges && m_leaf_ranges->size() < n) m_leaf_ranges->resize(n);
        if (m_leaf_pointers && m_leaf_pointers->size() < n) m_leaf_pointers->resize(n);
    }

    /// return a cuda context
    ///
    context_type get_context()
    {
        return context_type(
            m_nodes         ? raw_pointer( *m_nodes )           : NULL,
            m_leaf_ranges   ? raw_pointer( *m_leaf_ranges )     : NULL,
            m_leaf_pointers ? raw_pointer( *m_leaf_pointers )   : NULL,
            m_parents       ? raw_pointer( *m_parents )         : NULL,
            m_skip_nodes    ? raw_pointer( *m_skip_nodes )      : NULL );
    }

    node_vector*    m_nodes;
    range_vector*   m_leaf_ranges;
    index_vector*   m_leaf_pointers;
    index_vector*   m_parents;
    index_vector*   m_skip_nodes;
};

///@}

} // namespace cugar
