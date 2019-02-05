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

/*! \file bintree_visitor.h
 *   \brief Define binary tree visitors
 */

#pragma once

#include <cugar/basic/types.h>

namespace cugar {

///@addtogroup bintree
///@{

///
/// A binary tree visitor
///
template <typename node_type, typename leaf_type = typename node_type::node_tag>
struct Bintree_visitor {};

///
/// A binary tree visitor
///
template <typename Node_type>
struct Bintree_visitor<Node_type, leaf_index_tag>
{
    typedef Node_type node_type;

    /// constructor
    ///
    Bintree_visitor() :
        m_num_nodes(0), m_num_leaves(0), m_nodes(NULL), m_node_sizes(NULL), m_leaf_pointers(NULL), m_leaf_ranges(NULL), m_parents(NULL), m_skip_nodes(NULL) {}

    void set_node_count(const uint32 num_nodes)  { m_num_nodes = num_nodes; }
    void set_leaf_count(const uint32 num_leaves) { m_num_leaves = num_leaves; }

    void set_nodes(const node_type*      nodes)          { m_nodes = nodes; }
    void set_parents(const uint32*       parents)        { m_parents = parents; }
    void set_skip_nodes(const uint32*    skip_nodes)     { m_skip_nodes = skip_nodes; }
    void set_leaf_pointers(const uint32* leaf_pointers)  { m_leaf_pointers = leaf_pointers; }
    void set_leaf_ranges(const uint2*    leaf_ranges)    { m_leaf_ranges = leaf_ranges; }

    /// get node count
    ///
    CUGAR_HOST_DEVICE uint32 get_node_count() const { return m_num_nodes; }

    /// get leaf count
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_count() const { return m_num_leaves; }

    /// get child count
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_child_count(const uint32 node) const
    {
        return m_nodes[node].get_child_count();
    }

    /// return whether the node is a leaf
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE bool is_leaf(const uint32 node) const
    {
        return m_nodes[node].is_leaf();
    }

    /// get i-th child (among the active ones)
    ///
    /// \param node     node index
    /// \param i        child index
    CUGAR_HOST_DEVICE uint32 get_child(const uint32 node, const uint32 i) const
    {
        return m_nodes[node].get_child(i);
    }

    /// get parent
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_parent(const uint32 node) const
    {
        return m_parents[node];
    }

    /// get skip node
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_skip_node(const uint32 node) const
    {
        return m_skip_nodes[node];
    }

    /// get leaf range
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint2 get_leaf_range(const uint32 node) const
    {
        const uint32 leaf_index = m_nodes[node].get_leaf_index();
        return m_leaf_ranges ?
            m_leaf_ranges[leaf_index] :
            make_uint2(leaf_index,leaf_index+1);
    }

    /// get primitive range size
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_range_size(const uint32 node) const
    {
        return m_node_sizes[node];
    }

    /// return whether it's possible to locate leaf nodes
    ///
    CUGAR_HOST_DEVICE bool has_leaf_pointers() const { return m_leaf_pointers != NULL; }

    /// return the index of the i-th leaf node
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_node(const uint32 i) const
    {
        return m_leaf_pointers[i];
    }

    uint32           m_num_nodes;
    uint32           m_num_leaves;
    const node_type* m_nodes;
    const uint32*    m_node_sizes;
    const uint32*    m_leaf_pointers;
    const uint2*     m_leaf_ranges;
    const uint32*    m_parents;
    const uint32*    m_skip_nodes;
};

///
/// A binary tree visitor
///
template <typename Node_type>
struct Bintree_visitor<Node_type, leaf_range_tag>
{
    typedef Node_type node_type;

    /// constructor
    ///
    Bintree_visitor() :
        m_num_nodes(0), m_num_leaves(0), m_nodes(NULL), m_leaf_pointers(NULL), m_parents(NULL), m_skip_nodes(NULL) {}

    void set_node_count(const uint32 num_nodes)  { m_num_nodes = num_nodes; }
    void set_leaf_count(const uint32 num_leaves) { m_num_leaves = num_leaves; }

    void set_nodes(const node_type*      nodes)          { m_nodes = nodes; }
    void set_parents(const uint32*       parents)        { m_parents = parents; }
    void set_skip_nodes(const uint32*    skip_nodes)     { m_skip_nodes = skip_nodes; }
    void set_leaf_pointers(const uint32* leaf_pointers)  { m_leaf_pointers = leaf_pointers; }

    /// get node count
    ///
    CUGAR_HOST_DEVICE uint32 get_node_count() const { return m_num_nodes; }

    /// get leaf count
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_count() const { return m_num_leaves; }

    /// return whether the node is a leaf
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE bool is_leaf(const uint32 node) const
    {
        return m_nodes[node].is_leaf() ? true : false;
    }

    /// get child count
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_child_count(const uint32 node) const
    {
        return m_nodes[node].get_child_count();
    }

    /// get i-th child (among the active ones)
    ///
    /// \param node    node index
    /// \param i        child index
    CUGAR_HOST_DEVICE uint32 get_child(const uint32 node, const uint32 i) const
    {
        return m_nodes[node].get_child(i);
    }

    /// get parent
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_parent(const uint32 node) const
    {
        return m_parents[node];
    }

    /// get skip node
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_skip_node(const uint32 node) const
    {
        return m_skip_nodes[node];
    }

    /// get leaf range
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint2 get_leaf_range(const uint32 node) const
    {
        return m_nodes[node].get_leaf_range();
    }

    /// get primitive range size
    ///
    /// \param node    node index
    CUGAR_HOST_DEVICE uint32 get_range_size(const uint32 node) const
    {
        return m_nodes[node].get_range_size();
    }

    /// return whether it's possible to locate leaf nodes
    ///
    CUGAR_HOST_DEVICE bool has_leaf_pointers() const { return m_leaf_pointers != NULL; }

    /// return the index of the i-th leaf node
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_node(const uint32 i) const
    {
        return m_leaf_pointers[i];
    }

    uint32           m_num_nodes;
    uint32           m_num_leaves;
    const node_type* m_nodes;
    const uint32*    m_leaf_pointers;
    const uint32*    m_parents;
    const uint32*    m_skip_nodes;
};

template <typename bvh_visitor_type>
void check_tree_rec(const uint32 node_id, const uint32 parent_id, const bvh_visitor_type& visitor, const uint32 n_prims, const uint32 max_leaf_size)
{
    // check the parent pointer
    if (parent_id != visitor.get_parent( node_id ))
        throw cugar::logic_error("node[%u] has wrong parent: %u != %u", node_id, parent_id, visitor.get_parent( node_id ));

    if (visitor.is_leaf( node_id ) == false)
    {
        // check the number of children
        const uint32 child_count = visitor.get_child_count( node_id );
        if (child_count == 0 || child_count > 2)
           throw cugar::logic_error("node[%u] has %u children", node_id, child_count);

        // check the children
        for (uint32 i = 0; i < 2; ++i)
        {
            const uint32 child_id = visitor.get_child( node_id, i );
            if (child_id >= visitor.get_node_count())
                throw cugar::logic_error("node[%u].child(%u) out of bounds : %u / %u", node_id, i, child_id, visitor.get_node_count());

            check_tree_rec( child_id, node_id, visitor, n_prims, max_leaf_size );
        }
    }
    else
    {
        // check the leaf range
        const uint2 leaf_range = visitor.get_leaf_range( node_id );

        // check for malformed ranges
        if (leaf_range.x > leaf_range.y)
            throw cugar::logic_error("leaf[%u] : malformed range (%u, %u)", node_id, leaf_range.x, leaf_range.y);

        // check for out-of-bounds primitive indices
        if (leaf_range.y > n_prims)
            throw cugar::logic_error("leaf[%u] : range out of bounds (%u, %u) / %u", node_id, leaf_range.x, leaf_range.y, n_prims);

        // check for out-of-bounds primitive indices
        if (leaf_range.y - leaf_range.x > max_leaf_size)
            throw cugar::logic_error("leaf[%u] : maximum size overflow (%u, %u) / %u", node_id, leaf_range.x, leaf_range.y, max_leaf_size);
    }
}

/// A function to check a host binary tree
///
template <typename node_type, typename leaf_type>
void check_tree(const Bintree_visitor<node_type,leaf_type>& visitor, const uint32 n_prims, const uint32 max_leaf_size = uint32(-1))
{
    check_tree_rec( 0u, uint32(-1), visitor, n_prims, max_leaf_size );
}

///@} bintree

} // namespace cugar
