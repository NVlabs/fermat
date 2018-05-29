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

/*! \file bintree_node.h
 *   \brief Define CUDA based scan primitives.
 */

#pragma once

#include <cugar/basic/types.h>

namespace cugar {

/// \page bintree_page Binary Trees Module
///
/// This \ref bintree "module" implements binary tree concepts, such as node representations and visitor patterns.
///
/// - Bintree_node
/// - Bintree_visitor
/// - Bintree_writer
///
/// It also contains a submodule for the construction of radix trees: \subpage radixtree_page
///

///@defgroup bintree Binary Trees
/// This module defines generic binary tree concepts and objects, such as node representations
/// and visitor patterns.
///@{

/// tag struct to identify leaves specifying a single index
///
struct leaf_index_tag {};

/// tag struct to identify leaves specifying a full index range
///
struct leaf_range_tag {};

template <typename leaf_type_tag>
struct Bintree_node {};

///
/// A "slim" binary tree node, with a single primitive index per leaf.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, it can either
/// have one or two children: for example, it can have one
/// if a set of points is concentrated in one half-space.
///
template <>
struct Bintree_node<leaf_index_tag>
{
    typedef leaf_index_tag node_tag;

    static const uint32 kInvalid = uint32(-1);

    /// empty constructor
    ///
    CUGAR_HOST_DEVICE Bintree_node() {}

    /// full constructor
    ///
    /// \param child0   first child activation predicate
    /// \param child1   second child activation predicate
    /// \param index    child index
    CUGAR_HOST_DEVICE Bintree_node(bool child0, bool child1, uint32 index) :
        m_packed_info( (child0 ? 1u : 0u) | (child1 ? 2u : 0u) | (index << 2) ) {}

    /// is a leaf?
    ///
    CUGAR_HOST_DEVICE uint32 is_leaf() const
    {
        return (m_packed_info & 3u) == 0u;
    }
    /// get offset of the first child
    ///
    CUGAR_HOST_DEVICE uint32 get_child_index() const
    {
        return m_packed_info >> 2u;
    }
    /// get leaf index
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_index() const
    {
        return m_packed_info >> 2u;
    }
    /// get child count
    ///
    CUGAR_HOST_DEVICE uint32 get_child_count() const
    {
        return ((m_packed_info & 1u) ? 1u : 0u) +
               ((m_packed_info & 2u) ? 1u : 0u);
    }
    /// get i-th child (among the active ones)
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE uint32 get_child(const uint32 i) const
    {
        return get_child_index() + i;
    }
    /// is the i-th child active?
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE bool has_child(const uint32 i) const
    {
        return m_packed_info & (1u << i) ? true : false;
    }
    /// get left partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_left() const
    {
        return has_child(0) ? get_child_index() : kInvalid;
    }
    /// get right partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_right() const
    {
        return has_child(1) ? get_child_index() + (has_child(0) ? 1u : 0u) : kInvalid;
    }

    uint32 m_packed_info;
};

///
/// A "fat" binary tree node, specifying a whole primitive range per leaf.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, it can either
/// have one or two children. The ability for an internal node
/// to have one children is useful to represent singletons in
/// a radix trie.
///
template <>
struct CUGAR_ALIGN_BEGIN(8) Bintree_node<leaf_range_tag>
{
    typedef leaf_range_tag node_tag;

    static const uint32 kInvalid = uint32(-1);

    /// empty constructor
    ///
    CUGAR_HOST_DEVICE Bintree_node() {}

    /// full constructor
    ///
    /// \param child0   first child activation predicate
    /// \param child1   second child activation predicate
    /// \param index    child index
    CUGAR_HOST_DEVICE Bintree_node(bool child0, bool child1, uint32 index, const uint32 range_size = 0) :
        m_packed_info( (child0 ? 1u : 0u) | (child1 ? 2u : 0u) | (index << 2) ), m_range_size(range_size) {}

    /// leaf constructor
    ///
    /// \param index    leaf range begin
    /// \param index    leaf range end
    CUGAR_HOST_DEVICE Bintree_node(const uint32 leaf_begin, const uint32 leaf_end) :
        m_packed_info( (leaf_begin << 2) ), m_range_size( leaf_end - leaf_begin ) {}

    /// is a leaf?
    ///
    CUGAR_HOST_DEVICE uint32 is_leaf() const
    {
        return (m_packed_info & 3u) == 0u;
    }
    /// get offset of the first child
    ///
    CUGAR_HOST_DEVICE uint32 get_child_index() const
    {
        return m_packed_info >> 2u;
    }
    /// get leaf offset
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_begin() const
    {
        return m_packed_info >> 2u;
    }
	/// get range size
    ///
    CUGAR_HOST_DEVICE uint32 get_range_size() const
    {
        return m_range_size;
    }
    /// get leaf range
    ///
    CUGAR_HOST_DEVICE uint2 get_leaf_range() const
    {
        const uint32 leaf_begin = m_packed_info >> 2u;
        return make_uint2( leaf_begin, leaf_begin + m_range_size );
    }
    /// get leaf size
    ///
    CUGAR_HOST_DEVICE uint32 get_leaf_size() const
    {
        return m_range_size;
    }
    /// get child count
    ///
    CUGAR_HOST_DEVICE uint32 get_child_count() const
    {
        return ((m_packed_info & 1u) ? 1u : 0u) +
               ((m_packed_info & 2u) ? 1u : 0u);
    }
    /// get i-th child (among the active ones)
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE uint32 get_child(const uint32 i) const
    {
        return get_child_index() + i;
    }
    /// is the i-th child active?
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE bool has_child(const uint32 i) const
    {
        return m_packed_info & (1u << i) ? true : false;
    }
    /// get left partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_left() const
    {
        return has_child(0) ? get_child_index() : kInvalid;
    }
    /// get right partition (or kInvalid if not active)
    ///
    CUGAR_HOST_DEVICE uint32 get_right() const
    {
        return has_child(1) ? get_child_index() + (has_child(0) ? 1u : 0u) : kInvalid;
    }

    uint32 m_packed_info;
    uint32 m_range_size;
};
CUGAR_ALIGN_END(8)

///@} bintree

} // namespace cugar
