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

/*! \file bvh.h
 *   \brief Entry point to the generic Bounding Volume Hierarchy library.
 */

#pragma once

#include <sandbox/linalg/vector.h>
#include <sandbox/linalg/bbox.h>
#include <vector>
#include <stack>

namespace sandbox {

struct Indexed_leaf_bvh_tag;
struct Compact_bvh_tag;
struct Compact_bvh_3d_tag;

template <typename tag> struct Bvh_node {};

///
/// Bvh node struct
///
template <>
struct Bvh_node<Indexed_leaf_bvh_tag>
{
    typedef Indexed_leaf_bvh_tag Tag;

    typedef uint32 Type;
    const static uint32 kLeaf     = (1u << 31u);
    const static uint32 kInternal = 0x00000000u;
    const static uint32 kInvalid  = uint32(-1);

    /// empty constructor
    ///
    SANDBOX_HOST_DEVICE Bvh_node() {}
    /// full constructor
    ///
    /// \param type         node type
    /// \param index        child/leaf index
    /// \param skip_node    skip node index
    SANDBOX_HOST_DEVICE Bvh_node(const Type type, const uint32 index, const uint32 skip_node);
    /// set node type
    ///
    /// \param type         node type
	SANDBOX_HOST_DEVICE void set_type(const Type type);
    /// set child/leaf index
    ///
    /// \param index        child/leaf index
	SANDBOX_HOST_DEVICE void set_index(const uint32 index);

    /// is a leaf?
    ///
    SANDBOX_HOST_DEVICE bool is_leaf() const { return (m_packed_data & kLeaf) != 0u; }
    /// get child/leaf index
    ///
    SANDBOX_HOST_DEVICE uint32 get_index() const { return m_packed_data & (~kLeaf); }
    /// get leaf index
    ///
    SANDBOX_HOST_DEVICE uint32 get_leaf_index() const { return m_packed_data & (~kLeaf); }

    /// get child count
    ///
    SANDBOX_HOST_DEVICE uint32 get_child_count() const { return 2u; }
    /// get i-th child
    ///
    /// \param i    child index
    SANDBOX_HOST_DEVICE uint32 get_child(const uint32 i) const { return get_index() + i; }

    /// compute packed data
    ///
    /// \param type     node type
    /// \param index    child/leaf index
    static SANDBOX_HOST_DEVICE uint32 packed_data(const Type type, const uint32 index)
    {
	    return (uint32(type) | index);
    }
    /// set node type into a packed data
    ///
    /// \param packed_data  packed data
    /// \param type         node type
    static SANDBOX_HOST_DEVICE void set_type(uint32& packed_data, const Type type)
    {
	    packed_data &= ~kLeaf;
	    packed_data |= uint32(type);
    }
    /// set child/leaf index into a packed data
    ///
    /// \param packed_data  packed data
    /// \param index        child/leaf index
    static SANDBOX_HOST_DEVICE void set_index(uint32& packed_data, const uint32 index)
    {
	    packed_data &= kLeaf;
	    packed_data |= index;
    }
    static SANDBOX_HOST_DEVICE bool   is_leaf(const uint32 packed_data)   { return (packed_data & kLeaf) != 0u; }
    static SANDBOX_HOST_DEVICE uint32 get_index(const uint32 packed_data) { return packed_data & (~kLeaf); }

	uint32	m_packed_data;	// child index
};

///
/// Bvh node struct with topology and bbox
///
template <>
struct Bvh_node<Compact_bvh_tag>
{
    typedef Compact_bvh_tag Tag;
    typedef uint32 Type;
    const static uint32 kLeaf     = (1u << 31u);
    const static uint32 kInternal = 0x00000000u;
    const static uint32 kInvalid  = uint32(-1);

    /// empty constructor
    ///
    SANDBOX_HOST_DEVICE Bvh_node() {}
    /// full constructor
    ///
    /// \param type         node type
    /// \param index        child/leaf index
    /// \param range_size   primitive range size
    SANDBOX_HOST_DEVICE Bvh_node(const Type type, const uint32 index, const uint32 range_size);
    /// set node type
    ///
    /// \param type         node type
	SANDBOX_HOST_DEVICE void set_type(const Type type);
    /// set child/leaf index
    ///
    /// \param index        child/leaf index
	SANDBOX_HOST_DEVICE void set_index(const uint32 index);
    /// set skip node index
    ///
    /// \param range_size    primitive range size
    SANDBOX_HOST_DEVICE void set_range_size(const uint32 range_size) { m_data = range_size; }

    /// is a leaf?
    ///
    SANDBOX_HOST_DEVICE bool is_leaf() const { return (m_packed_data & kLeaf) != 0u; }
    /// get child/leaf index
    ///
    SANDBOX_HOST_DEVICE uint32 get_index() const { return m_packed_data & (~kLeaf); }
    /// get leaf index
    ///
    SANDBOX_HOST_DEVICE uint32 get_leaf_index() const { return m_packed_data & (~kLeaf); }
    /// get primitive range size
    ///
    SANDBOX_HOST_DEVICE uint32 get_range_size() const { return m_data; }

    /// get child count
    ///
    SANDBOX_HOST_DEVICE uint32 get_child_count() const { return 2u; }
    /// get i-th child
    ///
    /// \param i    child index
    SANDBOX_HOST_DEVICE uint32 get_child(const uint32 i) const { return get_index() + i; }

    /// compute packed data
    ///
    /// \param type     node type
    /// \param index    child/leaf index
    static SANDBOX_HOST_DEVICE uint32 packed_data(const Type type, const uint32 index)
    {
	    return (uint32(type) | index);
    }
    /// set node type into a packed data
    ///
    /// \param packed_data  packed data
    /// \param type         node type
    static SANDBOX_HOST_DEVICE void set_type(uint32& packed_data, const Type type)
    {
	    packed_data &= ~kLeaf;
	    packed_data |= uint32(type);
    }
    /// set child/leaf index into a packed data
    ///
    /// \param packed_data  packed data
    /// \param index        child/leaf index
    static SANDBOX_HOST_DEVICE void set_index(uint32& packed_data, const uint32 index)
    {
	    packed_data &= kLeaf;
	    packed_data |= index;
    }
    static SANDBOX_HOST_DEVICE bool   is_leaf(const uint32 packed_data)   { return (packed_data & kLeaf) != 0u; }
    static SANDBOX_HOST_DEVICE uint32 get_index(const uint32 packed_data) { return packed_data & (~kLeaf); }

	uint32	m_packed_data;	// child index
	uint32	m_data;	        // additional data
};

///
/// Bvh node struct with topology and bbox
///
template <>
struct Bvh_node<Compact_bvh_3d_tag> : public Bvh_node<Compact_bvh_tag>
{
    typedef Compact_bvh_3d_tag Tag;

    float3  m_bbox0;        // first bbox bound
    float3  m_bbox1;        // second bbox bound
};

typedef Bvh_node<Indexed_leaf_bvh_tag>  Indexed_leaf_bvh_node;
typedef Bvh_node<Compact_bvh_tag>       Compact_bvh_node;
typedef Bvh_node<Compact_bvh_3d_tag>    Compact_bvh_3d_node;

} // namespace sandbox

#include <sandbox/bvh/bvh3d_inline.h>
