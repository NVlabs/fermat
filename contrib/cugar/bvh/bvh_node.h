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

/*! \file bvh_node.h
 *   \brief Entry point to the generic Bounding Volume Hierarchy library.
 */

#pragma once

#include <cugar/bintree/bintree_node.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>

namespace cugar {

///@addtogroup bvh
///@{

/// Base bvh topology node class
///
struct Bvh_node : public Bintree_node<leaf_range_tag>
{
    typedef leaf_range_tag node_tag;

    enum internal_type { kInternal = 0 };
    enum leaf_type     { kLeaf     = 1 };

    /// default constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node() {}

    /// internal node constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node(const internal_type type, const uint32 child_index, const uint32 range_size = 0) : Bintree_node<leaf_range_tag>(true, true, child_index, range_size) {}

    /// leaf node constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node(const leaf_type type, const uint32 leaf_begin, const uint32 leaf_end) : Bintree_node<leaf_range_tag>( leaf_begin, leaf_end ) {}

    /// assign the base type
    ///
    CUGAR_HOST_DEVICE
    Bvh_node& operator= (const Bintree_node<leaf_range_tag>& base)
    {
        Bintree_node<leaf_range_tag>::operator=(base);
        return *this;
    }
};

/// 3d bvh node with bounding box (note that this struct has a convenient size of 32 bytes, which can be fetched with two float4 vector loads)
///
CUGAR_ALIGN_BEGIN(16)
struct Bvh_node_3d : public Bvh_node
{
    typedef Bvh_node::node_tag      node_tag;
    typedef Bvh_node::internal_type internal_type;
    typedef Bvh_node::leaf_type     leaf_type;

    /// default constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node_3d() {}

    /// internal node constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node_3d(const internal_type type, const uint32 child_index, const uint32 range_size = 0) : Bvh_node( type, child_index, range_size ) {}

    /// leaf node constructor
    ///
    CUGAR_HOST_DEVICE
    Bvh_node_3d(const leaf_type type, const uint32 leaf_begin, const uint32 leaf_end) : Bvh_node( type, leaf_begin, leaf_end ) {}

	/// construct a node from two float4's
	///
	CUGAR_HOST_DEVICE
	Bvh_node_3d(const float4 f0, const float4 f1)
	{
		Bintree_node<leaf_range_tag>::m_packed_info = cugar::binary_cast<uint32>(f0.x);
		Bintree_node<leaf_range_tag>::m_range_size  = cugar::binary_cast<uint32>(f0.y);
		bbox[0] = Vector3f(f0.z, f0.w, f1.x);
		bbox[1] = Vector3f(f1.y, f1.z, f1.w);
	}
	
	/// assign the base type
    ///
    CUGAR_HOST_DEVICE
    Bvh_node_3d& operator= (const Bintree_node<leaf_range_tag>& base)
    {
        Bintree_node<leaf_range_tag>::operator=(base);
        return *this;
    }

	/// construct a node from two float4's
	///
	CUGAR_HOST_DEVICE
	static Bvh_node_3d load_ldg(const Bvh_node_3d* node)
	{
	#if defined(CUGAR_DEVICE_COMPILATION)
		const float4 f0 = __ldg(reinterpret_cast<const float4*>(node));
		const float4 f1 = __ldg(reinterpret_cast<const float4*>(node) + 1);
		return Bvh_node_3d(f0,f1);
	#else
		return *node;
	#endif
	}
	
	Bbox<Vector3f> bbox;
};
CUGAR_ALIGN_END(16)

///@}  bvh

} // namespace cugar
