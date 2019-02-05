/*
 * Fermat
 *
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <mesh/MeshStorage.h>
#include <vtl.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>

/// A node of the binary UV bvh, supporting up to 4 objects per leaf
///
struct __align__(8) UVBvh_node
{
    typedef uint32_t Type;
    const static uint32_t kLeaf      = (1u << 31u);
    const static uint32_t kInternal  = 0x00000000u;
    const static uint32_t kInvalid   = uint32_t(-1);
    const static uint32_t kIndexMask = ~(7u << 29u);

    /// empty constructor
    ///
    CUGAR_HOST_DEVICE UVBvh_node() {}

    /// leaf constructor
    ///
    /// \param index        child/leaf index
    CUGAR_HOST_DEVICE UVBvh_node(const Type type, const uint32_t size, const uint32_t index, const uint32_t skip_node)
    {
        m_packed_data = (type | ((size-1) << 29) | index);
        m_skip_node   = skip_node;
    }

    /// is a leaf?
    ///
    CUGAR_HOST_DEVICE bool is_leaf() const { return (m_packed_data & kLeaf) != 0u; }

    /// get child/leaf index
    ///
    CUGAR_HOST_DEVICE uint32_t get_index() const { return m_packed_data & kIndexMask; }

    /// get leaf index
    ///
    CUGAR_HOST_DEVICE uint32_t get_leaf_index() const { return m_packed_data & kIndexMask; }

    /// get child count
    ///
    CUGAR_HOST_DEVICE uint32_t get_child_count() const { return 1u + ((m_packed_data >> 29) & 3u); }

    /// get size
    ///
    CUGAR_HOST_DEVICE uint32_t get_size() const { return 1u + ((m_packed_data >> 29) & 3u); }

    /// get i-th child
    ///
    /// \param i    child index
    CUGAR_HOST_DEVICE uint32_t get_child(const uint32_t i) const { return get_index() + i; }

    /// get the skip node
    ///
    CUGAR_HOST_DEVICE uint32_t get_skip_node() const { return m_skip_node; }
    
    uint32_t	m_packed_data;	///< type | size | child index
                                ///< bit 31    :  type
                                ///< bit 29-30 :  size-1
                                ///< bit 0-28  :  index
    uint32_t    m_skip_node;    ///< skip node
};

/// A plain view of a UVBvh
///
struct UVBvhView
{
	FERMAT_HOST_DEVICE
    UVBvhView(const UVBvh_node* _nodes = NULL, const cugar::Bbox2f* _bboxes = NULL, const uint32_t* _index = NULL) :
        nodes(_nodes), bboxes(_bboxes), index(_index) {}

    const UVBvh_node*       nodes;
    const cugar::Bbox2f*	bboxes;
    const uint32_t*         index;
};

struct UVHit
{
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    UVHit() {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    UVHit(const uint32_t _triId, const float _u, const float _v) { triId = _triId; u = _u; v = _v; }

    uint32_t triId;
    float    u;
    float    v;
};

CUGAR_HOST_DEVICE
inline UVHit locate(const UVBvhView& uvbvh, const MeshView& mesh, const uint32_t group_id, const float2 st)
{
    const MeshStorage::texture_triangle* triangle_indices = reinterpret_cast<const MeshStorage::texture_triangle*>(mesh.texture_indices);
    const float2* triangle_data    = reinterpret_cast<const float2*>(mesh.texture_data);

    // start from the corresponding root node
    uint32_t node_index = group_id;

    // traverse until we land on the invalid node
    while (node_index != UVBvh_node::kInvalid)
    {
        // fetch the current node
        const UVBvh_node node = uvbvh.nodes[node_index];

    #define COMBINED_CHILDREN_TEST
    #if defined(COMBINED_CHILDREN_TEST)
        if (!node.is_leaf())
        {
            const uint32_t child_index = node.get_index();

            const float4 child_bbox1 = reinterpret_cast<const float4*>(uvbvh.bboxes)[ child_index ];
            const float4 child_bbox2 = reinterpret_cast<const float4*>(uvbvh.bboxes)[ child_index+1 ];

            if (st.x >= child_bbox1.x && st.x <= child_bbox1.z &&
                st.y >= child_bbox1.y && st.y <= child_bbox1.w)
            {
                node_index = child_index;
                continue;
            }

            if (st.x >= child_bbox2.x && st.x <= child_bbox2.z &&
                st.y >= child_bbox2.y && st.y <= child_bbox2.w)
            {
                node_index = child_index + 1;
                continue;
            }
        }
    #else
        // test the bbox independently of whether this node is internal or a leaf
        const float4 node_bbox = reinterpret_cast<const float4*>(uvbvh.bboxes)[node_index];

        if (st.x < node_bbox.x || st.x > node_bbox.z &&
            st.y < node_bbox.y || st.y > node_bbox.w)
        {
            // jump to the skip node
            node_index = node.get_skip_node();
            continue;
        }

        if (!node.is_leaf())
        {
            // step directly into the child, without any test
            node_index = node.get_index();
            continue;
        }
    #endif
        else
        {
            // perform point-in-triangle tests against all prims in the leaf
            const uint32_t leaf_begin = node.get_leaf_index();
            const uint32_t leaf_end   = leaf_begin + node.get_size();

            for (uint32_t l = leaf_begin; l < leaf_end; ++l)
            {
                // load the triangle
                const uint32_t tri_idx = uvbvh.index[l];
                const MeshStorage::texture_triangle tri = triangle_indices[ tri_idx ];

                // compute the barycentric coordinates of our st point
                const float2 p1 = triangle_data[ tri.x ];
                const float2 p2 = triangle_data[ tri.y ];
                const float2 p0 = triangle_data[ tri.z ];

             #if 0
                const cugar::Vector2d v0 = cugar::Vector2d(cugar::Vector2f(p1)) - cugar::Vector2d(cugar::Vector2f(p0));
                const cugar::Vector2d v1 = cugar::Vector2d(cugar::Vector2f(p2)) - cugar::Vector2d(cugar::Vector2f(p0));
                const cugar::Vector2d v2 = cugar::Vector2d(cugar::Vector2f(st)) - cugar::Vector2d(cugar::Vector2f(p0));

                const double den = v0.x * v1.y - v1.x * v0.y;
                const double inv_den = 1.0f / den;
                const double u = (v2.x * v1.y - v1.x * v2.y) * inv_den;
                const double v = (v0.x * v2.y - v2.x * v0.y) * inv_den;
             #else
                const cugar::Vector2f v0 = cugar::Vector2f(p1) - cugar::Vector2f(p0);
                const cugar::Vector2f v1 = cugar::Vector2f(p2) - cugar::Vector2f(p0);
                const cugar::Vector2f v2 = cugar::Vector2f(st) - cugar::Vector2f(p0);

                const float den = v0.x * v1.y - v1.x * v0.y;
                const float inv_den = 1.0f / den;
                const float u = (v2.x * v1.y - v1.x * v2.y) * inv_den;
                const float v = (v0.x * v2.y - v2.x * v0.y) * inv_den;
             #endif

                // check whether we are inside the triangle
                if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f)
                    return UVHit( tri_idx, float(u), float(v) );
            }
        }

        // jump to the skip node
        node_index = node.get_skip_node();
    }

    // no triangle found
    return UVHit(uint32_t(-1), -1.0f, -1.0f);
}

CUGAR_HOST_DEVICE
inline uint32 locate(const UVBvhView& uvbvh, const VTL* vtls, const uint32_t prim_id, const float2 uv)
{
	// shift the st coordinates into the proper square, determined by the primitive id
	const float2 st = make_float2( uv.x + prim_id, uv.y );

	// start from the corresponding root node
    uint32_t node_index = 0u;

    // traverse until we land on the invalid node
    while (node_index != UVBvh_node::kInvalid)
    {
        // fetch the current node
        const UVBvh_node node = uvbvh.nodes[node_index];

    #define COMBINED_CHILDREN_TEST
    #if defined(COMBINED_CHILDREN_TEST)
        if (!node.is_leaf())
        {
            const uint32_t child_index = node.get_index();

            const float4 child_bbox1 = reinterpret_cast<const float4*>(uvbvh.bboxes)[ child_index ];
            const float4 child_bbox2 = reinterpret_cast<const float4*>(uvbvh.bboxes)[ child_index+1 ];

            if (st.x >= child_bbox1.x && st.x <= child_bbox1.z &&
                st.y >= child_bbox1.y && st.y <= child_bbox1.w)
            {
                node_index = child_index;
                continue;
            }

            if (st.x >= child_bbox2.x && st.x <= child_bbox2.z &&
                st.y >= child_bbox2.y && st.y <= child_bbox2.w)
            {
                node_index = child_index + 1;
                continue;
            }
        }
    #else
        // test the bbox independently of whether this node is internal or a leaf
        const float4 node_bbox = reinterpret_cast<const float4*>(uvbvh.bboxes)[node_index];

        if (st.x < node_bbox.x || st.x > node_bbox.z &&
            st.y < node_bbox.y || st.y > node_bbox.w)
        {
            // jump to the skip node
            node_index = node.get_skip_node();
            continue;
        }

        if (!node.is_leaf())
        {
            // step directly into the child, without any test
            node_index = node.get_index();
            continue;
        }
    #endif
        else
        {
            // perform point-in-triangle tests against all prims in the leaf
            const uint32_t leaf_begin = node.get_leaf_index();
            const uint32_t leaf_end   = leaf_begin + node.get_size();

            for (uint32_t l = leaf_begin; l < leaf_end; ++l)
            {
                // load the triangle
                const uint32_t	vtl_idx = uvbvh.index[l];
                const VTL		vtl     = vtls[ vtl_idx ];

                // compute the barycentric coordinates of our st point
                const float2 p1 = vtl.uv0;
                const float2 p2 = vtl.uv1;
                const float2 p0 = vtl.uv2;

             #if 0
                const cugar::Vector2d v0 = cugar::Vector2d(cugar::Vector2f(p1)) - cugar::Vector2d(cugar::Vector2f(p0));
                const cugar::Vector2d v1 = cugar::Vector2d(cugar::Vector2f(p2)) - cugar::Vector2d(cugar::Vector2f(p0));
                const cugar::Vector2d v2 = cugar::Vector2d(cugar::Vector2f(uv)) - cugar::Vector2d(cugar::Vector2f(p0));

                const double den = v0.x * v1.y - v1.x * v0.y;
                const double inv_den = 1.0f / den;
                const double u = (v2.x * v1.y - v1.x * v2.y) * inv_den;
                const double v = (v0.x * v2.y - v2.x * v0.y) * inv_den;
             #else
                const cugar::Vector2f v0 = cugar::Vector2f(p1) - cugar::Vector2f(p0);
                const cugar::Vector2f v1 = cugar::Vector2f(p2) - cugar::Vector2f(p0);
                const cugar::Vector2f v2 = cugar::Vector2f(uv) - cugar::Vector2f(p0);

                const float den = v0.x * v1.y - v1.x * v0.y;
                const float inv_den = 1.0f / den;
                const float u = (v2.x * v1.y - v1.x * v2.y) * inv_den;
                const float v = (v0.x * v2.y - v2.x * v0.y) * inv_den;
             #endif

                // check whether we are inside the triangle
                if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f)
                    return vtl_idx;
            }
        }

        // jump to the skip node
        node_index = node.get_skip_node();
    }

    // no VTL found
    return uint32(-1);
}
