/*
 * Copyright (c) 2010-2016, NVIDIA Corporation
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

#include <cugar/bvh/bvh_node.h>
#include <cugar/bintree/bintree_node.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>
#include <cugar/basic/cuda/pointers.h>
#include <vector>
#include <stack>

namespace cugar {

/// \page bvh_page BVH Module
///\par
/// This \ref bvh "module" implements data-structures and functions to store, build and manipulate BVHs.
///
/// - Bvh_node
/// - Bvh_node_3d
/// - Bvh
/// - Bvh_builder
/// - Bvh_sah_builder
/// - cuda::LBVH_builder
///
///\par
/// As an example, consider the following code to create an LBVH tree over a set of points, in parallel, on the device:
/// \code
///
/// #include <cugar/bvh/cuda/lbvh_builder.h>
///
/// thrust::device_vector<Vector3f> points;
/// ... // code to fill the input vector of points
///
/// thrust::device_vector<Bvh_node> bvh_nodes;
/// thrust::device_vector<uint32>   bvh_index;
///
/// cugar::cuda::LBVH_builder<uint64> builder( &bvh_nodes, &bvh_index );
/// builder.build(
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     points.begin(),                             // begin iterator
///     points.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///

///@addtogroup bvh Bounding Volume Hierarchies
///@{

///
/// A low dimensional bvh class
///
template <uint32 DIM>
struct Bvh
{
	typedef Vector<float,DIM>	vector_type;
	typedef Bbox<vector_type>	bbox_type;

	typedef Bvh_node			node_type;

	std::vector<node_type>		m_nodes;
	std::vector<bbox_type>		m_bboxes;
};

///
/// A bvh builder for sets of low dimensional bboxes
///
template <uint32 DIM>
class Bvh_builder
{
public:
	typedef Vector<float,DIM>	vector_type;
	typedef Bbox<vector_type>	bbox_type;

	/// constructor
    ///
	Bvh_builder() : m_max_leaf_size( 4u ) {}

	/// set bvh parameters
    ///
    /// \param max_leaf_size    maximum leaf size
	void set_params(const uint32 max_leaf_size) { m_max_leaf_size = max_leaf_size; }

	/// build
	///
	/// Iterator is supposed to dereference to a Vector<float,DIM>
	///
	/// \param begin			first point
	/// \param end				last point
	/// \param bvh				output bvh
	template <typename Iterator>
	void build(
		const Iterator	begin,
		const Iterator	end,
		Bvh<DIM>*		bvh);

	/// remapped point index
    ///
	uint32 index(const uint32 i) { return m_points[i].m_index; }

private:
	struct Point
	{
		bbox_type	m_bbox;
		uint32		m_index;

        float center(const uint32 dim) const { return (m_bbox[0][dim] + m_bbox[1][dim])*0.5f; }
	};

	struct Node
	{
		uint32		m_begin;
		uint32		m_end;
		uint32		m_node;
		uint32		m_depth;
	};
	typedef std::stack<Node> Node_stack;

	void compute_bbox(
		const uint32		begin,
		const uint32		end,
		bbox_type&			bbox);

	struct Bvh_partitioner;

	uint32				m_max_leaf_size;
	std::vector<Point>	m_points;
};

/// compute SAH cost of a subtree
inline float compute_sah_cost(const Bvh<3>& bvh, uint32 node_index = 0);

/// build skip nodes for a tree
inline void build_skip_nodes(const Bvh_node* nodes, uint32* skip_nodes);

///
/// A utility class to iterate through Bvh_node_3d's as if they were bboxes
///
struct Bvh_node_3d_bbox_iterator
{
    struct Reference
    {
        CUGAR_HOST_DEVICE
        Reference(cugar::Bvh_node_3d* _node) : node(_node) {}

        CUGAR_HOST_DEVICE
        operator cugar::Bbox3f() const
        {
          #if 0
            return ((volatile cugar::Bvh_node_3d*)node)->bbox;
          #elif 0
            // fetch the node's bbox as a float2 and a float4
            const float2 f1 = *(reinterpret_cast<const volatile float2*>( node ) + 1);
            const float4 f2 = *(reinterpret_cast<const volatile float4*>( node ) + 1);
            return cugar::Bbox3f( cugar::Vector3f(f1.x,f1.y,f2.x), cugar::Vector3f(f2.y,f2.z,f2.w) );
          #else
            // fetch the node's bbox as a float2 and a float4
            const float2 f1 = cuda::load<cuda::LOAD_CG>(reinterpret_cast<const float2*>( node ) + 1);
            const float4 f2 = cuda::load<cuda::LOAD_CG>(reinterpret_cast<const float4*>( node ) + 1);
            return cugar::Bbox3f( cugar::Vector3f(f1.x,f1.y,f2.x), cugar::Vector3f(f2.y,f2.z,f2.w) );
          #endif
        }

        CUGAR_HOST_DEVICE
        void operator=(const cugar::Bbox3f& bbox)
        {
          #if 0
            // overwrite the node's bbox only
            ((volatile cugar::Bvh_node_3d*)node)->bbox[0] = bbox[0];
            ((volatile cugar::Bvh_node_3d*)node)->bbox[1] = bbox[1];
          #elif 0
            // store the node's bbox as a float2 and a float4
            const float2 f1 = make_float2( bbox[0].x, bbox[0].y );
            const float4 f2 = make_float4( bbox[0].z, bbox[1].x, bbox[1].y, bbox[1].z );
            *(reinterpret_cast<volatile float2*>( node ) + 1) = f1;
            *(reinterpret_cast<volatile float4*>( node ) + 1) = f2;
          #else
            // store the node's bbox as a float2 and a float4
            const float2 f1 = make_float2( bbox[0].x, bbox[0].y );
            const float4 f2 = make_float4( bbox[0].z, bbox[1].x, bbox[1].y, bbox[1].z );
            cuda::store<cuda::STORE_CG>( reinterpret_cast<float2*>( node ) + 1, f1 );
            cuda::store<cuda::STORE_CG>( reinterpret_cast<float4*>( node ) + 1, f2 );
          #endif
        }

        cugar::Bvh_node_3d* node;
    };

    CUGAR_HOST_DEVICE
    Bvh_node_3d_bbox_iterator(cugar::Bvh_node_3d* _nodes) : nodes(_nodes) {}

    CUGAR_HOST_DEVICE
    Reference operator[] (const uint32 i) const { return Reference(nodes + i); }

    CUGAR_HOST_DEVICE
    Reference operator[] (const uint32 i)       { return Reference(nodes + i); }

    cugar::Bvh_node_3d* nodes;
};

///@}  bvh

} // namespace cugar

#include <cugar/bvh/bvh_inline.h>
