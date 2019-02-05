/*
 * Fermat
 *
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

// ------------------------------------------------------------------------- //
//
// Declaration of classes used to represent and manipulate paths.
//
// ------------------------------------------------------------------------- //

#include <vertex.h>
#include <renderer.h>

///@addtogroup Fermat
///@{

///@defgroup PathModule
///\par
/// This module defines the basic classes used to represent and manipulate light paths throughout Fermat,
/// particularly:
///\par
/// - Path : represent a light path in compact form, storing its vertex ids only
/// - BidirPath : represent a bidirectional path in compact form, storing its vertex ids only
///@{

///
/// A class to represent light paths in compact form, storing only the vertex ids.
///
struct Path
{
	VertexGeometryId*	vertex_ids;
	uint32				n_vertices;
	uint32				stride;

	FERMAT_HOST_DEVICE
	Path() {}

	FERMAT_HOST_DEVICE
	Path(const uint32 _n_vertices, VertexGeometryId* _verts, const uint32 _stride) :
		vertex_ids(_verts),
		n_vertices(_n_vertices),
		stride(_stride)
	{}

	FERMAT_HOST_DEVICE
	const VertexGeometryId& v_L(const uint32 i) const
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertex_ids[i * stride];
	}

	FERMAT_HOST_DEVICE
	const VertexGeometryId& v_E(const uint32 i) const
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertex_ids[(n_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	VertexGeometryId& v_L(const uint32 i)
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertex_ids[i * stride];
	}

	FERMAT_HOST_DEVICE
	VertexGeometryId& v_E(const uint32 i)
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertex_ids[(n_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	float G(const uint32 i, const RenderingContextView& renderer) const
	{
		VertexGeometry v;
		VertexGeometry v_next;

		setup_differential_geometry(renderer.mesh, v_L(i), &v);
		setup_differential_geometry(renderer.mesh, v_L(i+1), &v_next);

		cugar::Vector3f out = v_next.position - v.position;

		const float d2 = cugar::max( cugar::square_length(out), 1.0e-8f);

		out /= sqrtf( d2 );

		return fabsf(cugar::dot(v.normal_s, out) * cugar::dot(v_next.normal_s, out)) / d2;
	}

	FERMAT_HOST_DEVICE
	cugar::Vector3f edge_L(const uint32 i, const RenderingContextView& renderer) const
	{
		return
			interpolate_position(renderer.mesh, v_L(i + 1)) -
			interpolate_position(renderer.mesh, v_L(i));
	}

	FERMAT_HOST_DEVICE
	cugar::Vector3f edge_E(const uint32 i, const RenderingContextView& renderer) const
	{
		return
			interpolate_position(renderer.mesh, v_E(i + 1)) -
			interpolate_position(renderer.mesh, v_E(i));
	}
};


///
/// A class to represent light paths in compact form, storing only the vertex ids.
///
struct BidirPath
{
	VertexGeometryId*	l_vertex_ids;
	VertexGeometryId*	e_vertex_ids;
	uint32				l_vertices;
	uint32				e_vertices;
	uint32				n_vertices;
	uint32				stride;

	FERMAT_HOST_DEVICE
	BidirPath() {}

	FERMAT_HOST_DEVICE
	BidirPath(const uint32 _l_vertices, const uint32 _e_vertices, VertexGeometryId* _l_verts, VertexGeometryId* _e_verts, const uint32 _stride) :
		l_vertex_ids(_l_verts),
		e_vertex_ids(_e_verts),
		l_vertices(_l_vertices),
		e_vertices(_e_vertices),
		n_vertices(_l_vertices + _e_vertices),
		stride(_stride)
	{}

	FERMAT_HOST_DEVICE
	const VertexGeometryId& v_L(const uint32 i) const
	{
		FERMAT_ASSERT(i < l_vertices + e_vertices);
		return i < l_vertices ?
			l_vertex_ids[i * stride] :
			e_vertex_ids[(l_vertices + e_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	const VertexGeometryId& v_E(const uint32 i) const
	{
		FERMAT_ASSERT(i < l_vertices + e_vertices);
		return i < e_vertices ?
			e_vertex_ids[i * stride] :
			l_vertex_ids[(l_vertices + e_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	VertexGeometryId& v_L(const uint32 i)
	{
		FERMAT_ASSERT(i < l_vertices + e_vertices);
		return i < l_vertices ?
			l_vertex_ids[i * stride] :
			e_vertex_ids[(l_vertices + e_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	VertexGeometryId& v_E(const uint32 i)
	{
		FERMAT_ASSERT(i < l_vertices + e_vertices);
		return i < e_vertices ?
			e_vertex_ids[i * stride] :
			l_vertex_ids[(l_vertices + e_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	float G(const uint32 i, const RenderingContextView& renderer) const
	{
		VertexGeometry v;
		VertexGeometry v_next;

		setup_differential_geometry(renderer.mesh, v_L(i), &v);
		setup_differential_geometry(renderer.mesh, v_L(i+1), &v_next);

		cugar::Vector3f out = v_next.position - v.position;

		const float d2 = cugar::max( cugar::square_length(out), 1.0e-8f);

		out /= sqrtf( d2 );

		return fabsf(cugar::dot(v.normal_s, out) * cugar::dot(v_next.normal_s, out)) / d2;
	}

	FERMAT_HOST_DEVICE
	cugar::Vector3f edge_L(const uint32 i, const RenderingContextView& renderer) const
	{
		return
			interpolate_position(renderer.mesh, v_L(i + 1)) -
			interpolate_position(renderer.mesh, v_L(i));
	}

	FERMAT_HOST_DEVICE
	cugar::Vector3f edge_E(const uint32 i, const RenderingContextView& renderer) const
	{
		return
			interpolate_position(renderer.mesh, v_E(i + 1)) -
			interpolate_position(renderer.mesh, v_E(i));
	}
};


///
/// A class to represent a cached quantity associated with light paths
///
template <typename T>
struct PathCache
{
	T*			vertices;
	uint32		n_vertices;
	uint32		stride;

	FERMAT_HOST_DEVICE
	PathCache() {}

	FERMAT_HOST_DEVICE
	PathCache(const uint32 _n_vertices, T* _verts, const uint32 _stride) :
		vertices(_verts),
		n_vertices(_n_vertices),
		stride(_stride)
	{}

	FERMAT_HOST_DEVICE
	const T& v_L(const uint32 i) const
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertices[i * stride];
	}

	FERMAT_HOST_DEVICE
	const T& v_E(const uint32 i) const
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertices[(n_vertices - i - 1) * stride];
	}

	FERMAT_HOST_DEVICE
	T& v_L(const uint32 i)
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertices[i * stride];
	}

	FERMAT_HOST_DEVICE
	T& v_E(const uint32 i)
	{
		FERMAT_ASSERT(i < n_vertices);
		return vertices[(n_vertices - i - 1) * stride];
	}
};

///@} PathModule
///@} Fermat
