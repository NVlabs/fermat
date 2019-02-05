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

#include <types.h>
#include <ray.h>
#include <vertex.h>
#include <mesh_utils.h>

///@addtogroup Fermat
///@{

///@addtogroup LightsModule
///@{

/// Represent a Virtual Triangular Light
///
struct VTL
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	VTL() {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	VTL(const uint32 _prim_id, const float2 _uv0, const float2 _uv1, const float2 _uv2, const float _area) 
	{
		prim_id	= _prim_id;
		area	= _area;
		uv0		= _uv2;
		uv1		= _uv1;
		uv2		= _uv0;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void interpolate_positions(MeshView mesh, cugar::Vector3f& p0, cugar::Vector3f& p1, cugar::Vector3f& p2) const
	{
		// fetch the original triangle
		const MeshStorage::vertex_triangle tri = reinterpret_cast<const MeshStorage::vertex_triangle*>(mesh.vertex_indices)[prim_id];
		const cugar::Vector3f vtp0 = load_vertex(mesh, tri.x);
		const cugar::Vector3f vtp1 = load_vertex(mesh, tri.y);
		const cugar::Vector3f vtp2 = load_vertex(mesh, tri.z);

		// interpolate the VTL
		p0 = vtp2 * (1.0f - uv0.x - uv0.y) + vtp0 * uv0.x + vtp1 * uv0.y;
		p1 = vtp2 * (1.0f - uv1.x - uv1.y) + vtp0 * uv1.x + vtp1 * uv1.y;
		p2 = vtp2 * (1.0f - uv2.x - uv2.y) + vtp0 * uv2.x + vtp1 * uv2.y;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void interpolate_tex_coords(MeshView mesh, cugar::Vector2f& t0, cugar::Vector2f& t1, cugar::Vector2f& t2) const
	{
		// estimate the triangle area in texture space
		const MeshStorage::texture_triangle tri = reinterpret_cast<const MeshStorage::texture_triangle*>(mesh.texture_indices)[prim_id];
		const cugar::Vector2f vtt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.x] : cugar::Vector2f(1.0f, 0.0f);
		const cugar::Vector2f vtt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.y] : cugar::Vector2f(0.0f, 1.0f);
		const cugar::Vector2f vtt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.z] : cugar::Vector2f(0.0f, 0.0f);

		// interpolate the VTL
		t0 = vtt2 * (1.0f - uv0.x - uv0.y) + vtt0 * uv0.x + vtt1 * uv0.y;
		t1 = vtt2 * (1.0f - uv1.x - uv1.y) + vtt0 * uv1.x + vtt1 * uv1.y;
		t2 = vtt2 * (1.0f - uv2.x - uv2.y) + vtt0 * uv2.x + vtt1 * uv2.y;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector2f interpolate_uv(const cugar::Vector2f& uv) const
	{
		return uv2 * (1.0f - uv.x - uv.y) + uv0 * uv.x + uv1 * uv.y;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector2f uv_centroid() const
	{
		return (uv0 + uv1 + uv2) / 3.0f;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f centroid(MeshView mesh) const
	{
		const cugar::Vector2f uv = (uv0 + uv1 + uv2) / 3.0f;
		return interpolate_position( mesh, VertexGeometryId(prim_id, uv) );
	}

	uint32			prim_id;
	float			area;
	cugar::Vector2f	uv0;
	cugar::Vector2f	uv1;
	cugar::Vector2f	uv2;
};

///@} LightsModule
///@} Fermat
