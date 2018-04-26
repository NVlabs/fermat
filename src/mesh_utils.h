/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <vertex.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>

FERMAT_HOST_DEVICE inline
float prim_area(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = reinterpret_cast<const int3*>(mesh.vertex_indices)[tri_id];
	const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.x];
	const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.y];
	const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.z];

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;

	return 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));
}

FERMAT_HOST_DEVICE inline
void setup_differential_geometry(const MeshView& mesh, const uint32 tri_id, const float u, const float v, VertexGeometry* geom, float* pdf = 0)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = reinterpret_cast<const int3*>(mesh.vertex_indices)[tri_id];
	const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.x];
	const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.y];
	const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.z];

	geom->position = vp2 * (1.0f - u - v) + vp0 * u + vp1 * v; // P = rays[idx].origin + hit.t * rays[idx].dir;
	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;
	//geom->dp_du = dp_du;
	//geom->dp_dv = dp_dv;
	geom->normal_g = cugar::normalize(cugar::cross(dp_du, dp_dv));
	if (pdf)
		*pdf = 2.0f / cugar::length(cugar::cross(dp_du, dp_dv));

	if (mesh.normal_indices && mesh.normal_data)
	{
		const int3 tri = reinterpret_cast<const int3*>(mesh.normal_indices)[tri_id];
		FERMAT_ASSERT(
			(tri.x < 0 || tri.x < mesh.num_normals) &&
			(tri.y < 0 || tri.y < mesh.num_normals) &&
			(tri.z < 0 || tri.z < mesh.num_normals));
		const cugar::Vector3f vn0 = tri.x >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.x] : geom->normal_g;
		const cugar::Vector3f vn1 = tri.y >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.y] : geom->normal_g;
		const cugar::Vector3f vn2 = tri.z >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.z] : geom->normal_g;
		const cugar::Vector3f N = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);

		geom->normal_s = N;
		geom->tangent  = cugar::orthogonal(N);
		geom->binormal = cugar::cross(N, geom->tangent);
	}
	else
	{
		geom->normal_s = geom->normal_g;
		geom->tangent  = cugar::orthogonal(geom->normal_g);
		geom->binormal = cugar::cross(geom->normal_g, geom->tangent);
	}

	if (mesh.texture_indices && mesh.texture_data)
	{
		const int3 tri = reinterpret_cast<const int3*>(mesh.texture_indices)[tri_id];
		const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.x] : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.y] : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.z] : cugar::Vector2f(0.0f,0.0f);
		const cugar::Vector2f st = vt2 * (1.0f - u - v) + vt0 * u + vt1 * v;
		geom->texture_coords = cugar::Vector4f(st.x, st.y, 0.0f, 0.0f);
	}
	else
		geom->texture_coords = cugar::Vector4f(u, v, 0.0f, 0.0f);
}

FERMAT_HOST_DEVICE inline
void setup_differential_geometry(const MeshView& mesh, const VertexGeometryId v, VertexGeometry* geom, float* pdf = 0)
{
	setup_differential_geometry(mesh, v.prim_id, v.uv.x, v.uv.y, geom, pdf);
}

FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_position(const MeshView& mesh, const uint32 tri_id, const float u, const float v, float* pdf = 0)
{
	const int3 tri = reinterpret_cast<const int3*>(mesh.vertex_indices)[tri_id];
	const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.x];
	const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.y];
	const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.z];

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;

	if (pdf)
		*pdf = 2.0f / cugar::length(cugar::cross(dp_du, dp_dv));

	return vp2 * (1.0f - u - v) + vp0 * u + vp1 * v; // P = rays[idx].origin + hit.t * rays[idx].dir;
}

FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_position(const MeshView& mesh, const VertexGeometryId v, float* pdf = 0)
{
	return interpolate_position(mesh, v.prim_id, v.uv.x, v.uv.y, pdf);
}

FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_normal_s(const MeshView& mesh, const uint32 tri_id, const float u, const float v)
{
	cugar::Vector3f Ng;
	{
		const int3 tri = reinterpret_cast<const int3*>(mesh.vertex_indices)[tri_id];
		const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.x];
		const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.y];
		const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(mesh.vertex_data)[tri.z];

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		Ng = cugar::normalize(cugar::cross(dp_du, dp_dv));
	}

	if (mesh.normal_indices && mesh.normal_data)
	{
		const int3 tri = reinterpret_cast<const int3*>(mesh.normal_indices)[tri_id];
		const cugar::Vector3f vn0 = tri.x >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.x] : Ng;
		const cugar::Vector3f vn1 = tri.y >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.y] : Ng;
		const cugar::Vector3f vn2 = tri.z >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.z] : Ng;
		const cugar::Vector3f N = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);

		return N;
	}
	else
		return Ng;
}

FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_normal_s(const MeshView& mesh, const VertexGeometryId v)
{
	return interpolate_normal_s(mesh, v.prim_id, v.uv.x, v.uv.y);
}
