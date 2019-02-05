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

#include <mesh/MeshStorage.h>
#include <mesh/MeshCompression.h>
#include <vertex.h>
#include <cugar/basic/cuda/pointers.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <cugar/spherical/mappings.h>

#define NORMAL_COMPRESSION			1
#define TEX_COORD_COMPRESSION		1
#define UNIFIED_VERTEX_ATTRIBUTES	1

///@addtogroup Fermat
///@{

///@addtogroup MeshModule
///@{

enum TextureSet
{
	kTextureCoords0 = 0,
	kLightmapCoords = 1
};

template <typename T>
struct load_triangle_dispatch {};

template <>
struct load_triangle_dispatch<int3>
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static int3 load(const int* indices, const uint32 tri_idx) { return reinterpret_cast<const int3*>(indices)[tri_idx]; }
};

template <>
struct load_triangle_dispatch<int4>
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static int3 load(const int* indices, const uint32 tri_idx)
	{
	  #ifdef FERMAT_DEVICE_COMPILATION
		const int4 i4 = cugar::cuda::load<cugar::cuda::LOAD_DEFAULT>( reinterpret_cast<const int4*>(indices) + tri_idx );
//		const int4 i4 = reinterpret_cast<const int4*>(indices)[tri_idx];
	  #else
		const int4 i4 = reinterpret_cast<const int4*>(indices)[tri_idx];
	  #endif
		return make_int3(i4.x, i4.y, i4.z);
	}
};

template <typename T>
struct load_vertex_dispatch {};

template <>
struct load_vertex_dispatch<float3>
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static float3 load(const MeshView& mesh, const uint32 vert_idx) { return reinterpret_cast<const float3*>(mesh.vertex_data)[vert_idx]; }
};

template <>
struct load_vertex_dispatch<float4>
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static float4 load4(const MeshView& mesh, const uint32 vert_idx)
	{
	  #ifdef FERMAT_DEVICE_COMPILATION
		return cugar::cuda::load<cugar::cuda::LOAD_LDG>( reinterpret_cast<const float4*>(mesh.vertex_data) + vert_idx );
	  #else
		return reinterpret_cast<const float4*>(mesh.vertex_data)[vert_idx];
	  #endif
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static float3 load(const MeshView& mesh, const uint32 vert_idx)
	{
		const float4 v4 = load4( mesh, vert_idx );

		return make_float3(v4.x, v4.y, v4.z);
	}
};

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f vertex_comp(const cugar::Vector3f& v) { return v; }

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f vertex_comp(const cugar::Vector4f& v) { return v.xyz(); }

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f load_vertex(const MeshView& mesh, const uint32 vert_idx)
{
	return load_vertex_dispatch<MeshView::vertex_type>::load( mesh, vert_idx );
}

#if UNIFIED_VERTEX_ATTRIBUTES
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector4f load_full_vertex(const MeshView& mesh, const uint32 vert_idx)
{
	return load_vertex_dispatch<MeshView::vertex_type>::load4( mesh, vert_idx );
}
#endif

/// helper method to fetch the vertex indices of a given triangle
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
int3 load_vertex_triangle(const int* triangle_indices, const uint32 tri_idx)
{
	return load_triangle_dispatch<MeshStorage::vertex_triangle>::load( triangle_indices, tri_idx );
}

/// helper method to fetch the normal indices of a given triangle
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
int3 load_normal_triangle(const int* triangle_indices, const uint32 tri_idx)
{
	return load_triangle_dispatch<MeshStorage::normal_triangle>::load( triangle_indices, tri_idx );
}

/// helper method to fetch the texture coordinate indices of a given triangle
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
int3 load_texture_triangle(const int* triangle_indices, const uint32 tri_idx)
{
	return load_triangle_dispatch<MeshStorage::texture_triangle>::load( triangle_indices, tri_idx );
}

/// helper method to fetch the lightmap coordinate indices of a given triangle
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
int3 load_lightmap_triangle(const int* triangle_indices, const uint32 tri_idx)
{
	return load_triangle_dispatch<MeshStorage::lightmap_triangle>::load( triangle_indices, tri_idx );
}

/// return the area of a given primitive
///
FERMAT_HOST_DEVICE inline
float prim_area(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;

	return 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));
}

/// return the differential geometry of a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
void setup_differential_geometry(const MeshView& mesh, const uint32 tri_id, const float u, const float v, VertexGeometry* geom, float* pdf = 0)
{
#if UNIFIED_VERTEX_ATTRIBUTES
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector4f vp0_w = load_full_vertex(mesh, tri.x);
	const cugar::Vector4f vp1_w = load_full_vertex(mesh, tri.y);
	const cugar::Vector4f vp2_w = load_full_vertex(mesh, tri.z);

	const cugar::Vector3f vp0 = vp0_w.xyz();
	const cugar::Vector3f vp1 = vp1_w.xyz();
	const cugar::Vector3f vp2 = vp2_w.xyz();

	geom->position = vp2 * (1.0f - u - v) + vp0 * u + vp1 * v; // P = rays[idx].origin + hit.t * rays[idx].dir;
	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;
	//geom->dp_du = dp_du;
	//geom->dp_dv = dp_dv;
	geom->normal_g = cugar::normalize(cugar::cross(dp_du, dp_dv));
	if (pdf)
		*pdf = 2.0f / cugar::length(cugar::cross(dp_du, dp_dv));

	// unpack the normals
	{
		const cugar::Vector3f vn0 = cugar::unpack_normal(cugar::binary_cast<uint32>( vp0_w.w ));
		const cugar::Vector3f vn1 = cugar::unpack_normal(cugar::binary_cast<uint32>( vp1_w.w ));
		const cugar::Vector3f vn2 = cugar::unpack_normal(cugar::binary_cast<uint32>( vp2_w.w ));

		const cugar::Vector3f N = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);

		geom->normal_s = N;
		geom->tangent  = cugar::orthogonal(N);
		geom->binormal = cugar::cross(N, geom->tangent);
	}
#else
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

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
	#if NORMAL_COMPRESSION
		const int3 tri = load_normal_triangle( mesh.normal_indices_comp, tri_id );
		const cugar::Vector3f vn0 = tri.x >= 0 ? cugar::unpack_normal(tri.x) : geom->normal_g;
		const cugar::Vector3f vn1 = tri.y >= 0 ? cugar::unpack_normal(tri.y) : geom->normal_g;
		const cugar::Vector3f vn2 = tri.z >= 0 ? cugar::unpack_normal(tri.z) : geom->normal_g;
	#else
		const int3 tri = load_normal_triangle( mesh.normal_indices, tri_id );
		FERMAT_ASSERT(
			(tri.x < 0 || tri.x < mesh.num_normals) &&
			(tri.y < 0 || tri.y < mesh.num_normals) &&
			(tri.z < 0 || tri.z < mesh.num_normals));
		const cugar::Vector3f vn0 = tri.x >= 0 ? cugar::unpack_normal(mesh.normal_data_comp[tri.x]) : geom->normal_g;
		const cugar::Vector3f vn1 = tri.y >= 0 ? cugar::unpack_normal(mesh.normal_data_comp[tri.y]) : geom->normal_g;
		const cugar::Vector3f vn2 = tri.z >= 0 ? cugar::unpack_normal(mesh.normal_data_comp[tri.z]) : geom->normal_g;
		//const cugar::Vector3f vn0 = tri.x >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.x] : geom->normal_g;
		//const cugar::Vector3f vn1 = tri.y >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.y] : geom->normal_g;
		//const cugar::Vector3f vn2 = tri.z >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.z] : geom->normal_g;
	#endif
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
#endif

	#if TEX_COORD_COMPRESSION
	if (mesh.texture_indices_comp)
	{
		const int3 tri = load_texture_triangle( mesh.texture_indices_comp, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? decompress_tex_coord(mesh, tri.x) : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? decompress_tex_coord(mesh, tri.y) : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? decompress_tex_coord(mesh, tri.z) : cugar::Vector2f(0.0f,0.0f);
	#else
	if (mesh.texture_indices && mesh.texture_data)
	{
		const int3 tri = load_texture_triangle( mesh.texture_indices, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.x] : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.y] : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.z] : cugar::Vector2f(0.0f,0.0f);
	#endif
		const cugar::Vector2f st = vt2 * (1.0f - u - v) + vt0 * u + vt1 * v;
		geom->texture_coords = cugar::Vector4f(st.x, st.y, 0.0f, 0.0f);
	}
	else
		geom->texture_coords = cugar::Vector4f(u, v, 0.0f, 0.0f);

	#if TEX_COORD_COMPRESSION
	if (mesh.lightmap_indices_comp)
	{
		const int3 tri = load_lightmap_triangle( mesh.lightmap_indices_comp, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? decompress_lightmap_coord(mesh, tri.x) : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? decompress_lightmap_coord(mesh, tri.y) : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? decompress_lightmap_coord(mesh, tri.z) : cugar::Vector2f(0.0f,0.0f);
	#else
	if (mesh.lightmap_indices && mesh.lightmap_data)
	{
		const int3 tri = load_lightmap_triangle( mesh.lightmap_indices, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.x] : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.y] : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.z] : cugar::Vector2f(0.0f,0.0f);
	#endif
		const cugar::Vector2f st = vt2 * (1.0f - u - v) + vt0 * u + vt1 * v;
		geom->lightmap_coords = cugar::Vector2f(st.x, st.y);
	}
	else
		geom->lightmap_coords = cugar::Vector2f(0.0f, 0.0f);
}

/// return the differential geometry of a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
void setup_differential_geometry(const MeshView& mesh, const VertexGeometryId v, VertexGeometry* geom, float* pdf = 0)
{
	setup_differential_geometry(mesh, v.prim_id, v.uv.x, v.uv.y, geom, pdf);
}

/// return the interpolated position at a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_position(const MeshView& mesh, const uint32 tri_id, const float u, const float v, float* pdf = 0)
{
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;

	if (pdf)
		*pdf = 2.0f / cugar::length(cugar::cross(dp_du, dp_dv));

	return vp2 * (1.0f - u - v) + vp0 * u + vp1 * v; // P = rays[idx].origin + hit.t * rays[idx].dir;
}

/// return the interpolated position at a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_position(const MeshView& mesh, const VertexGeometryId v, float* pdf = 0)
{
	return interpolate_position(mesh, v.prim_id, v.uv.x, v.uv.y, pdf);
}

/// return the interpolated normal at a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_normal(const MeshView& mesh, const uint32 tri_id, const float u, const float v)
{
#if UNIFIED_VERTEX_ATTRIBUTES
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vn0 = cugar::unpack_normal( cugar::binary_cast<uint32>( fetch_vertex(mesh, tri.z).w ) );
	const cugar::Vector3f vn1 = cugar::unpack_normal( cugar::binary_cast<uint32>( fetch_vertex(mesh, tri.y).w ) );
	const cugar::Vector3f vn2 = cugar::unpack_normal( cugar::binary_cast<uint32>( fetch_vertex(mesh, tri.z).w ) );

	const cugar::Vector3f N = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);

	return N;
#else
	cugar::Vector3f Ng;
	{
		const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
		const cugar::Vector3f vp0 = load_vertex(mesh, tri.z);
		const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
		const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		Ng = cugar::normalize(cugar::cross(dp_du, dp_dv));
	}

	if (mesh.normal_indices && mesh.normal_data)
	{
	#if NORMAL_COMPRESSION
		const int3 tri = load_normal_triangle( mesh.normal_indices_comp, tri_id );
		const cugar::Vector3f vn0 = tri.x >= 0 ? cugar::unpack_normal(tri.x) : Ng;
		const cugar::Vector3f vn1 = tri.y >= 0 ? cugar::unpack_normal(tri.y) : Ng;
		const cugar::Vector3f vn2 = tri.z >= 0 ? cugar::unpack_normal(tri.z) : Ng;
	#else
		const int3 tri = load_normal_triangle( mesh.normal_indices, tri_id );
		const cugar::Vector3f vn0 = tri.x >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.x] : Ng;
		const cugar::Vector3f vn1 = tri.y >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.y] : Ng;
		const cugar::Vector3f vn2 = tri.z >= 0 ? reinterpret_cast<const float3*>(mesh.normal_data)[tri.z] : Ng;
	#endif
		const cugar::Vector3f N = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);

		return N;
	}
	else
		return Ng;
#endif
}

/// return the interpolated normal at a given point on the mesh, specified by a (prim_id, uv) pair
///
FERMAT_HOST_DEVICE inline
cugar::Vector3f interpolate_normal(const MeshView& mesh, const VertexGeometryId v)
{
	return interpolate_normal(mesh, v.prim_id, v.uv.x, v.uv.y);
}


/// build a matrix representing the rate of change of (s,t) wrt to (u,v) coordinates;
/// specifically, the matrix which sends:
///
///  u=(1,0) in dst_du
///  v=(0,1) in dst_dv
///
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_dst_duv(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	if (TEXTURE_SET == kTextureCoords0 && mesh.texture_indices)
	{
		const int3 tri = load_texture_triangle( mesh.texture_indices, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.x] : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.y] : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.texture_data)[tri.z] : cugar::Vector2f(0.0f,0.0f);

		// build the matrix which sends:
		//  u=(1,0) in dst_du
		//  v=(0,1) in dst_dv
		cugar::Matrix2x2f M;
		M[0] = vt0 - vt2; // dst_du
		M[1] = vt1 - vt2; // dst_dv
		return cugar::transpose(M);
	}
	else if (TEXTURE_SET == kLightmapCoords && mesh.lightmap_indices)
	{
		const int3 tri = load_lightmap_triangle( mesh.lightmap_indices, tri_id );
		const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.x] : cugar::Vector2f(1.0f,0.0f);
		const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.y] : cugar::Vector2f(0.0f,1.0f);
		const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(mesh.lightmap_data)[tri.z] : cugar::Vector2f(0.0f,0.0f);

		// build the matrix which sends:
		//  u=(1,0) in dst_du
		//  v=(0,1) in dst_dv
		cugar::Matrix2x2f M;
		M[0] = vt0 - vt2; // dst_du
		M[1] = vt1 - vt2; // dst_dv
		return cugar::transpose(M);
	}
	return cugar::Matrix2x2f::one();
}

/// build a matrix representing the rate of change of (u,v) wrt (s,t) coordinates;
/// specifically, the matrix which sends:
///
///  dst_du in u=(1,0)
///  dst_dv in v=(0,1)
///
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_duv_dst(const MeshView& mesh, const uint32 tri_id)
{
	const cugar::Matrix2x2f dst_duv = prim_dst_duv<TEXTURE_SET>(mesh, tri_id);
	cugar::Matrix2x2f R;
	cugar::invert(dst_duv, R);
	return R;
}

/// build a matrix representing the rate of change of dp wrt to (u,v,w) coordinates;
/// specifically, the matrix which sends:
///
///  u=(1,0) in dp_du
///  v=(0,1) in dp_dv
///
FERMAT_HOST_DEVICE inline
cugar::Matrix3x2f prim_dp_duv(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;

	// build the matrix which sends:
	//  u=(1,0,0) in dp_dpu
	//  v=(0,1,0) in dp_dpv
	cugar::Matrix3x2f M;
	M[0][0] = dp_du.x;
	M[1][0] = dp_du.y;
	M[2][0] = dp_du.z;
	M[0][1] = dp_dv.x;
	M[1][1] = dp_dv.y;
	M[2][1] = dp_dv.z;
	return M;
}

/// build a matrix representing the rate of change of dp wrt to (u,v,w) coordinates;
/// specifically, the matrix which sends:
///
///  u=(1,0,0) in dp_du
///  v=(0,1,0) in dp_dv
///  w=(0,0,1) in dp_dw
///
FERMAT_HOST_DEVICE inline
cugar::Matrix3x3f prim_dp_duvw(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;
	const cugar::Vector3f dp_dw = cugar::cross(dp_du, dp_dv);

	// build the matrix which sends:
	//  u=(1,0,0) in dp_du
	//  v=(0,1,0) in dp_dv
	//  w=(0,0,1) in dp_dw
	cugar::Matrix3x3f M;
	M[0][0] = dp_du.x;
	M[1][0] = dp_du.y;
	M[2][0] = dp_du.z;
	M[0][1] = dp_dv.x;
	M[1][1] = dp_dv.y;
	M[2][1] = dp_dv.z;
	M[0][2] = dp_dw.x;
	M[1][2] = dp_dw.y;
	M[2][2] = dp_dw.z;
	return M;
}

/// build a matrix representing the rate of change of dp wrt to (s,t) coordinates;
/// specifically, the matrix which sends:
///
///  s=(1,0) in dp_ds
///  t=(0,1) in dp_dt
///
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix3x2f prim_dp_dst(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	return prim_dp_duv( mesh, tri_id ) * prim_duv_dst<TEXTURE_SET>( mesh, tri_id );
}

/// build a matrix representing the rate of change of (u,v,w) wrt to world coordinates;
/// specifically, the matrix which sends:
///
///  dp_du in u=(1,0,0)
///  dp_dv in v=(0,1,0)
///  dp_dw in w=(0,0,1)
///
FERMAT_HOST_DEVICE inline
cugar::Matrix3x3f prim_duvw_dp(const MeshView& mesh, const uint32 tri_id)
{
	// build the matrix which sends:
	//  dp_du in u=(1,0,0)
	//  dp_dv in v=(0,1,0)
	//  dp_dw in w=(0,0,1)
	const cugar::Matrix3x3f dp_duvw = prim_dp_duvw(mesh,tri_id);
	cugar::Matrix3x3f R;
	cugar::invert( dp_duvw, R );
	return R;
}

/// build a matrix representing the rate of change of dp wrt to (u,v,w) coordinates,
/// where dp is expressed in the local tangent frame (t,b);
/// specifically, the matrix which sends:
///
///  u=(1,0) in the coordinates of dp_dpu wrt (t,b), and
///  v=(0,1) in the coordinates of dp_dpv wrt (t,b)
///  
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_dtb_duv(const MeshView& mesh, const uint32 tri_id)
{
	FERMAT_ASSERT(tri_id < uint32(mesh.num_triangles));
	const int3 tri = load_vertex_triangle( mesh.vertex_indices, tri_id );
	const cugar::Vector3f vp0 = load_vertex(mesh, tri.x);
	const cugar::Vector3f vp1 = load_vertex(mesh, tri.y);
	const cugar::Vector3f vp2 = load_vertex(mesh, tri.z);

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;
	const cugar::Vector3f dp_dw = cugar::cross(dp_du, dp_dv);
	const cugar::Vector3f n = cugar::normalize(dp_dw);
	const cugar::Vector3f t = cugar::normalize(dp_du); //cugar::orthogonal(n);
	const cugar::Vector3f b = cugar::cross(n, t);

	// build the matrix which sends:
	//  u=(1,0) in the coordinates of dp_dpu wrt (t,b), and
	//  v=(0,1) in the coordinates of dp_dpv wrt (t,b)
	cugar::Matrix2x2f M;
	M[0][0] = cugar::dot(dp_du, t);
	M[1][0] = 0.0f;
	M[0][1] = cugar::dot(dp_dv, t);
	M[1][1] = cugar::dot(dp_dv, b);
	return M;
}

/// build a matrix representing the rate of change of (u,v) wrt to dp, where dp is
/// expressed in the local tangent frame (t,b);
/// specifically, the matrix which sends:
///
///  dp_du wrt (t,b) in u=(1,0)
///  dp_dv wrt (t,b) in v=(0,1)
///  
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_duv_dtb(const MeshView& mesh, const uint32 tri_id)
{
	// build the matrix which sends:
	//  t in u=(1,0)
	//  b in v=(0,1)
	const cugar::Matrix2x2f dtb_duv = prim_dtb_duv(mesh,tri_id);
	cugar::Matrix2x2f R;
	cugar::invert( dtb_duv, R );
	return R;
}

/// build a matrix representing the rate of change of (u,v) wrt to dp;
/// specifically, the matrix which sends:
///
///  dp_du in u=(1,0)
///  dp_dv in v=(0,1)
///  
FERMAT_HOST_DEVICE inline
cugar::Matrix2x3f prim_duv_dp(const MeshView& mesh, const uint32 tri_id)
{
	// build the matrix which sends:
	//  dp_du in u=(1,0)
	//  dp_dv in v=(0,1)
	const cugar::Matrix3x3f duvw_dp = prim_duvw_dp(mesh,tri_id);
	cugar::Matrix2x3f R;
	R[0] = duvw_dp[0];
	R[1] = duvw_dp[1];
	return R;
}

/// build a matrix representing the rate of change of (s,t) wrt to dp, where dp is
/// expressed in the world coordinates;
/// specifically, the matrix which sends:
///
///  dp_du in dst_du
///  dp_dv in dst_dv
///  
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix2x3f prim_dst_dp(const MeshView& mesh, const uint32 tri_id)
{
	// build the matrix which sends:
	//  dp_du in dst_du
	//  dp_dv in dst_dv
	const cugar::Matrix2x2f dst_duv = prim_dst_duv<TEXTURE_SET>(mesh,tri_id);
	const cugar::Matrix2x3f duv_dp =  prim_duv_dp(mesh,tri_id);
	return dst_duv * duv_dp;
}

/// build a matrix representing the rate of change of (s,t) wrt to dp, where dp is
/// expressed in the local tangent frame (t,b);
/// specifically, the matrix which sends:
///
///  t=(1,0) in dst_dt
///  b=(0,1) in dst_db
///  
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_dst_dtb(const MeshView& mesh, const uint32 tri_id)
{
	// build the matrix which sends:
	//  t=(1,0) in dst_dt
	//  b=(0,1) in dst_db
	const cugar::Matrix2x2f dst_duv = prim_dst_duv<TEXTURE_SET>(mesh,tri_id);
	const cugar::Matrix2x2f duv_dtb = prim_duv_dtb(mesh,tri_id);
	return dst_duv * duv_dtb;
}

/// build a matrix representing the rate of change of dp wrt (s,t), where dp is
/// expressed in the local tangent frame (t,b);
/// specifically, the matrix which sends:
///
///  dst_dt in t=(1,0)
///  dst_db in b=(0,1)
///  
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Matrix2x2f prim_dtb_dst(const MeshView& mesh, const uint32 tri_id)
{
	const cugar::Matrix2x2f duv_dst = prim_duv_dst<TEXTURE_SET>(mesh,tri_id);
	const cugar::Matrix2x2f dtb_duv = prim_duv_dtb(mesh,tri_id);
	return dtb_duv * duv_dst;
}

/// return the length of the semi-axes of the differential texture-space (s,t) ellipse corresponding to a unit-radius circle
/// on the local tangent plane.
///  
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
cugar::Vector2f prim_dst_ellipse_size(const MeshView& mesh, const uint32 tri_id)
{
	// if we consider the transformation of the unit-circle defined by the equation: Y = dst_dtb * X
	// the semi-axes of the resulting ellipse can be obtained looking at the SVD of dst_dtb
	const cugar::Matrix2x2f dst_dtb = prim_dst_dtb<TEXTURE_SET>(mesh,tri_id);
	return cugar::singular_values(dst_dtb);
}

/// return the the semi-axes of the differential texture-space (s,t) ellipse corresponding to a unit-radius circle
/// on the local tangent plane.
///
/// \param u			output matrix storing the two semi-axes in the columns
/// \param s			output vector storing the length of the two semi-axes
///
template <TextureSet TEXTURE_SET>
FERMAT_HOST_DEVICE inline
void prim_dst_ellipse(const MeshView& mesh, const uint32 tri_id, cugar::Matrix2x2f& u, cugar::Vector2f& s)
{
	// if we consider the transformation of the unit-circle defined by the equation: Y = dst_dtb * X
	// the semi-axes of the resulting ellipse can be obtained looking at the SVD of dst_dtb
	const cugar::Matrix2x2f dst_dtb = prim_dst_dtb<TEXTURE_SET>(mesh,tri_id);
	cugar::Matrix2x2f v;
	cugar::svd(dst_dtb, u, s, v);
}

///@} MeshModule
///@} Fermat
