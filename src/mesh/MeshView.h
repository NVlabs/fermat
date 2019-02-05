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

#include <texture_reference.h>

///@addtogroup Fermat
///@{

///@addtogroup MeshModule
///@{

#ifndef SUTILCLASSAPI
#define SUTILCLASSAPI
#endif

#ifndef SUTILAPI
#define SUTILAPI
#endif

#define TEX_COORD_COMPRESSION_FIXED		0
#define TEX_COORD_COMPRESSION_HALF		1
#define TEX_COORD_COMPRESSION_MODE		TEX_COORD_COMPRESSION_HALF

///
/// The class to represent materials attached to a mesh
///
struct SUTILCLASSAPI MeshMaterial
{
	float4 diffuse;
	float4 diffuse_trans;
	float4 ambient;
	float4 specular;
	float4 emissive;
	float4 reflectivity;
	float  roughness;
	float  index_of_refraction;
	float  opacity;
	int    flags;

	TextureReference ambient_map;
	TextureReference diffuse_map;
	TextureReference diffuse_trans_map;
	TextureReference specular_map;
	TextureReference emissive_map;
	TextureReference bump_map;
};

///
/// This class provides basic Mesh view
///
struct SUTILCLASSAPI MeshView
{
	static const uint32 VERTEX_TRIANGLE_SIZE	= 4;
	static const uint32 NORMAL_TRIANGLE_SIZE	= 4;
	static const uint32 TEXTURE_TRIANGLE_SIZE	= 4;
	static const uint32 LIGHTMAP_TRIANGLE_SIZE	= 4;

	typedef int4 vertex_triangle;
	typedef int4 normal_triangle;
	typedef int4 texture_triangle;
	typedef int4 lightmap_triangle;

	typedef float4 vertex_type;
	typedef float3 normal_type;
	typedef float2 texture_coord_type;

	int num_vertices;
	int num_normals;
	int num_texture_coordinates;
	int num_triangles;
	int num_groups;
	int num_materials;

	int vertex_stride;
	int normal_stride;
	int texture_stride;
	//int padding;

	float2 tex_bias;
	float2 tex_scale;
	float2 lm_bias;
	float2 lm_scale;

	int*	vertex_indices;
	int*	normal_indices;
	int*	normal_indices_comp;
	int*	material_indices;
	int*	texture_indices;
	int*	texture_indices_comp;
	int*    group_offsets;
	float*  vertex_data;
	float*	normal_data;
	uint32*	normal_data_comp;
	float*	texture_data;
	int*	lightmap_indices;
	int*	lightmap_indices_comp;
	float*	lightmap_data;

	MeshMaterial* materials;
};

/// helper method to fetch a vertex
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
const MeshView::vertex_type& fetch_vertex(const MeshView& mesh, const uint32 vert_idx)
{
	return reinterpret_cast<const MeshView::vertex_type*>(mesh.vertex_data)[vert_idx];
}

/// helper method to fetch a normal vertex
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
const MeshView::normal_type& fetch_normal(const MeshView& mesh, const uint32 vert_idx)
{
	return reinterpret_cast<const MeshView::normal_type*>(mesh.normal_data)[vert_idx];
}

/// helper method to fetch a texture coordinate vertex
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
const MeshView::texture_coord_type& fetch_tex_coord(const MeshView& mesh, const uint32 vert_idx)
{
	return reinterpret_cast<const MeshView::texture_coord_type*>(mesh.texture_data)[vert_idx];
}

///@} MeshModule
///@} Fermat
