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

#include <MeshException.h>
#include <MeshView.h>
#include <texture_reference.h>
#include <buffers.h>
#include <string>
#include <map>

class MeshGroup;
class MeshMaterialParams;

/**
 * This class provides basic Mesh storage for either the host or device
 */
class SUTILCLASSAPI MeshStorage
{
public:
#if 0
	static const uint32 VERTEX_TRIANGLE_SIZE	= 4;
	static const uint32 NORMAL_TRIANGLE_SIZE	= 3;
	static const uint32 TEXTURE_TRIANGLE_SIZE	= 3;
	static const uint32 LIGHTMAP_TRIANGLE_SIZE	= 3;

	typedef int4 vertex_triangle;
	typedef int3 normal_triangle;
	typedef int3 texture_triangle;
	typedef int3 lightmap_triangle;

	static normal_triangle make_normal_triangle(const int32 x, const int32 y, const int32 z) { return make_int3(x,y,z); }
	static texture_triangle make_texture_triangle(const int32 x, const int32 y, const int32 z) { return make_int3(x,y,z); }
	static lightmap_triangle make_lightmap_triangle(const int32 x, const int32 y, const int32 z) { return make_int3(x,y,z); }
#else
	static const uint32 VERTEX_TRIANGLE_SIZE	= MeshView::VERTEX_TRIANGLE_SIZE;
	static const uint32 NORMAL_TRIANGLE_SIZE	= MeshView::NORMAL_TRIANGLE_SIZE;
	static const uint32 TEXTURE_TRIANGLE_SIZE	= MeshView::TEXTURE_TRIANGLE_SIZE;
	static const uint32 LIGHTMAP_TRIANGLE_SIZE	= MeshView::LIGHTMAP_TRIANGLE_SIZE;

	typedef MeshView::vertex_triangle	vertex_triangle;
	typedef MeshView::normal_triangle	normal_triangle;
	typedef MeshView::texture_triangle	texture_triangle;
	typedef MeshView::lightmap_triangle	lightmap_triangle;

	static normal_triangle make_normal_triangle(const int32 x, const int32 y, const int32 z) { return make_int4(x,y,z,0); }
	static texture_triangle make_texture_triangle(const int32 x, const int32 y, const int32 z) { return make_int4(x,y,z,0); }
	static lightmap_triangle make_lightmap_triangle(const int32 x, const int32 y, const int32 z) { return make_int4(x,y,z,0); }
#endif

	SUTILAPI MeshStorage() :
		m_num_vertices(0),
		m_num_normals(0),
		m_num_texture_coordinates(0),
		m_num_lightmap_coordinates(0),
		m_num_triangles(0),
		m_num_groups(0),
		m_vertex_stride(sizeof(MeshView::vertex_type) / 4u),
		m_normal_stride(sizeof(MeshView::normal_type) / 4u),
		m_texture_stride(sizeof(MeshView::texture_coord_type) / 4u) {}

	void alloc(
		const int num_triangles,
		const int num_vertices,
		const int num_normals,
		const int num_texture_coordinates,
		const int num_groups)
	{
		m_num_vertices			  = num_vertices;
		m_num_normals			  = num_normals;
		m_num_texture_coordinates = num_texture_coordinates;
		m_num_triangles			  = num_triangles;
		m_num_groups			  = num_groups;

		m_vertex_stride  = sizeof(MeshView::vertex_type) / 4u;
		m_normal_stride  = sizeof(MeshView::normal_type) / 4u;
		m_texture_stride = sizeof(MeshView::texture_coord_type) / 4u;

		// alloc per-triangle indices
		m_vertex_indices.alloc(4 * num_triangles);
		if (num_normals)
		{
            m_normal_indices.alloc(NORMAL_TRIANGLE_SIZE * num_triangles);
			m_normal_indices.clear(0); // initialize to 0
            m_normal_indices_comp.alloc(NORMAL_TRIANGLE_SIZE * num_triangles);
		}
		if (num_texture_coordinates)
		{
			m_texture_indices.alloc(TEXTURE_TRIANGLE_SIZE * num_triangles);
			m_texture_indices.clear(0); // initialize to 0

			m_texture_indices_comp.alloc(TEXTURE_TRIANGLE_SIZE * num_triangles);
		}

        m_material_indices.alloc(num_triangles);
        m_material_indices.clear(0xFF); // initialize to undefined|-1

		m_group_offsets.alloc(num_groups + 1);
		m_group_names.resize(num_groups);

		// alloc vertex data
		m_vertex_data.alloc(m_vertex_stride * num_vertices);
		m_normal_data.alloc(m_normal_stride * num_normals);
		m_normal_data_comp.alloc(num_normals);
		m_texture_data.alloc(m_texture_stride * num_texture_coordinates);
	}

	void alloc_lightmap(
		const int num_lightmap_coordinates)
	{
		if (num_lightmap_coordinates)
		{
			m_num_lightmap_coordinates = num_lightmap_coordinates;
			m_lightmap_indices.alloc(LIGHTMAP_TRIANGLE_SIZE * m_num_triangles);
			m_lightmap_indices_comp.alloc(TEXTURE_TRIANGLE_SIZE * m_num_triangles);
			m_lightmap_data.alloc(m_texture_stride * num_lightmap_coordinates);
		}
	}

	void compress_normals();
	void compress_tex();

	SUTILAPI MeshMaterial* alloc_materials(const size_t n) { m_materials.resize(n);  m_material_name_offsets.resize(n); return m_materials.ptr(); }
	SUTILAPI char* alloc_material_names(const size_t n_chars) { m_material_names.resize(n_chars); return m_material_names.ptr(); }

	SUTILAPI int getNumVertices() const						{ return m_num_vertices; }
	SUTILAPI int getNumNormals() const						{ return m_num_normals; }
	SUTILAPI int getNumTextureCoordinates() const			{ return m_num_texture_coordinates;  }
	SUTILAPI int getNumTriangles() const					{ return m_num_triangles; }
	SUTILAPI int getNumGroups() const					    { return m_num_groups; }
	SUTILAPI int getNumMaterials() const					{ return (int)m_materials.count(); }
	SUTILAPI int getNumTextures() const						{ return (int)m_textures.size(); }

	SUTILAPI int getVertexStride() const					{ return m_vertex_stride;  }
	SUTILAPI int getNormalStride() const					{ return m_normal_stride; }
	SUTILAPI int getTextureCoordinateStride() const			{ return m_texture_stride; }

	SUTILAPI int* getVertexIndices()						{ return m_vertex_indices.ptr(); }
	SUTILAPI const int* getVertexIndices() const			{ return m_vertex_indices.ptr(); }
	SUTILAPI int* getNormalIndices()						{ return m_normal_indices.ptr(); }
	SUTILAPI const int* getNormalIndices() const			{ return m_normal_indices.ptr(); }
	SUTILAPI int* getMaterialIndices()						{ return m_material_indices.ptr(); }
	SUTILAPI const int* getMaterialIndices() const			{ return m_material_indices.ptr(); }
	SUTILAPI int* getTextureCoordinateIndices()				{ return m_texture_indices.ptr(); }
	SUTILAPI const int* getTextureCoordinateIndices() const { return m_texture_indices.ptr(); }
	SUTILAPI int* getLightmapIndices()						{ return m_lightmap_indices.ptr(); }
	SUTILAPI const int* getLightmapIndices() const			{ return m_lightmap_indices.ptr(); }
	SUTILAPI int* getGroupOffsets()							{ return m_group_offsets.ptr(); }
	SUTILAPI const int* getGroupOffsets() const				{ return m_group_offsets.ptr(); }

	SUTILAPI float* getVertexData()							{ return m_vertex_data.ptr(); }
	SUTILAPI const float* getVertexData() const				{ return m_vertex_data.ptr(); }

	SUTILAPI float* getNormalData()							{ return m_normal_data.ptr(); }
	SUTILAPI const float* getNormalData() const				{ return m_normal_data.ptr(); }

	SUTILAPI float* getTextureCoordinateData()				{ return m_texture_data.ptr(); }
	SUTILAPI const float* getTextureCoordinateData() const	{ return m_texture_data.ptr(); }

	SUTILAPI	   std::string& getGroupName(const uint32 i)       { return m_group_names[i]; }
	SUTILAPI const std::string& getGroupName(const uint32 i) const { return m_group_names[i]; }

	SUTILAPI const char* getMaterialName(const uint32 i) const { return m_material_names.ptr() + m_material_name_offsets[i]; }

	SUTILAPI uint32 insert_texture(const std::string& tex_name);

	SUTILAPI MeshView view()
	{
		MeshView mesh;
		mesh.num_vertices				= m_num_vertices;
		mesh.num_normals				= m_num_normals;
		mesh.num_texture_coordinates	= m_num_texture_coordinates;
		mesh.num_triangles				= m_num_triangles;
		mesh.num_materials				= (int)m_materials.count();
		mesh.vertex_stride				= m_vertex_stride;
		mesh.normal_stride				= m_normal_stride;
		mesh.texture_stride				= m_texture_stride;
		mesh.vertex_indices				= m_vertex_indices.ptr();
		mesh.normal_indices				= m_normal_indices.ptr();
		mesh.normal_indices_comp		= m_normal_indices_comp.ptr();
		mesh.material_indices			= m_material_indices.ptr();
		mesh.texture_indices			= m_texture_indices.ptr();
		mesh.texture_indices_comp		= m_texture_indices_comp.ptr();
		mesh.vertex_data				= m_vertex_data.ptr();
		mesh.normal_data				= m_normal_data.ptr();
		mesh.normal_data_comp			= m_normal_data_comp.ptr();
		mesh.texture_data				= m_texture_data.ptr();
		mesh.lightmap_indices			= m_lightmap_indices.ptr();
		mesh.lightmap_indices_comp		= m_lightmap_indices_comp.ptr();
		mesh.lightmap_data				= m_lightmap_data.ptr();
		mesh.materials					= m_materials.ptr();

		mesh.tex_bias	= m_tex_bias;
		mesh.tex_scale	= m_tex_scale;
		mesh.lm_bias	= m_lm_bias;
		mesh.lm_scale	= m_lm_scale;
		return mesh;
	}

    void reorder_triangles(const int* index);
    void reset_groups(const int num_groups, const int* group_offsets);

public:
	int m_num_vertices;
	int m_num_normals;
	int m_num_texture_coordinates;
	int m_num_lightmap_coordinates;
	int m_num_triangles;
	int m_num_groups;

	int m_vertex_stride;
	int m_normal_stride;
	int m_texture_stride;

	float2 m_tex_bias;
	float2 m_tex_scale;
	float2 m_lm_bias;
	float2 m_lm_scale;

	std::map<std::string,uint32> m_textures_map;
	std::vector<std::string>	 m_textures;

	std::vector<std::string>	 m_group_names;

	Buffer<int>				m_vertex_indices;
	Buffer<int>				m_normal_indices;
	Buffer<int>				m_normal_indices_comp;
	Buffer<int>				m_material_indices;
	Buffer<int>				m_texture_indices;
	Buffer<int>				m_texture_indices_comp;
	Buffer<int>				m_lightmap_indices;
	Buffer<int>				m_lightmap_indices_comp;
	Buffer<int>				m_group_offsets;
	Buffer<float>			m_vertex_data;
	Buffer<float>			m_normal_data;
	Buffer<uint32>			m_normal_data_comp;
	Buffer<float>			m_texture_data;
	Buffer<float>			m_lightmap_data;
	Buffer<MeshMaterial>	m_materials;
	Buffer<char>			m_material_names;
	Buffer<int>				m_material_name_offsets;
};

/**
* This class provides basic Mesh storage for either the host or device
*/
class SUTILCLASSAPI DeviceMeshStorage
{
public:

	SUTILAPI DeviceMeshStorage() : m_num_vertices(0), m_num_normals(0), m_num_texture_coordinates(0), m_num_triangles(0) {}

	SUTILAPI DeviceMeshStorage& operator= (MeshStorage& mesh)
	{
		m_num_vertices		= mesh.m_num_vertices;
		m_num_normals		= mesh.m_num_normals;
		m_num_texture_coordinates	= mesh.m_num_texture_coordinates;
		m_num_lightmap_coordinates	= mesh.m_num_lightmap_coordinates;
		m_num_triangles		= mesh.m_num_triangles;
		m_num_groups		= mesh.m_num_groups;

		m_vertex_stride		= mesh.m_vertex_stride;
		m_normal_stride		= mesh.m_normal_stride;
		m_texture_stride	= mesh.m_texture_stride;

		m_vertex_indices	= mesh.m_vertex_indices;
		m_normal_indices	= mesh.m_normal_indices;
		m_material_indices	= mesh.m_material_indices;
		m_texture_indices	= mesh.m_texture_indices;
		m_lightmap_indices	= mesh.m_lightmap_indices;
		m_group_offsets		= mesh.m_group_offsets;
		m_vertex_data		= mesh.m_vertex_data;
		m_normal_data		= mesh.m_normal_data;
		m_normal_data_comp	= mesh.m_normal_data_comp;
		m_texture_data		= mesh.m_texture_data;
		m_lightmap_data		= mesh.m_lightmap_data;
		m_materials			= mesh.m_materials;

		m_normal_indices_comp	= mesh.m_normal_indices_comp;
		m_texture_indices_comp	= mesh.m_texture_indices_comp;
		m_lightmap_indices_comp	= mesh.m_lightmap_indices_comp;

		m_tex_bias	= mesh.m_tex_bias;
		m_tex_scale	= mesh.m_tex_scale;
		m_lm_bias	= mesh.m_lm_bias;
		m_lm_scale	= mesh.m_lm_scale;
		return *this;
	}

	SUTILAPI int getNumVertices() const						{ return m_num_vertices; }
	SUTILAPI int getNumNormals() const						{ return m_num_normals; }
	SUTILAPI int getNumTextureCoordinates() const			{ return m_num_texture_coordinates; }
	SUTILAPI int getNumTriangles() const					{ return m_num_triangles; }
	SUTILAPI int getNumMaterials() const					{ return (int)m_materials.count(); }

	SUTILAPI int getVertexStride() const					{ return m_vertex_stride; }
	SUTILAPI int getNormalStride() const					{ return m_normal_stride; }
	SUTILAPI int getTextureCoordinateStride() const			{ return m_texture_stride; }

	SUTILAPI int* getVertexIndices()						{ return m_vertex_indices.ptr(); }
	SUTILAPI const int* getVertexIndices() const			{ return m_vertex_indices.ptr(); }
	SUTILAPI int* getNormalIndices()						{ return m_normal_indices.ptr(); }
	SUTILAPI const int* getNormalIndices() const			{ return m_normal_indices.ptr(); }
	SUTILAPI int* getMaterialIndices()						{ return m_material_indices.ptr(); }
	SUTILAPI const int* getMaterialIndices() const			{ return m_material_indices.ptr(); }
	SUTILAPI int* getTextureCoordinateIndices()				{ return m_texture_indices.ptr(); }
	SUTILAPI const int* getTextureCoordinateIndices() const { return m_texture_indices.ptr(); }
	SUTILAPI int* getLightmapIndices()						{ return m_lightmap_indices.ptr(); }
	SUTILAPI const int* getLightmapIndices() const			{ return m_lightmap_indices.ptr(); }

	SUTILAPI float* getVertexData()							{ return m_vertex_data.ptr(); }
	SUTILAPI const float* getVertexData() const				{ return m_vertex_data.ptr(); }

	SUTILAPI float* getNormalData()							{ return m_normal_data.ptr(); }
	SUTILAPI const float* getNormalData() const				{ return m_normal_data.ptr(); }

	SUTILAPI float* getTextureCoordinateData()				{ return m_texture_data.ptr(); }
	SUTILAPI const float* getTextureCoordinateData() const	{ return m_texture_data.ptr(); }

	SUTILAPI MeshView view()
	{
		MeshView mesh;
		mesh.num_vertices				= m_num_vertices;
		mesh.num_normals				= m_num_normals;
		mesh.num_texture_coordinates	= m_num_texture_coordinates;
		mesh.num_triangles				= m_num_triangles;
		mesh.num_groups					= m_num_groups;
		mesh.num_materials				= (int)m_materials.count();
		mesh.vertex_stride				= m_vertex_stride;
		mesh.normal_stride				= m_normal_stride;
		mesh.texture_stride				= m_texture_stride;
		mesh.vertex_indices				= m_vertex_indices.ptr();
		mesh.normal_indices				= m_normal_indices.ptr();
		mesh.normal_indices_comp		= m_normal_indices_comp.ptr();
		mesh.material_indices			= m_material_indices.ptr();
		mesh.texture_indices			= m_texture_indices.ptr();
		mesh.texture_indices_comp		= m_texture_indices_comp.ptr();
		mesh.group_offsets				= m_group_offsets.ptr();
		mesh.vertex_data				= m_vertex_data.ptr();
		mesh.normal_data				= m_normal_data.ptr();
		mesh.normal_data_comp			= m_normal_data_comp.ptr();
		mesh.texture_data				= m_texture_data.ptr();
		mesh.lightmap_indices			= m_lightmap_indices.ptr();
		mesh.lightmap_indices_comp		= m_lightmap_indices_comp.ptr();
		mesh.lightmap_data				= m_lightmap_data.ptr();
		mesh.materials					= m_materials.ptr();

		mesh.tex_bias	= m_tex_bias;
		mesh.tex_scale	= m_tex_scale;
		mesh.lm_bias	= m_lm_bias;
		mesh.lm_scale	= m_lm_scale;
		return mesh;
	}

public:
	int m_num_vertices;
	int m_num_normals;
	int m_num_texture_coordinates;
	int m_num_lightmap_coordinates;
	int m_num_triangles;
	int m_num_groups;

	int m_vertex_stride;
	int m_normal_stride;
	int m_texture_stride;

	float2 m_tex_bias;
	float2 m_tex_scale;
	float2 m_lm_bias;
	float2 m_lm_scale;

	DomainBuffer<CUDA_BUFFER, int>				m_vertex_indices;
	DomainBuffer<CUDA_BUFFER, int>				m_normal_indices;
	DomainBuffer<CUDA_BUFFER, int>				m_normal_indices_comp;
	DomainBuffer<CUDA_BUFFER, int>				m_material_indices;
	DomainBuffer<CUDA_BUFFER, int>				m_texture_indices;
	DomainBuffer<CUDA_BUFFER, int>				m_texture_indices_comp;
	DomainBuffer<CUDA_BUFFER, int>				m_lightmap_indices;
	DomainBuffer<CUDA_BUFFER, int>				m_lightmap_indices_comp;
	DomainBuffer<CUDA_BUFFER, int>				m_group_offsets;
	DomainBuffer<CUDA_BUFFER, float>			m_vertex_data;
	DomainBuffer<CUDA_BUFFER, float>			m_normal_data;
	DomainBuffer<CUDA_BUFFER, uint32>			m_normal_data_comp;
	DomainBuffer<CUDA_BUFFER, float>			m_texture_data;
	DomainBuffer<CUDA_BUFFER, float>			m_lightmap_data;
	DomainBuffer<CUDA_BUFFER, MeshMaterial>		m_materials;
};

/// load a mesh
///
SUTILAPI void loadModel(const std::string& filename, MeshStorage& mesh);

/// load a material library
///
SUTILAPI void loadMaterials(const std::string& filename, MeshStorage& mesh);

/// apply material flags
///
SUTILAPI void apply_material_flags(MeshStorage& mesh);

/// translate a given group
///
SUTILAPI void translate_group(
	MeshStorage&	mesh,
	const uint32	group_id,
	const float3	delta);

/// translate a given group
///
SUTILAPI void translate_group(
	DeviceMeshStorage&	mesh,
	const uint32		group_id,
	const float3		delta);

/// merge two meshes
///
SUTILAPI void merge(MeshStorage& mesh, const MeshStorage& other);

/// transform a mesh
///
SUTILAPI void transform(
	MeshStorage&	mesh,
	const float		mat[4*4]);

/// add per-triangle normals
///
SUTILAPI void add_per_triangle_normals(MeshStorage& mesh);

/// add per-triangle texture coordinates
///
SUTILAPI void add_per_triangle_texture_coordinates(MeshStorage& mesh);

/// unify all vertex attributes, so as to have a single triplet of indices per triangle
///
SUTILAPI void unify_vertex_attributes(MeshStorage& mesh);
