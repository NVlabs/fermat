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

#include "MeshStorage.h"
#include "MeshLoader.h"
#include "MeshCompression.h"
#include <set>
#include <cugar/linalg/matrix.h>
#include <cugar/linalg/bbox.h>

TextureReference insert_texture(std::map<std::string, uint32>& map, std::vector<std::string>& vec, uint32& texture_count, const MeshTextureMap& tex)
{
	TextureReference res;

	if (tex.name.length() == 0)
		res.texture = TextureReference::INVALID;
	else
	{
		std::map<std::string, uint32>::const_iterator it = map.find(tex.name);
		if (it == map.end())
		{
			map.insert(std::make_pair(tex.name, texture_count));
			vec.push_back(tex.name);
			res.texture = texture_count++;
		}
		else
			res.texture = it->second;
	}

	res.scaling.x = tex.scaling[0];
	res.scaling.y = tex.scaling[1];
	return res;
}

uint32 insert_texture(std::map<std::string, uint32>& map, std::vector<std::string>& vec, const std::string& tex_name)
{
	if (tex_name.length() == 0)
		return TextureReference::INVALID;

	std::map<std::string, uint32>::const_iterator it = map.find(tex_name);
	if (it == map.end())
	{
		const uint32 texture_count = uint32(vec.size());

		map.insert(std::make_pair(tex_name, texture_count));
		vec.push_back(tex_name);
		return texture_count;
	}
	else
		return it->second;
}

uint32 MeshStorage::insert_texture(const std::string& tex_name)
{
	return ::insert_texture( m_textures_map, m_textures, tex_name );
}

namespace {

void convert_color(const float4& in, float out[4])
{
	out[0] = in.x;
	out[1] = in.y;
	out[2] = in.z;
	out[3] = in.w;
}

MeshTextureMap convert_texture(const MeshStorage& mesh, const TextureReference& texture_ref)
{
	if (texture_ref.is_valid() == false)
		return MeshTextureMap();

	MeshTextureMap r;
	r.name = mesh.m_textures[ texture_ref.texture ];
	r.scaling[0] = texture_ref.scaling.x;
	r.scaling[1] = texture_ref.scaling.y;
	return r;
}

MeshMaterialParams convert_material(const MeshStorage& mesh, const MeshMaterial& material)
{
	MeshMaterialParams params;
	convert_color( material.ambient,		params.ambient );
	convert_color( material.diffuse,		params.diffuse );
	convert_color( material.diffuse_trans,	params.diffuse_trans );
	convert_color( material.specular,		params.specular );
	convert_color( material.emissive,		params.emissive );
	convert_color( material.reflectivity,	params.reflectivity );

	params.index_of_refraction	= material.index_of_refraction;
	params.opacity				= material.opacity;
	params.flags				= material.flags;

	params.ambient_map			= convert_texture( mesh, material.ambient_map );
	params.diffuse_map			= convert_texture( mesh, material.diffuse_map );
	params.diffuse_trans_map	= convert_texture( mesh, material.diffuse_trans_map );
	params.specular_map			= convert_texture( mesh, material.specular_map );
	params.emissive_map			= convert_texture( mesh, material.emissive_map );
	params.bump_map				= convert_texture( mesh, material.bump_map );
}

}

void loadModel(const std::string& filename, MeshStorage& mesh)
{
	MeshLoader loader( &mesh );
    loader.setMeshGrouping( kKeepGroups );
	loader.loadModel( filename );

	// perform normal compression
	//mesh.compress_normals();
	//mesh.compress_tex();

	MeshMaterial* materials = mesh.alloc_materials(loader.getMaterialCount());

	// keep track of the number of characters needed for the material names
	int n_chars = 0;

	// keep track of the number of textures
	uint32 texture_count = 0;

	for (int i = 0; i < (int)loader.getMaterialCount(); ++i)
	{
		const MeshMaterialParams params = loader.getMeshMaterialParams(i);

		// setup the material name directory
		mesh.m_material_name_offsets.set( i, n_chars );

		// keep track of the number of characters needed for the material names
		n_chars += (int)params.name.length() + 1;

		materials[i].ambient				= make_float4(params.ambient[0], params.ambient[1], params.ambient[2],0.0f);
		materials[i].diffuse				= make_float4(params.diffuse[0], params.diffuse[1], params.diffuse[2],0.0f);
		materials[i].diffuse_trans			= make_float4(params.diffuse_trans[0], params.diffuse_trans[1], params.diffuse_trans[2], 0.0f);
		materials[i].specular				= make_float4(params.specular[0], params.specular[1], params.specular[2], 0.0f);
		materials[i].emissive				= make_float4(params.emissive[0], params.emissive[1], params.emissive[2], 0.0f);
		materials[i].reflectivity			= make_float4(params.reflectivity[0], params.reflectivity[1], params.reflectivity[2], 0.0f);
		materials[i].roughness				= params.phong_exponent ? 1.0f / powf(params.phong_exponent,1.0f) : 1.0f;
		materials[i].index_of_refraction	= params.index_of_refraction;
		materials[i].opacity				= params.opacity;
		materials[i].flags					= params.flags;

		materials[i].ambient_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.ambient_map);
		materials[i].diffuse_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_map);
		materials[i].diffuse_trans_map	= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_trans_map);
		materials[i].specular_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.specular_map);
		materials[i].emissive_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.emissive_map);
		materials[i].bump_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.bump_map);
	}

	// copy the material names
	mesh.alloc_material_names( n_chars );
	for (int i = 0; i < (int)loader.getMaterialCount(); ++i)
	{
		const MeshMaterialParams params = loader.getMeshMaterialParams(i);

		const int offset = mesh.m_material_name_offsets[i];

		strcpy( mesh.m_material_names.ptr() + offset, params.name.c_str() );
	}
}

// load a material library
//
SUTILAPI void loadMaterials(const std::string& filename, MeshStorage& mesh)
{
	MeshLoader loader( &mesh );
	loader.loadMaterials( filename );

	const int material_offset = mesh.getNumMaterials();

	MeshMaterial* materials = mesh.alloc_materials(material_offset + loader.getMaterialCount());

	// keep track of the number of characters needed for the material names
	int n_chars = (int)mesh.m_material_names.count();

	uint32 texture_count = (uint32)mesh.m_textures.size();

	for (int i = 0; i < (int)loader.getMaterialCount(); ++i)
	{
		const MeshMaterialParams params = loader.getMeshMaterialParams(i);

		// setup the material name directory
		mesh.m_material_name_offsets.set( material_offset + i, n_chars );

		// keep track of the number of characters needed for the material names
		n_chars += (int)params.name.length() + 1;

		materials[material_offset + i].ambient				= make_float4(params.ambient[0], params.ambient[1], params.ambient[2],0.0f);
		materials[material_offset + i].diffuse				= make_float4(params.diffuse[0], params.diffuse[1], params.diffuse[2],0.0f);
		materials[material_offset + i].diffuse_trans		= make_float4(params.diffuse_trans[0], params.diffuse_trans[1], params.diffuse_trans[2], 0.0f);
		materials[material_offset + i].specular				= make_float4(params.specular[0], params.specular[1], params.specular[2], 0.0f);
		materials[material_offset + i].emissive				= make_float4(params.emissive[0], params.emissive[1], params.emissive[2], 0.0f);
		materials[material_offset + i].reflectivity			= make_float4(params.reflectivity[0], params.reflectivity[1], params.reflectivity[2], 0.0f);
		materials[material_offset + i].roughness			= params.phong_exponent ? 1.0f / powf(params.phong_exponent,1.0f) : 1.0f;
		materials[material_offset + i].index_of_refraction	= params.index_of_refraction;
		materials[material_offset + i].opacity				= params.opacity;
		materials[material_offset + i].flags				= params.flags;

		materials[material_offset + i].ambient_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.ambient_map);
		materials[material_offset + i].diffuse_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_map);
		materials[material_offset + i].diffuse_trans_map	= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_trans_map);
		materials[material_offset + i].specular_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.specular_map);
		materials[material_offset + i].emissive_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.emissive_map);
		materials[material_offset + i].bump_map.texture		= TextureReference::INVALID;
		//materials[material_offset + i].bump_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.bump_map);
	}

	// copy the material names
	mesh.alloc_material_names( n_chars );
	for (int i = 0; i < (int)loader.getMaterialCount(); ++i)
	{
		const MeshMaterialParams params = loader.getMeshMaterialParams(i);

		const int offset = mesh.m_material_name_offsets[material_offset + i];

		strcpy( mesh.m_material_names.ptr() + offset, params.name.c_str() );
	}
}

void MeshStorage::compress_normals()
{
	if (m_num_normals)
	{
		m_normal_data_comp.alloc(m_num_normals);
		m_normal_indices_comp.alloc(NORMAL_TRIANGLE_SIZE * m_num_triangles);

		for (int i = 0; i < m_num_normals; ++i)
			m_normal_data_comp.set( i, cugar::pack_normal( reinterpret_cast<const float3*>(m_normal_data.ptr())[i] ) );

		normal_triangle* normal_indices      = reinterpret_cast<normal_triangle*>(m_normal_indices.ptr());
		normal_triangle* comp_normal_indices = reinterpret_cast<normal_triangle*>(m_normal_indices_comp.ptr());

		for (int i = 0; i < m_num_triangles; ++i)
		{
			normal_triangle tri = normal_indices[i];

			comp_normal_indices[i].x = tri.x >= 0 ? m_normal_data_comp[ tri.x ] : -1;
			comp_normal_indices[i].y = tri.y >= 0 ? m_normal_data_comp[ tri.y ] : -1;
			comp_normal_indices[i].z = tri.z >= 0 ? m_normal_data_comp[ tri.z ] : -1;
		}
	}
}

void MeshStorage::compress_tex()
{
	if (m_num_texture_coordinates)
	{
		MeshView::texture_coord_type* texture_data = reinterpret_cast<MeshView::texture_coord_type*>(m_texture_data.ptr());

		cugar::Bbox2f bbox;
		for (int i = 0; i < m_num_texture_coordinates; ++i)
			bbox.insert( texture_data[i] );

		m_tex_bias  = bbox[0];
		m_tex_scale = bbox[1] - bbox[0];

		m_texture_indices_comp.alloc(TEXTURE_TRIANGLE_SIZE * m_num_triangles);

		texture_triangle* texture_indices      = reinterpret_cast<texture_triangle*>(m_texture_indices.ptr());
		texture_triangle* texture_indices_comp = reinterpret_cast<texture_triangle*>(m_texture_indices_comp.ptr());

		for (int i = 0; i < m_num_triangles; ++i)
		{
			texture_triangle tri = texture_indices[i];

			cugar::Vector2f t0 = tri.x >= 0 ? texture_data[tri.x] : cugar::Vector2f(0.0f);
			cugar::Vector2f t1 = tri.y >= 0 ? texture_data[tri.y] : cugar::Vector2f(0.0f);
			cugar::Vector2f t2 = tri.z >= 0 ? texture_data[tri.z] : cugar::Vector2f(0.0f);

			texture_indices_comp[i].x = tri.x >= 0 ? compress_tex_coord(t0, m_tex_bias, m_tex_scale) : -1;
			texture_indices_comp[i].y = tri.y >= 0 ? compress_tex_coord(t1, m_tex_bias, m_tex_scale) : -1;
			texture_indices_comp[i].z = tri.z >= 0 ? compress_tex_coord(t2, m_tex_bias, m_tex_scale) : -1;
		}
	}

	if (m_num_lightmap_coordinates)
	{
		MeshView::texture_coord_type* lightmap_data = reinterpret_cast<MeshView::texture_coord_type*>(m_lightmap_data.ptr());

		cugar::Bbox2f bbox;
		for (int i = 0; i < m_num_lightmap_coordinates; ++i)
			bbox.insert( lightmap_data[i] );

		m_lm_bias  = bbox[0];
		m_lm_scale = bbox[1] - bbox[0];

		m_lightmap_indices_comp.alloc(TEXTURE_TRIANGLE_SIZE * m_num_triangles);

		texture_triangle* lightmap_indices      = reinterpret_cast<texture_triangle*>(m_lightmap_indices.ptr());
		texture_triangle* lightmap_indices_comp = reinterpret_cast<texture_triangle*>(m_lightmap_indices_comp.ptr());

		for (int i = 0; i < m_num_triangles; ++i)
		{
			texture_triangle tri = lightmap_indices[i];

			cugar::Vector2f t0 = tri.x >= 0 ? lightmap_data[tri.x] : cugar::Vector2f(0.0f);
			cugar::Vector2f t1 = tri.y >= 0 ? lightmap_data[tri.y] : cugar::Vector2f(0.0f);
			cugar::Vector2f t2 = tri.z >= 0 ? lightmap_data[tri.z] : cugar::Vector2f(0.0f);

			lightmap_indices_comp[i].x = tri.x >= 0 ? compress_tex_coord(t0, m_lm_bias, m_lm_scale) : -1;
			lightmap_indices_comp[i].y = tri.y >= 0 ? compress_tex_coord(t1, m_lm_bias, m_lm_scale) : -1;
			lightmap_indices_comp[i].z = tri.z >= 0 ? compress_tex_coord(t2, m_lm_bias, m_lm_scale) : -1;
		}
	}
}

void MeshStorage::reorder_triangles(const int* index)
{
    // reorder vertices
    {
        Buffer<int> temp_indices( VERTEX_TRIANGLE_SIZE * m_num_triangles );

        temp_indices.swap(m_vertex_indices);

        vertex_triangle* old_indices = reinterpret_cast<vertex_triangle*>(temp_indices.ptr());
        vertex_triangle* new_indices = reinterpret_cast<vertex_triangle*>(m_vertex_indices.ptr());

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
    // reorder normals
    if (m_normal_indices.ptr())
    {
        Buffer<int> temp_indices( NORMAL_TRIANGLE_SIZE * m_num_triangles );

        temp_indices.swap(m_normal_indices);

        normal_triangle* old_indices = reinterpret_cast<normal_triangle*>(temp_indices.ptr());
        normal_triangle* new_indices = reinterpret_cast<normal_triangle*>(m_normal_indices.ptr());

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
    // reorder textures
    if (m_texture_indices.ptr())
    {
        Buffer<int> temp_indices( TEXTURE_TRIANGLE_SIZE * m_num_triangles );

        temp_indices.swap(m_texture_indices);

        texture_triangle* old_indices = reinterpret_cast<texture_triangle*>(temp_indices.ptr());
        texture_triangle* new_indices = reinterpret_cast<texture_triangle*>(m_texture_indices.ptr());

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
    // reorder materials
    if (m_material_indices.ptr())
    {
        Buffer<int> temp_indices( m_num_triangles );

        temp_indices.swap(m_material_indices);

        int* old_indices = temp_indices.ptr();
        int* new_indices = m_material_indices.ptr();

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
}

void MeshStorage::reset_groups(const int num_groups, const int* group_offsets)
{
    m_num_groups = num_groups;
    m_group_offsets.alloc( num_groups + 1 );

    int* new_indices = m_group_offsets.ptr();
    for (int i = 0; i <= num_groups; ++i)
        new_indices[i] = group_offsets[i];
}

void translate_group(
	MeshStorage&	mesh,
	const uint32	group_id,
	const float3	delta)
{
	if (group_id < (uint32)mesh.getNumGroups())
	{
		const uint32 begin	= mesh.getGroupOffsets()[group_id];
		const uint32 end	= mesh.getGroupOffsets()[group_id + 1];

		std::set<int> marked_vertices;

		for (uint32 tri_id = begin; tri_id < end; ++tri_id)
		{
			for (uint32 i = 0; i < 3; ++i)
			{
				const uint32 vertex_id = mesh.getVertexIndices()[tri_id * MeshStorage::VERTEX_TRIANGLE_SIZE + i];
				if (marked_vertices.find(vertex_id) == marked_vertices.end())
				{
					float3* v = reinterpret_cast<float3*>(mesh.getVertexData()) + vertex_id;
					v->x += delta.x;
					v->y += delta.y;
					v->z += delta.z;
					marked_vertices.insert(vertex_id);
				}
			}
		}
	}
}

// apply material flags
//
SUTILAPI void apply_material_flags(MeshStorage& mesh)
{
	const int* material_indices = mesh.getMaterialIndices();
	MeshStorage::vertex_triangle* tris = reinterpret_cast<MeshStorage::vertex_triangle*>(mesh.getVertexIndices());

	for (int t = 0; t < mesh.getNumTriangles(); ++t)
	{
		const int material_index = material_indices[t];
		if (material_index > -1)
		{
			const MeshMaterial& material = mesh.m_materials[material_index];

			tris[t].w = material.flags;
		}
	}
}

// add per-triangle normals
//
void add_per_triangle_normals(MeshStorage& mesh)
{
	const int num_triangles = mesh.m_num_triangles;

	mesh.m_num_normals = num_triangles;
	
    mesh.m_normal_indices.alloc(4 * num_triangles);
	mesh.m_normal_data.alloc(mesh.m_normal_stride * num_triangles);

	int* normal_indices = mesh.m_normal_indices.ptr();

	MeshView mesh_view = mesh.view();

	for (int tri_id = 0; tri_id < num_triangles; ++tri_id)
	{
		normal_indices[tri_id*MeshStorage::NORMAL_TRIANGLE_SIZE + 0] = tri_id;
		normal_indices[tri_id*MeshStorage::NORMAL_TRIANGLE_SIZE + 1] = tri_id;
		normal_indices[tri_id*MeshStorage::NORMAL_TRIANGLE_SIZE + 2] = tri_id;
		//normal_indices[tri_id*MeshStorage::NORMAL_TRIANGLE_SIZE + 3] = 0;

		// fetch triangle vertices
		const MeshStorage::vertex_triangle tri = reinterpret_cast<const MeshStorage::vertex_triangle*>(mesh.getVertexIndices())[tri_id];
		const cugar::Vector3f vp0 = cugar::Vector4f( fetch_vertex( mesh_view, tri.x ) ).xyz();
		const cugar::Vector3f vp1 = cugar::Vector4f( fetch_vertex( mesh_view, tri.y ) ).xyz();
		const cugar::Vector3f vp2 = cugar::Vector4f( fetch_vertex( mesh_view, tri.z ) ).xyz();

		// compute the geometric normal
		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		*reinterpret_cast<float3*>(mesh.getNormalData() + tri_id*mesh.m_normal_stride) = cugar::normalize(cugar::cross(dp_du, dp_dv));
	}
}

// add per-triangle texture coordinates
//
void add_per_triangle_texture_coordinates(MeshStorage& mesh)
{
	const int num_triangles = mesh.m_num_triangles;

	mesh.m_num_texture_coordinates = 3;
	
    mesh.m_texture_indices.alloc(4 * num_triangles);
	mesh.m_texture_data.alloc(mesh.m_texture_stride * mesh.m_num_texture_coordinates);

	int* texture_indices = mesh.m_texture_indices.ptr();

	for (int tri_id = 0; tri_id < num_triangles; ++tri_id)
	{
		texture_indices[tri_id*MeshStorage::TEXTURE_TRIANGLE_SIZE + 0] = 0;
		texture_indices[tri_id*MeshStorage::TEXTURE_TRIANGLE_SIZE + 1] = 1;
		texture_indices[tri_id*MeshStorage::TEXTURE_TRIANGLE_SIZE + 2] = 2;
		//texture_indices[tri_id*MeshStorage::TEXTURE_TRIANGLE_SIZE + 3] = 0;
	}

	// add one texture triangle
	mesh.m_texture_data.set(0 * mesh.m_texture_stride + 0, 0.0f);
	mesh.m_texture_data.set(0 * mesh.m_texture_stride + 1, 0.0f);

	mesh.m_texture_data.set(1 * mesh.m_texture_stride + 0, 1.0f);
	mesh.m_texture_data.set(1 * mesh.m_texture_stride + 1, 0.0f);

	mesh.m_texture_data.set(2 * mesh.m_texture_stride + 0, 0.0f);
	mesh.m_texture_data.set(2 * mesh.m_texture_stride + 1, 1.0f);
}

void merge(MeshStorage& mesh, const MeshStorage& other)
{
	const uint32 num_materials = mesh.getNumMaterials();
	const uint32 num_textures  = mesh.getNumTextures();

	MeshStorage        other_copy;
	const MeshStorage* other_ptr = &other;

	if ((mesh.m_num_normals >  0 && other.m_num_normals == 0) ||
		(mesh.m_num_normals == 0 && other.m_num_normals > 0))
	{
		if (mesh.m_num_normals == 0)
			add_per_triangle_normals( mesh );
		else
		{
			other_copy = other;
			other_ptr  = &other_copy;

			add_per_triangle_normals( other_copy );
		}
	}
	if ((mesh.m_num_texture_coordinates >  0 && other.m_num_texture_coordinates == 0) ||
		(mesh.m_num_texture_coordinates == 0 && other.m_num_texture_coordinates >  0))
	{
		if (mesh.m_num_texture_coordinates == 0)
			add_per_triangle_texture_coordinates( mesh );
		else
		{
			if (other_ptr == &other)
			{
				other_copy = other;
				other_ptr  = &other_copy;
			}

			add_per_triangle_texture_coordinates( other_copy );
		}
	}

	mesh.m_vertex_indices.resize( mesh.m_vertex_indices.count() + other_ptr->m_vertex_indices.count() );
	mesh.m_normal_indices.resize( mesh.m_normal_indices.count() + other_ptr->m_normal_indices.count() );
	mesh.m_texture_indices.resize( mesh.m_texture_indices.count() + other_ptr->m_texture_indices.count() );
	mesh.m_material_indices.resize( mesh.m_material_indices.count() + other_ptr->m_material_indices.count() );

	mesh.m_group_offsets.resize( mesh.m_num_groups + other_ptr->m_num_groups + 1 );
	mesh.m_vertex_data.resize( mesh.m_vertex_data.count() + other_ptr->m_vertex_data.count() );
	mesh.m_normal_data.resize( mesh.m_normal_data.count() + other_ptr->m_normal_data.count() );
	mesh.m_texture_data.resize( mesh.m_texture_data.count() + other_ptr->m_texture_data.count() );

	mesh.m_materials.resize( mesh.m_materials.count() + other_ptr->m_materials.count() );
	mesh.m_material_name_offsets.resize( mesh.m_materials.count() + other_ptr->m_materials.count() );

	for (size_t i = 0; i < other_ptr->m_vertex_indices.count(); ++i)
		mesh.m_vertex_indices.set( mesh.m_num_triangles*MeshStorage::VERTEX_TRIANGLE_SIZE + i, other_ptr->m_vertex_indices[i] + ((i % 4) < 3 ? mesh.m_num_vertices : 0u) );

	for (size_t i = 0; i < other_ptr->m_normal_indices.count(); ++i)
		mesh.m_normal_indices.set( mesh.m_num_triangles*MeshStorage::NORMAL_TRIANGLE_SIZE + i, other_ptr->m_normal_indices[i] + ((i % 4) < 3 ? mesh.m_num_normals : 0u) );

	for (size_t i = 0; i < other_ptr->m_texture_indices.count(); ++i)
		mesh.m_texture_indices.set( mesh.m_num_triangles*MeshStorage::TEXTURE_TRIANGLE_SIZE + i, other_ptr->m_texture_indices[i] + ((i % 4) < 3 ? mesh.m_num_texture_coordinates : 0u) );

	for (size_t i = 0; i < other_ptr->m_material_indices.count(); ++i)
		mesh.m_material_indices.set( mesh.m_num_triangles + i, other_ptr->m_material_indices[i] + num_materials );

	mesh.m_vertex_data.copy_from( other_ptr->m_vertex_data.count(),		other_ptr->m_vertex_data.type(),	other_ptr->m_vertex_data.ptr(),		mesh.m_num_vertices				* mesh.m_vertex_stride );
	mesh.m_normal_data.copy_from( other_ptr->m_normal_data.count(),		other_ptr->m_normal_data.type(),	other_ptr->m_normal_data.ptr(),		mesh.m_num_normals				* mesh.m_normal_stride );
	mesh.m_texture_data.copy_from( other_ptr->m_texture_data.count(),	other_ptr->m_texture_data.type(),	other_ptr->m_texture_data.ptr(),	mesh.m_num_texture_coordinates	* mesh.m_texture_stride );

	for (size_t i = 0; i <= other_ptr->m_num_groups; ++i)
		mesh.m_group_offsets.set( mesh.m_num_groups + i, mesh.m_num_triangles + other_ptr->m_group_offsets[i] );

	mesh.m_num_vertices				+= other_ptr->m_num_vertices;
	mesh.m_num_normals				+= other_ptr->m_num_normals;
	mesh.m_num_texture_coordinates	+= other_ptr->m_num_texture_coordinates;
	mesh.m_num_triangles			+= other_ptr->m_num_triangles;
	mesh.m_num_groups				+= other_ptr->m_num_groups;

	mesh.m_group_names.insert( mesh.m_group_names.end(), other_ptr->m_group_names.begin(), other_ptr->m_group_names.end() );

	//mesh.m_textures.insert( mesh.m_textures.end(), other_ptr->m_textures.begin(), other_ptr->m_textures.end() );
	//for (std::map<std::string,uint32>::const_iterator it = other_ptr->m_textures_map.begin(); it != other_ptr->m_textures_map.end(); ++it)
	//	mesh.m_textures_map.insert( std::make_pair( it->first, num_textures + it->second ) );

	for (uint32 i = 0; i < (uint32)other_ptr->getNumMaterials(); ++i)
	{
		MeshMaterial material = other_ptr->m_materials[i];

		// insert & relink textures
		if (material.ambient_map.is_valid())		material.ambient_map.texture		= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.ambient_map.texture] );
		if (material.diffuse_map.is_valid())		material.diffuse_map.texture		= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.diffuse_map.texture] );
		if (material.diffuse_trans_map.is_valid())	material.diffuse_trans_map.texture	= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.diffuse_trans_map.texture] );
		if (material.specular_map.is_valid())		material.specular_map.texture		= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.specular_map.texture] );
		if (material.emissive_map.is_valid())		material.emissive_map.texture		= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.emissive_map.texture] );
		if (material.bump_map.is_valid())			material.bump_map.texture			= insert_texture( mesh.m_textures_map, mesh.m_textures, other_ptr->m_textures[material.bump_map.texture] );

		mesh.m_materials.set( num_materials + i, material );
	}

	// merge material names
	{
		const uint32 names_offset = (uint32)mesh.m_material_names.count();

		mesh.m_material_names.resize( names_offset + other_ptr->m_material_names.count() );
		memcpy( mesh.m_material_names.ptr() + names_offset, other_ptr->m_material_names.ptr(), other_ptr->m_material_names.count() );

		for (uint32 i = 0; i < (uint32)other_ptr->getNumMaterials(); ++i)
			mesh.m_material_name_offsets.set( num_materials + i, other_ptr->m_material_name_offsets[i] + names_offset );
	}
}

void transform(
	MeshStorage&	mesh,
	const float		mat[4*4])
{
	cugar::Matrix4x4f M( mat );
	cugar::Matrix4x4f N;

	cugar::invert(M,N);
	N = cugar::transpose(N);

	for (int i = 0; i < mesh.m_num_vertices; ++i)
		*reinterpret_cast<cugar::Vector3f*>(mesh.m_vertex_data.ptr() + mesh.m_vertex_stride*i) = cugar::ptrans( M, *reinterpret_cast<cugar::Vector3f*>(mesh.m_vertex_data.ptr() + mesh.m_vertex_stride*i) );

	for (int i = 0; i < mesh.m_num_normals; ++i)
		*reinterpret_cast<cugar::Vector3f*>(mesh.m_normal_data.ptr() + mesh.m_normal_stride*i) = cugar::vtrans( N, *reinterpret_cast<cugar::Vector3f*>(mesh.m_normal_data.ptr() + mesh.m_normal_stride*i) );
}

namespace {

	int normal_index(const int t, const int n_i)
	{
		return n_i >= 0 ? n_i : -t - 1;
	}

}

// unify all vertex attributes, so as to have a single triplet of indices per triangle
//
void unify_vertex_attributes(MeshStorage& mesh)
{
	typedef std::tuple<int, int, int, int>		attribute_index;
	typedef std::map<attribute_index, uint32>	attribute_map;

	attribute_map map;

	std::vector<attribute_index> vertices;
	uint32 vertex_count = 0;

	int4* v_indices = reinterpret_cast<int4*>( mesh.getVertexIndices() );
	int4* n_indices = reinterpret_cast<int4*>( mesh.getNormalIndices() );
	int4* t_indices = reinterpret_cast<int4*>( mesh.getTextureCoordinateIndices() );
	int4* l_indices = reinterpret_cast<int4*>( mesh.getLightmapIndices() );

	for (int t = 0; t < mesh.getNumTriangles(); ++t)
	{
		int4 null_vertex = make_int4(-1, -1, -1, -1);

		int4 v_tri = v_indices[t];
		int4 n_tri = n_indices ? n_indices[t] : null_vertex;
		int4 t_tri = t_indices ? t_indices[t] : null_vertex;
		int4 l_tri = l_indices ? l_indices[t] : null_vertex;

		// add the first vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.x, normal_index( t, n_tri.x ), t_tri.x, l_tri.x );
			attribute_map::iterator it = map.find( vertex );

			if (it == map.end())
			{
				map.insert( std::make_pair( vertex, vertex_count++ ) );

				vertices.push_back( vertex );
			}
		}
		// add the second vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.y, normal_index( t, n_tri.y ), t_tri.y, l_tri.y );
			attribute_map::iterator it = map.find( vertex );

			if (it == map.end())
			{
				map.insert( std::make_pair( vertex, vertex_count++ ) );

				vertices.push_back( vertex );
			}
		}
		// add the third vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.z, normal_index( t, n_tri.z ), t_tri.z, l_tri.z );
			attribute_map::iterator it = map.find( vertex );

			if (it == map.end())
			{
				map.insert( std::make_pair( vertex, vertex_count++ ) );

				vertices.push_back( vertex );
			}
		}
	}

	Buffer<float>	vertex_data( vertex_count * (sizeof(MeshView::vertex_type)/4) );
	Buffer<float>	normal_data( vertex_count * (sizeof(MeshView::normal_type)/4) );
	Buffer<uint32>	normal_data_comp( vertex_count );
	Buffer<float>	texture_data( vertex_count * (sizeof(MeshView::texture_coord_type)/4) );
	Buffer<float>	lightmap_data( vertex_count * (sizeof(MeshView::texture_coord_type)/4) );

	MeshView::vertex_type* dst_vertex_data = reinterpret_cast<MeshView::vertex_type*>( vertex_data.ptr() );
	MeshView::vertex_type* src_vertex_data = reinterpret_cast<MeshView::vertex_type*>( mesh.m_vertex_data.ptr() );

	MeshView::normal_type* dst_normal_data = reinterpret_cast<MeshView::normal_type*>( normal_data.ptr() );
	MeshView::normal_type* src_normal_data = reinterpret_cast<MeshView::normal_type*>( mesh.m_normal_data.ptr() );

	uint32* dst_normal_data_comp = reinterpret_cast<uint32*>( normal_data_comp.ptr() );
	uint32* src_normal_data_comp = reinterpret_cast<uint32*>( mesh.m_normal_data_comp.ptr() );

	MeshView::texture_coord_type* dst_texture_data = reinterpret_cast<MeshView::texture_coord_type*>( texture_data.ptr() );
	MeshView::texture_coord_type* src_texture_data = reinterpret_cast<MeshView::texture_coord_type*>( mesh.m_texture_data.ptr() );

	MeshView::texture_coord_type* dst_lightmap_data = reinterpret_cast<MeshView::texture_coord_type*>( lightmap_data.ptr() );
	MeshView::texture_coord_type* src_lightmap_data = reinterpret_cast<MeshView::texture_coord_type*>( mesh.m_lightmap_data.ptr() );

	for (uint32 i = 0; i < vertex_count; ++i)
	{
		const int v_idx = std::get<0>( vertices[i] );
		const int n_idx = std::get<1>( vertices[i] );
		const int t_idx = std::get<2>( vertices[i] );
		const int l_idx = std::get<3>( vertices[i] );

		dst_vertex_data[i] = src_vertex_data[ v_idx ];

		if (t_idx >= 0) dst_texture_data[i]  = src_texture_data[ t_idx ];
		if (l_idx >= 0) dst_lightmap_data[i] = src_lightmap_data[ t_idx ]; else dst_lightmap_data[i] = cugar::Vector2f(0.0f);

		if (n_idx >= 0)
		{
			dst_normal_data[i]		= src_normal_data[ n_idx ];
			dst_normal_data_comp[i] = src_normal_data_comp[ n_idx ];

			// pack the compressed normal in the .w component of the vertex data
			dst_vertex_data[i].w = cugar::binary_cast<float>( dst_normal_data_comp[i] );
		}
		else
		{
			// decode the triangle index
			const uint32 t = -n_idx - 1;
			
			// fetch the triangle
			const int4 tri = v_indices[t];
			
			// calculate the triangle normal
			const cugar::Vector3f vp0 = cugar::Vector4f( src_vertex_data[ tri.x ] ).xyz();
			const cugar::Vector3f vp1 = cugar::Vector4f( src_vertex_data[ tri.y ] ).xyz();
			const cugar::Vector3f vp2 = cugar::Vector4f( src_vertex_data[ tri.z ] ).xyz();

			const cugar::Vector3f dp_du = vp0 - vp2;
			const cugar::Vector3f dp_dv = vp1 - vp2;
			const cugar::Vector3f Ng = cugar::normalize(cugar::cross(dp_du, dp_dv));

			// write it out
			dst_normal_data[i] = Ng;

			// compress it
			dst_normal_data_comp[i] = cugar::pack_normal( Ng );

			// pack the compressed normal in the .w component of the vertex data
			dst_vertex_data[i].w = cugar::binary_cast<float>( dst_normal_data_comp[i] );

			// we'd need to pack the triangle normal here... but we just leave a -1 instead
			//dst_vertex_data[i].w = cugar::binary_cast<float>(-1);
		}
	}

	// replace the original vectors
	mesh.m_vertex_data.swap( vertex_data );
	mesh.m_normal_data.swap( normal_data );
	mesh.m_normal_data_comp.swap( normal_data_comp );
	mesh.m_texture_data.swap( texture_data );
	if (mesh.m_lightmap_data.ptr())
		mesh.m_lightmap_data.swap( lightmap_data );

	// reset the vertex attribute counters
	mesh.m_num_vertices				= vertex_count;
	mesh.m_num_normals				= vertex_count;
	mesh.m_num_texture_coordinates	= vertex_count;
	if (mesh.m_num_lightmap_coordinates)
		mesh.m_num_lightmap_coordinates = vertex_count;

	// replace the triangle indices
	for (int t = 0; t < mesh.getNumTriangles(); ++t)
	{
		int4 null_vertex = make_int4(-1, -1, -1, -1);

		int4& v_tri = v_indices[t];
		int4& n_tri = n_indices ? n_indices[t] : null_vertex;
		int4& t_tri = t_indices ? t_indices[t] : null_vertex;
		int4& l_tri = l_indices ? l_indices[t] : null_vertex;

		// add the first vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.x, normal_index( t, n_tri.x ), t_tri.x, l_tri.x );
			attribute_map::iterator it = map.find( vertex );

			v_tri.x = it->second;
			n_tri.x = it->second;
			t_tri.x = it->second;
			l_tri.x = it->second;
		}
		// add the second vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.y, normal_index( t, n_tri.y ), t_tri.y, l_tri.y );
			attribute_map::iterator it = map.find( vertex );

			v_tri.y = it->second;
			n_tri.y = it->second;
			t_tri.y = it->second;
			l_tri.y = it->second;
		}
		// add the third vertex
		{
			attribute_index vertex = std::make_tuple( v_tri.z, normal_index( t, n_tri.z ), t_tri.z, l_tri.z );
			attribute_map::iterator it = map.find( vertex );

			v_tri.z = it->second;
			n_tri.z = it->second;
			t_tri.z = it->second;
			l_tri.z = it->second;
		}
	}
}