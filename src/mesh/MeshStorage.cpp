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

#include "MeshStorage.h"
#include "MeshLoader.h"
#include <set>
#include <cugar/linalg/matrix.h>

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

		materials[i].ambient_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.ambient_map);
		materials[i].diffuse_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_map);
		materials[i].diffuse_trans_map	= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.diffuse_trans_map);
		materials[i].specular_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.specular_map);
		materials[i].emissive_map		= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.emissive_map);
		materials[i].bump_map.texture   = TextureReference::INVALID;
		//materials[i].bump_map			= insert_texture(mesh.m_textures_map, mesh.m_textures, texture_count, params.bump_map);
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

void MeshStorage::reorder_triangles(const int* index)
{
    // reorder vertices
    {
        Buffer<int> temp_indices( 3 * m_num_triangles );

        temp_indices.swap(m_vertex_indices);

        int3* old_indices = reinterpret_cast<int3*>(temp_indices.ptr());
        int3* new_indices = reinterpret_cast<int3*>(m_vertex_indices.ptr());

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
    // reorder normals
    if (m_normal_indices.ptr())
    {
        Buffer<int> temp_indices( 3 * m_num_triangles );

        temp_indices.swap(m_normal_indices);

        int3* old_indices = reinterpret_cast<int3*>(temp_indices.ptr());
        int3* new_indices = reinterpret_cast<int3*>(m_normal_indices.ptr());

        for (int i = 0; i < m_num_triangles; ++i)
            new_indices[i] = old_indices[index[i]];
    }
    // reorder textures
    if (m_texture_indices.ptr())
    {
        Buffer<int> temp_indices( 3 * m_num_triangles );

        temp_indices.swap(m_texture_indices);

        int3* old_indices = reinterpret_cast<int3*>(temp_indices.ptr());
        int3* new_indices = reinterpret_cast<int3*>(m_texture_indices.ptr());

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
				const uint32 vertex_id = mesh.getVertexIndices()[tri_id * 3 + i];
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

// add per-triangle normals
//
void add_per_triangle_normals(MeshStorage& mesh)
{
	const int num_triangles = mesh.m_num_triangles;

	mesh.m_num_normals = num_triangles;
	
    mesh.m_normal_indices.alloc(3 * num_triangles);
	mesh.m_normal_data.alloc(mesh.m_normal_stride * num_triangles);

	int* normal_indices = mesh.m_normal_indices.ptr();

	for (int tri_id = 0; tri_id < num_triangles; ++tri_id)
	{
		normal_indices[tri_id*3 + 0] = tri_id;
		normal_indices[tri_id*3 + 1] = tri_id;
		normal_indices[tri_id*3 + 2] = tri_id;

		// fetch triangle vertices
		const int3 tri = reinterpret_cast<const int3*>(mesh.getVertexIndices())[tri_id];
		const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(mesh.getVertexData())[tri.x];
		const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(mesh.getVertexData())[tri.y];
		const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(mesh.getVertexData())[tri.z];

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
	
    mesh.m_texture_indices.alloc(3 * num_triangles);
	mesh.m_texture_data.alloc(mesh.m_texture_stride * mesh.m_num_texture_coordinates);

	int* texture_indices = mesh.m_texture_indices.ptr();

	for (int tri_id = 0; tri_id < num_triangles; ++tri_id)
	{
		texture_indices[tri_id*3 + 0] = 0;
		texture_indices[tri_id*3 + 1] = 1;
		texture_indices[tri_id*3 + 2] = 2;
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
		mesh.m_vertex_indices.set( mesh.m_num_triangles*3 + i, other_ptr->m_vertex_indices[i] + mesh.m_num_vertices );

	for (size_t i = 0; i < other_ptr->m_normal_indices.count(); ++i)
		mesh.m_normal_indices.set( mesh.m_num_triangles*3 + i, other_ptr->m_normal_indices[i] + mesh.m_num_normals );

	for (size_t i = 0; i < other_ptr->m_texture_indices.count(); ++i)
		mesh.m_texture_indices.set( mesh.m_num_triangles*3 + i, other_ptr->m_texture_indices[i] + mesh.m_num_texture_coordinates );

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