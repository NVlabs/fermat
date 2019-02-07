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

#include <pbrt_importer.h>
#include <mesh/MeshStorage.h>
#include <camera.h>
#include <lights.h>
#include <files.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <vector>
#include <stack>
#include <string>
#include <map>

namespace pbrt {

void make_sphere(MeshStorage& other, const float radius, const bool inner_normals = false);

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
cugar::Vector3f FresnelConductor(
	float cosThetaI,
	const cugar::Vector3f &etai,
    const cugar::Vector3f &etat,
	const cugar::Vector3f &k)
{
	cosThetaI = cugar::min(cugar::max(cosThetaI, -1.0f), 1.0f);
	cugar::Vector3f eta = etat / etai;
	cugar::Vector3f etak = k / etai;

	float cosThetaI2 = cosThetaI * cosThetaI;
	float sinThetaI2 = 1.f - cosThetaI2;
	cugar::Vector3f eta2 = eta * eta;
	cugar::Vector3f etak2 = etak * etak;

	cugar::Vector3f t0 = eta2 - etak2 - sinThetaI2;
	cugar::Vector3f a2plusb2 = cugar::sqrt(t0 * t0 + 4.0f * eta2 * etak2);
	cugar::Vector3f t1 = a2plusb2 + cosThetaI2;
	cugar::Vector3f a = cugar::sqrt(0.5f * (a2plusb2 + t0));
	cugar::Vector3f t2 = (float)2 * cosThetaI * a;
	cugar::Vector3f Rs = (t1 - t2) / (t1 + t2);

	cugar::Vector3f t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
	cugar::Vector3f t4 = t2 * sinThetaI2;
	cugar::Vector3f Rp = Rs * (t3 - t4) / (t3 + t4);

	return 0.5f * (Rp + Rs);
}

FermatImporter::FermatImporter(const char* filename, MeshStorage* mesh, Camera* camera, std::vector<DirectionalLight>* dir_lights, std::vector<std::string>* scene_dirs) : 
	m_mesh(mesh), m_camera(camera), m_dir_lights(dir_lights), m_dirs(*scene_dirs)
{
	// initialize the transform stack
	m_transform_stack.push(cugar::Matrix4x4f::one());

	// initialize the material stack
	m_material_stack.push(-1);

	// initialize the light source stack
	m_emission_stack.push( cugar::Vector3f(0.0f) );

	m_default_material = -1;

	// set some camera defaults
	m_camera->eye = cugar::Vector3f(0, 0, 0);
	m_camera->aim = cugar::Vector3f(0, 0, 1);
	m_camera->up  = cugar::Vector3f(0, 1, 0);
	m_camera->dx  = cugar::Vector3f(1, 0, 0);

	char complete_name[4096];
	strcpy(complete_name, filename);

	if (!find_file(complete_name, m_dirs))
	{
		char error_string[1024];
		sprintf(error_string, "unable to find file \"%s\"", filename);
		throw MeshException(error_string);
	}

	// add the local path of this scene to the scene directory
	{
		char local_path[2048];
		extract_path(complete_name, local_path);
		m_dirs.push_back(local_path);
	}
}

FermatImporter::~FermatImporter()
{
}

void FermatImporter::identity()
{
	m_transform_stack.top() = cugar::Matrix4x4f::one();
}
void FermatImporter::transform(const Value& floats)
{
	m_transform_stack.top() = cugar::transpose( cugar::Matrix4x4f( floats.get_floats() ) ) * m_transform_stack.top();
}
void FermatImporter::rotate(const float angle, const float x, const float y, const float z)
{
	m_transform_stack.top() = cugar::rotation_around_axis( angle * M_PIf/180.0f, cugar::Vector3f(x,y,z) ) * m_transform_stack.top();
}
void FermatImporter::scale(const float x, const float y, const float z)
{
	cugar::Matrix4x4f m = cugar::Matrix4x4f::one();
	m[0][0] *= x;
	m[1][1] *= y;
	m[2][2] *= z;

	m_transform_stack.top() = m * m_transform_stack.top();
}
void FermatImporter::translate(const float x, const float y, const float z)
{
	m_transform_stack.top() = cugar::translate( cugar::Vector3f(x,y,z) ) * m_transform_stack.top();
}
void FermatImporter::look_at(
	const float ex, const float ey, const float ez,
	const float lx, const float ly, const float lz,
	const float ux, const float uy, const float uz)
{
	// TODO: make this into a transform
	m_camera->eye = make_float3(ex, ey, ez);
	m_camera->aim = make_float3(lx, ly, lz);
	m_camera->up  = make_float3(ux, ly, uy);
}

void FermatImporter::film(const char* name, const ParameterList& params)
{
	for (size_t i = 0; i < params.names.size(); ++i)
	{
		if (params.names[i] == "exposure" && params.values[i].type == FLOAT_TYPE)
			m_film.exposure = params.values[i].get_float(0);
		else if (params.names[i] == "gamma" && params.values[i].type == FLOAT_TYPE)
			m_film.gamma = params.values[i].get_float(0);
	}
}

void FermatImporter::camera(const char* name, const ParameterList& params)
{
	// take the current transform and apply it to the camera
	cugar::Matrix4x4f M;
	cugar::invert( m_transform_stack.top(), M );
	m_camera->eye = cugar::ptrans( M, cugar::Vector3f(0.0f, 0.0f, 0.0f) );
	m_camera->aim = cugar::ptrans( M, cugar::Vector3f(0.0f, 0.0f, 1.0f) );
	m_camera->up  = cugar::vtrans( M, cugar::Vector3f(0.0f, 1.0f, 0.0f) );
	m_camera->dx  = cugar::vtrans( M, cugar::Vector3f(1.0f, 0.0f, 0.0f) );

	for (size_t i = 0; i < params.names.size(); ++i)
	{
		if (params.names[i] == "fov")
			m_camera->fov = params.values[i].get_float(0) * M_PIf / 180.0f;
	}
}

void FermatImporter::world_begin() 
{
	m_transform_stack.push( cugar::Matrix4x4f::one() );
}
void FermatImporter::world_end()
{
	m_transform_stack.pop();
}

void FermatImporter::attribute_begin()
{
	m_material_stack.push( m_material_stack.top() );
	m_emission_stack.push( m_emission_stack.top() );
}
void FermatImporter::attribute_end()
{
	m_material_stack.pop();
	m_default_material = m_material_stack.top();
	m_emission_stack.pop();
}

void FermatImporter::transform_begin()
{
	m_transform_stack.push( m_transform_stack.top() );
}
void FermatImporter::transform_end()
{
	m_transform_stack.pop();
}

void FermatImporter::texture(const char* name, const char* texel_type, const char* texture_type, const ParameterList& params)
{
	std::string filename;

	for (size_t i = 0; i < params.names.size(); ++i)
	{
		if (params.names[i] == "filename")
			filename = params.values[i].get_string(0);
	}

	const uint32 texture_ref = m_mesh->insert_texture( filename );

	m_texture_map.insert( std::make_pair( std::string(name), texture_ref ) );
}

void FermatImporter::make_named_medium(const char* name, const ParameterList& params)
{
}

void FermatImporter::make_named_material(const char* name, const ParameterList& params)
{
	MeshMaterial material;
	build_material( params.values[0].get_string(0).c_str(), params, material );

	m_materials.push_back( material );
	m_material_names.push_back( name );

	const uint32 material_id = uint32(m_materials.size() - 1);
	m_material_map.insert( std::make_pair(std::string(name), material_id ) );
}
void FermatImporter::named_material(const char* name)
{
	std::map<std::string, uint32>::const_iterator it = m_material_map.find( std::string(name) );
	if (it != m_material_map.end())
		m_material_stack.top() = m_default_material = int( it->second );
	else
		fprintf(stderr, "warning: material named \"%s\" not found!\n", name);
}

void FermatImporter::medium_interface(const char* name1, const char* name2)
{
}

void FermatImporter::material(const char* type, const ParameterList& params)
{
	MeshMaterial material;
	build_material( type, params, material );

	m_materials.push_back( material );
	m_material_names.push_back( "" ); // unnamed material

	const int material_id = int(m_materials.size() - 1);
	m_material_stack.top() = m_default_material = material_id;
}

void FermatImporter::area_light_source(const char* type, const ParameterList& params)
{
	// flag the next material for emission
	for (size_t i = 0; i < params.names.size(); ++i)
	{
		if (params.names[i] == "L" && params.values[i].type == RGB_TYPE)
		{
			m_emission_stack.top() = cugar::Vector3f(
				params.values[i].get_float(0),
				params.values[i].get_float(1),
				params.values[i].get_float(2) );
		}
 	}
}

void FermatImporter::light_source(const char* type, const ParameterList& params)
{
	if (strcmp(type, "distant") == 0)
	{
		cugar::Vector3f from(0.0f, 0.0f, 0.0f);
		cugar::Vector3f to(0.0f, 0.0f, 1.0f);

		DirectionalLight light;

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "L" && params.values[i].type == RGB_TYPE)
			{
				light.color = cugar::Vector3f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2) );
			}
			else if (params.names[i] == "from" && params.values[i].type == POINT_TYPE)
			{
				from = cugar::Vector3f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2) );
			}
			else if (params.names[i] == "to" && params.values[i].type == POINT_TYPE)
			{
				to = cugar::Vector3f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2) );
			}
		}

		light.dir = to - from;

		m_dir_lights->push_back( light );
	}
	else if (strcmp(type, "infinite") == 0)
	{
		std::string filename;

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "mapname")
				filename = params.values[i].get_string(0);
		}

		const uint32 texture_ref = m_mesh->insert_texture( filename );
	
		// make a sphere mesh
		float radius = 1.0e6f; // make it very large...

		MeshStorage other;
		make_sphere(other, radius, true);

		cugar::Matrix4x4f M = m_transform_stack.top();
		::transform( other, &M[0][0] );

		const int triangle_offset = m_mesh->getNumTriangles();
		const int num_materials   = m_mesh->getNumMaterials();

		merge( *m_mesh, other );

		// make a new, custom material for this guy
		MeshMaterial material	= MeshMaterial::zero_material();
		material.emissive		= cugar::Vector4f(1.0f);
		material.emissive_map	= texture_ref;
		material.roughness		= 1.0f;
		material.index_of_refraction = 0.0f; // NOTE: set the IOR to zero to signal the suppression of glossy reflections

		m_materials.push_back( material );
		m_material_names.push_back( "" ); // unnamed material

		const int material_id = int(m_materials.size() - 1);

		// replace the default material
		{
			// NOTE: here, we assume that each loaded mesh has a default material placed at index 0, and that merging
			// will cause it to be moved at index 'num_materials (the number of materials in 'mesh before merging).
			for (int i = 0; i < other.getNumTriangles(); ++i)
				m_mesh->m_material_indices.set( triangle_offset + i, material_id );
		}
	}
}

void FermatImporter::shape(const char* type, const ParameterList& params) 
{
	std::string filename;

	for (size_t i = 0; i < params.names.size(); ++i)
	{
		if (params.names[i] == "filename")
			filename = params.values[i].get_string(0);
	}

	if (strcmp(type, "plymesh") == 0)
	{
		MeshStorage other;

		char complete_name[4096];
		strcpy(complete_name, filename.c_str());

		if (!find_file(complete_name, m_dirs))
		{
			char error_string[1024];
			sprintf(error_string, "unable to find file \"%s\"", filename.c_str());
			throw MeshException(error_string);
		}

		loadModel(complete_name, other);

		cugar::Matrix4x4f M = m_transform_stack.top();
		::transform( other, &M[0][0] );

		const int triangle_offset = m_mesh->getNumTriangles();
		const int num_materials   = m_mesh->getNumMaterials();

		merge( *m_mesh, other );

		// replace the default material
		if (m_default_material != -1)
		{
			// NOTE: here, we assume that each loaded mesh has a default material placed at index 0, and that merging
			// will cause it to be moved at index 'num_materials (the number of materials in 'mesh before merging).
			// This is somewhat dirty, and will need to be substantially cleaned up together with the entire (non-)design
			// of the .obj/.ply mesh loader classes.
			for (int i = 0; i < other.getNumTriangles(); ++i)
			{
				//if (m_mesh->m_material_indices[triangle_offset + i] == num_materials)
					m_mesh->m_material_indices.set( triangle_offset + i, m_default_material );
			}
		}
	}
	else if (strcmp(type, "trianglemesh") == 0)
	{
		const uint32* param_indices	= nullptr;
		const float* param_P		= nullptr;
		const float* param_N		= nullptr;
		const float* param_UV		= nullptr;
		uint32 triangle_count		= 0;
		uint32 vertex_count			= 0;

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "indices")
			{
				param_indices = (const uint32*)params.values[i].get_ints();
				triangle_count = uint32( params.values[i].size() / 3 );
			}
			else if (params.names[i] == "P")
			{
				param_P = params.values[i].get_floats();
				vertex_count = uint32( params.values[i].size() / 3 );
			}
			else if (params.names[i] == "N")
			{
				param_N = params.values[i].get_floats();
			}
			else if (params.names[i] == "uv")
			{
				param_UV = params.values[i].get_floats();
			}
		}

		MeshStorage other;
		other.alloc( triangle_count, vertex_count, param_N ? vertex_count : 0u, param_UV ? vertex_count : 0u, 1u );
		{
			MeshStorage::vertex_triangle* triangles = reinterpret_cast<MeshStorage::vertex_triangle*>(other.m_vertex_indices.ptr());
			for (uint32 i = 0; i < triangle_count; ++i)
			{
				triangles[i].x = param_indices[i*3];
				triangles[i].y = param_indices[i*3+1];
				triangles[i].z = param_indices[i*3+2];
				triangles[i].w = 0u;	// triangle flags
			}
		}
		if (param_N)
		{
			MeshStorage::normal_triangle* triangles = reinterpret_cast<MeshStorage::normal_triangle*>(other.m_normal_indices.ptr());
			for (uint32 i = 0; i < triangle_count; ++i)
			{
				triangles[i].x = param_indices[i*3];
				triangles[i].y = param_indices[i*3+1];
				triangles[i].z = param_indices[i*3+2];
			}
		}
		if (param_UV)
		{
			MeshStorage::texture_triangle* triangles = reinterpret_cast<MeshStorage::texture_triangle*>(other.m_texture_indices.ptr());
			for (uint32 i = 0; i < triangle_count; ++i)
			{
				triangles[i].x = param_indices[i*3];
				triangles[i].y = param_indices[i*3+1];
				triangles[i].z = param_indices[i*3+2];
			}
		}

		if (param_P)
		{
			MeshView::vertex_type* vertices = reinterpret_cast<MeshView::vertex_type*>(other.m_vertex_data.ptr());
			for (uint32 i = 0; i < vertex_count; ++i)
			{
				vertices[i].x = param_P[i*3];
				vertices[i].y = param_P[i*3+1];
				vertices[i].z = param_P[i*3+2];
			}
		}
		if (param_N)
		{
			MeshView::normal_type* vertices = reinterpret_cast<MeshView::normal_type*>(other.m_normal_data.ptr());
			for (uint32 i = 0; i < vertex_count; ++i)
			{
				vertices[i].x = param_N[i*3];
				vertices[i].y = param_N[i*3+1];
				vertices[i].z = param_N[i*3+2];
			}
		}
		if (param_UV)
		{
			MeshView::texture_coord_type* vertices = reinterpret_cast<MeshView::texture_coord_type*>(other.m_texture_data.ptr());
			for (uint32 i = 0; i < vertex_count; ++i)
			{
				vertices[i].x = param_UV[i*2];
				vertices[i].y = param_UV[i*2+1];
			}
		}

		// set the group offsets
		other.m_group_offsets[0] = 0;
		other.m_group_offsets[1] = triangle_count;

		// set the group names
		other.m_group_names[0] = "trianglemesh";

		// set the zero-th default material
		other.alloc_materials(1u);
		other.alloc_material_names(0u);
		memset( other.m_material_indices.ptr(), 0u, sizeof(uint32)*triangle_count );

		cugar::Matrix4x4f M = m_transform_stack.top();
		::transform( other, &M[0][0] );

		const int triangle_offset = m_mesh->getNumTriangles();
		const int num_materials   = m_mesh->getNumMaterials();

		merge( *m_mesh, other );

		// replace the default material
		if (m_default_material != -1)
		{
			// NOTE: here, we assume that each loaded mesh has a default material placed at index 0, and that merging
			// will cause it to be moved at index 'num_materials (the number of materials in 'mesh before merging).
			for (int i = 0; i < other.getNumTriangles(); ++i)
			{
				//if (m_mesh->m_material_indices[triangle_offset + i] == num_materials)
				{
					m_mesh->m_material_indices.set( triangle_offset + i, m_default_material );
				}
			}
		}
	}
	else if (strcmp(type, "disk") == 0)
	{
		uint32 triangle_count = 128;	// 128 triangles per disk
		uint32 vertex_count	  = 129;

		float radius = 1.0f;

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "radius")
				radius = params.values[i].get_float(0);
		}

		MeshStorage other;
		other.alloc( triangle_count, vertex_count, 0u, 0u, 1u );
		{
			MeshStorage::vertex_triangle* triangles = reinterpret_cast<MeshStorage::vertex_triangle*>(other.m_vertex_indices.ptr());
			for (uint32 i = 0; i < triangle_count; ++i)
			{
				triangles[i].x = (i+1) % triangle_count;
				triangles[i].y = i;
				triangles[i].z = triangle_count;
				triangles[i].w = 0u;	// triangle flags
			}
		}
		{
			MeshView::vertex_type* vertices = reinterpret_cast<MeshView::vertex_type*>(other.m_vertex_data.ptr());
			const float angle = M_TWO_PIf / float(triangle_count);
			for (uint32 i = 0; i < triangle_count; ++i)
			{
				vertices[i].x = sinf(angle * i) * radius;
				vertices[i].y = 0;
				vertices[i].z = cosf(angle * i) * radius;
			}
			// write the central vertex
			vertices[triangle_count].x = 0;
			vertices[triangle_count].y = 0;
			vertices[triangle_count].z = 0;
		}

		// set the group offsets
		other.m_group_offsets[0] = 0;
		other.m_group_offsets[1] = triangle_count;

		// set the group names
		other.m_group_names[0] = "disk";

		// set the zero-th default material
		other.alloc_materials(1u);
		other.alloc_material_names(0u);
		memset( other.m_material_indices.ptr(), 0u, sizeof(uint32)*triangle_count );

		cugar::Matrix4x4f M = m_transform_stack.top();
		::transform( other, &M[0][0] );

		const int triangle_offset = m_mesh->getNumTriangles();
		const int num_materials   = m_mesh->getNumMaterials();

		merge( *m_mesh, other );

		// replace the default material
		if (m_default_material != -1)
		{
			// NOTE: here, we assume that each loaded mesh has a default material placed at index 0, and that merging
			// will cause it to be moved at index 'num_materials (the number of materials in 'mesh before merging).
			for (int i = 0; i < other.getNumTriangles(); ++i)
			{
				//if (m_mesh->m_material_indices[triangle_offset + i] == num_materials)
					m_mesh->m_material_indices.set( triangle_offset + i, m_default_material );
			}
		}
	}
}

void FermatImporter::finish()
{
	// alloc and add all the materials to our mesh
	m_mesh->alloc_materials( m_materials.size() );

	for (size_t i = 0; i < m_materials.size(); ++i)
		m_mesh->m_materials[i] = m_materials[i];

	// alloc the material names
	uint32 material_name_length = 0;
	for (size_t i = 0; i < m_materials.size(); ++i)
		material_name_length += uint32( m_material_names[i].length() + 1 );

	m_mesh->alloc_material_names( material_name_length );

	// copy the material names
	uint32 material_name_offset = 0;
	for (size_t i = 0; i < m_materials.size(); ++i)
	{
		m_mesh->m_material_name_offsets[i] = material_name_offset;
		strcpy( m_mesh->m_material_names.ptr() + material_name_offset, m_material_names[i].c_str() );

		material_name_offset += uint32( m_material_names[i].length() + 1 );
	}
}

void FermatImporter::build_material(const char* type, const ParameterList& params, MeshMaterial& material)
{
	material.ambient				= cugar::Vector4f(0.0f);
	material.diffuse_trans			= cugar::Vector4f(0.0f);
	material.diffuse				= cugar::Vector4f(0.0f);
	material.emissive				= cugar::Vector4f(m_emission_stack.top(), 0.0f);
	material.specular				= cugar::Vector4f(0.0f);
	material.reflectivity			= cugar::Vector4f(0.0f);
	material.flags					= 0;
	material.index_of_refraction	= 1.0f;
	material.opacity				= 1.0f;
	material.roughness				= 1.0f;

	if (strcmp(type, "matte") == 0)
	{
		material.diffuse = cugar::Vector4f(0.5f);
		//material.index_of_refraction = 0.0f; // NOTE: set the IOR to zero to signal the suppression of glossy reflections

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "Kd" && params.values[i].type == RGB_TYPE)
			{
				material.diffuse = cugar::Vector4f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2),
					0.0f
				);
			}
			else if (params.names[i] == "Kd" && params.values[i].type == STRING_TYPE)
			{
				texture_map_type::const_iterator texture_it = m_texture_map.find( params.values[i].get_string(0) );
				if (texture_it == m_texture_map.end())
					fprintf(stderr, "warning: texture \"%s\" not found!", params.values[i].get_string(0).c_str());
				else
					material.diffuse_map = TextureReference( texture_it->second );
			}
		}
	}
	else if (strcmp(type, "substrate") == 0)
	{
		float uroughness = 0.1f;
		float vroughness = 0.1f;

		material.diffuse  = cugar::Vector4f(0.5f);
		material.specular = cugar::Vector4f(0.5f);

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "Kd" && params.values[i].type == RGB_TYPE)
			{
				material.diffuse = cugar::Vector4f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2),
					0.0f
				);
			}
			else if (params.names[i] == "Kd" && params.values[i].type == STRING_TYPE)
			{
				texture_map_type::const_iterator texture_it = m_texture_map.find( params.values[i].get_string(0) );
				if (texture_it == m_texture_map.end())
					fprintf(stderr, "warning: texture \"%s\" not found!", params.values[i].get_string(0).c_str());
				else
					material.diffuse_map = TextureReference( texture_it->second );
			}
			else if (params.names[i] == "Ks" && params.values[i].type == RGB_TYPE)
			{
				material.specular = cugar::Vector4f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2),
					0.0f
				);
			}
			else if (params.names[i] == "Ks" && params.values[i].type == STRING_TYPE)
			{
				texture_map_type::const_iterator texture_it = m_texture_map.find( params.values[i].get_string(0) );
				if (texture_it == m_texture_map.end())
					fprintf(stderr, "warning: texture \"%s\" not found!", params.values[i].get_string(0).c_str());
				else
					material.specular_map = TextureReference( texture_it->second );
			}
			else if (params.names[i] == "uroughness" && params.values[i].type == FLOAT_TYPE)
			{
				uroughness = params.values[i].get_float(0);
			}
			else if (params.names[i] == "vroughness" && params.values[i].type == FLOAT_TYPE)
			{
				vroughness = params.values[i].get_float(0);
			}
			else if ((params.names[i] == "eta"   && params.values[i].type == FLOAT_TYPE) ||
					 (params.names[i] == "index" && params.values[i].type == FLOAT_TYPE))
			{
				material.index_of_refraction = params.values[i].get_float(0);
			}
		}
		material.roughness = (uroughness + vroughness) / 2; // TODO: add anisotropic materials support!
	}
	else if (strcmp(type, "glass") == 0)
	{
		float uroughness = 0.00001f;
		float vroughness = 0.00001f;

		material.opacity = 0.0f;
		material.specular = cugar::Vector4f(1.0f);

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "Kt" && params.values[i].type == RGB_TYPE)
			{
				material.opacity = 
					(params.values[i].get_float(0) +
						params.values[i].get_float(1) +
						params.values[i].get_float(2)) / 3.0f; // TODO: allow colored transparency!
			}
			else if (params.names[i] == "Kr" && params.values[i].type == RGB_TYPE)
			{
				material.specular = cugar::Vector4f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2),
					0.0f
				);
			}
			else if (params.names[i] == "Kr" && params.values[i].type == STRING_TYPE)
			{
				texture_map_type::const_iterator texture_it = m_texture_map.find( params.values[i].get_string(0) );
				if (texture_it == m_texture_map.end())
					fprintf(stderr, "warning: texture \"%s\" not found!", params.values[i].get_string(0).c_str());
				else
					material.specular_map = TextureReference( texture_it->second );
			}
			else if ((params.names[i] == "eta"   && params.values[i].type == FLOAT_TYPE) ||
					 (params.names[i] == "index" && params.values[i].type == FLOAT_TYPE))
			{
				material.index_of_refraction = params.values[i].get_float(0);
			}
			else if (params.names[i] == "uroughness" && params.values[i].type == FLOAT_TYPE)
			{
				uroughness = params.values[i].get_float(0);
			}
			else if (params.names[i] == "vroughness" && params.values[i].type == FLOAT_TYPE)
			{
				vroughness = params.values[i].get_float(0);
			}
		}
		material.roughness = (uroughness + vroughness) / 2; // TODO: add anisotropic materials support!
	}
	else if (strcmp(type, "metal") == 0)
	{
		float uroughness = 0.01f;
		float vroughness = 0.01f;

		cugar::Vector3f eta( 0.265787f, 0.195610f, 0.220920f );
		cugar::Vector3f k( 3.540174f, 2.311131f, 1.668593f );

		material.opacity = 1.0f;

		for (size_t i = 0; i < params.names.size(); ++i)
		{
			if (params.names[i] == "eta" && params.values[i].type == RGB_TYPE)
			{
				eta = cugar::Vector3f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2) );
			}
			else if (params.names[i] == "k" && params.values[i].type == RGB_TYPE)
			{
				k = cugar::Vector3f(
					params.values[i].get_float(0),
					params.values[i].get_float(1),
					params.values[i].get_float(2) );
			}
			else if (params.names[i] == "roughness" && params.values[i].type == FLOAT_TYPE)
			{
				uroughness = vroughness = params.values[i].get_float(0);
			}
			else if (params.names[i] == "uroughness" && params.values[i].type == FLOAT_TYPE)
			{
				uroughness = params.values[i].get_float(0);
			}
			else if (params.names[i] == "vroughness" && params.values[i].type == FLOAT_TYPE)
			{
				vroughness = params.values[i].get_float(0);
			}
		}
		material.specular  = cugar::Vector4f( FresnelConductor( 1.0f, cugar::Vector3f(1.0f), eta, k ), 0.0f );
		material.roughness = (uroughness + vroughness) / 2; // TODO: add anisotropic materials support!
	}
}

void make_sphere(MeshStorage& other, const float radius, const bool inner_normals)
{
	// make a sphere
	const uint32 u_subdivs = 256;
	const uint32 v_subdivs = 128;
	uint32 triangle_count = u_subdivs * v_subdivs * 2;
	uint32 vertex_count	  = u_subdivs * (v_subdivs + 1);

	other.alloc( triangle_count, vertex_count, 0u, vertex_count, 1u );
	// write vertex indices
	{
		MeshStorage::vertex_triangle* triangles = reinterpret_cast<MeshStorage::vertex_triangle*>(other.m_vertex_indices.ptr());
		for (uint32 v = 0; v < v_subdivs; ++v)
		{
			for (uint32 u = 0; u < u_subdivs; ++u)
			{
				const uint32 t = (u + v * u_subdivs) * 2;

				const uint32 bu = inner_normals ? 0u : 1u;
				const uint32 iu = inner_normals ? 1u : 0u;

				triangles[t+0].x = ((u+bu) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+0].y = ((u+iu) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+0].z = ((u+ 0) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+0].w = 0u;	// triangle flags

				triangles[t+1].x = ((u+ 1) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+1].y = ((u+iu) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+1].z = ((u+bu) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+1].w = 0u;	// triangle flags
			}
		}
	}
	// write texture indices
	{
		MeshStorage::texture_triangle* triangles = reinterpret_cast<MeshStorage::texture_triangle*>(other.m_texture_indices.ptr());
		for (uint32 v = 0; v < v_subdivs; ++v)
		{
			for (uint32 u = 0; u < u_subdivs; ++u)
			{
				const uint32 t = (u + v * u_subdivs) * 2;

				const uint32 bu = inner_normals ? 0u : 1u;
				const uint32 iu = inner_normals ? 1u : 0u;

				triangles[t+0].x = ((u+bu) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+0].y = ((u+iu) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+0].z = ((u+ 0) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+0].w = 0u;	// triangle flags

				triangles[t+1].x = ((u+ 1) % u_subdivs) + ((v+0) * u_subdivs);
				triangles[t+1].y = ((u+iu) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+1].z = ((u+bu) % u_subdivs) + ((v+1) * u_subdivs);
				triangles[t+1].w = 0u;	// triangle flags
			}
		}
	}
	// write vertex data
	{
		MeshView::vertex_type* vertices = reinterpret_cast<MeshView::vertex_type*>(other.m_vertex_data.ptr());
		const float phi_delta   = M_TWO_PIf / float(u_subdivs);
		const float theta_delta = M_PIf / float(v_subdivs);
		for (uint32 v = 0; v <= v_subdivs; ++v)
		{
			for (uint32 u = 0; u < u_subdivs; ++u)
			{
				const uint32 t = (u + v * u_subdivs);

				const float phi   = u * phi_delta;
				const float theta = M_PIf - v * theta_delta;

				const cugar::Vector3f p(
					cosf(phi)*sinf(theta),
					sinf(phi)*sinf(theta),
					cosf(theta) );

				// write the vertex
				vertices[t].x = p.x * radius;
				vertices[t].y = p.y * radius;
				vertices[t].z = p.z * radius;
			}
		}
	}
	// write texture data
	{
		MeshView::texture_coord_type* vertices = reinterpret_cast<MeshView::texture_coord_type*>(other.m_texture_data.ptr());
		for (uint32 v = 0; v <= v_subdivs; ++v)
		{
			for (uint32 u = 0; u < u_subdivs; ++u)
			{
				const uint32 t = (u + v * u_subdivs);

				// write the vertex
				vertices[t].x = float(u) / float(u_subdivs);
				vertices[t].y = 1.0f - float(v) / float(v_subdivs);
			}
		}
	}

	// set the group offsets
	other.m_group_offsets[0] = 0;
	other.m_group_offsets[1] = triangle_count;

	// set the group names
	other.m_group_names[0] = "sphere";

	// set the zero-th default material
	other.alloc_materials(1u);
	other.alloc_material_names(0u);
	memset( other.m_material_indices.ptr(), 0u, sizeof(uint32)*triangle_count );
}

} // namespace pbrt
