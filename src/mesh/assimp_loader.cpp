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

#include "MeshStorage.h"
#include <cugar/linalg/matrix.h>
#include <files.h>
#include <stack>
#include <string>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

inline float4 color_to_float4(aiColor3D col) { return make_float4( col.r, col.g, col.b, 0.0f ); }

MeshMaterial convert(aiMaterial* material, MeshStorage& out_mesh)
{
	// fetch all relevant properties
	aiColor3D diffuse(0.f,0.f,0.f);
	aiColor3D emissive(0.f,0.f,0.f);
	aiColor3D specular(0.f,0.f,0.f);
	aiColor3D transparent(0.f,0.f,0.f);
	float     opacity = 1.0f;
	float	  shininess = 1.0f;
	float	  shininess_strength = 1.0f;
	float	  ior = 1.0f;
	material->Get( AI_MATKEY_COLOR_DIFFUSE, diffuse );
	material->Get( AI_MATKEY_COLOR_EMISSIVE, emissive );
	material->Get( AI_MATKEY_COLOR_SPECULAR, specular );
	material->Get( AI_MATKEY_COLOR_TRANSPARENT, transparent );
	material->Get( AI_MATKEY_OPACITY, opacity );
	material->Get( AI_MATKEY_SHININESS, shininess );
	material->Get( AI_MATKEY_SHININESS_STRENGTH, shininess_strength );
	material->Get( AI_MATKEY_REFRACTI, ior );

	aiString diffuse_tex;
	aiString emissive_tex;
	aiString specular_tex;
	aiString shininess_tex;
	material->GetTexture( aiTextureType_DIFFUSE,  0u, &diffuse_tex );
	material->GetTexture( aiTextureType_EMISSIVE, 0u, &emissive_tex );
	material->GetTexture( aiTextureType_SPECULAR, 0u, &specular_tex );
	material->GetTexture( aiTextureType_SHININESS, 0u, &shininess_tex );

	const uint32 diffuse_tex_id  = diffuse_tex  == aiString() ? TextureReference::INVALID : out_mesh.insert_texture( diffuse_tex.C_Str() );
	const uint32 emissive_tex_id = emissive_tex == aiString() ? TextureReference::INVALID : out_mesh.insert_texture( emissive_tex.C_Str() );
	const uint32 specular_tex_id = specular_tex == aiString() ? TextureReference::INVALID : out_mesh.insert_texture( specular_tex.C_Str() );

	// and convert them to our internal material representation
	MeshMaterial out_material;
	out_material.diffuse		= color_to_float4( diffuse );
	out_material.diffuse_map	= TextureReference( diffuse_tex_id );
	out_material.emissive		= color_to_float4( emissive );
	out_material.emissive_map	= TextureReference( emissive_tex_id );
	out_material.specular		= color_to_float4( specular /** shininess_strength*/ );
	out_material.specular_map	= TextureReference( specular_tex_id );
	out_material.opacity		= opacity;
	out_material.diffuse_trans	= color_to_float4( transparent ); // not exactly the right mapping... since in the historical rasterizer world this was a sort of refraction color (typically with ior=1)
	out_material.roughness		= 1.0f / (1.0f + shininess);
	out_material.index_of_refraction = ior;

	// FIXME: this works only as long as there's a single material per mesh!
	aiString name;
	material->Get( AI_MATKEY_NAME, name );
	out_mesh.alloc_material_names( name.length + 1u );
	strcpy( out_mesh.m_material_names.ptr(), name.C_Str() );
	out_mesh.m_material_name_offsets.set( 0, 0 );

  #if 1
	fprintf(stderr, "  material %s\n", name.C_Str());
	fprintf(stderr, "   %u properties\n", material->mNumProperties);
	for (uint32 i = 0; i < material->mNumProperties; ++i)
		fprintf(stderr, "     property %u : %s (semantic : %u)\n", i, material->mProperties[i]->mKey.C_Str(), material->mProperties[i]->mSemantic);
		fprintf(stderr, "   emissive  %f %f %f (%s)\n", emissive.r, emissive.g, emissive.b, emissive_tex.C_Str());
		fprintf(stderr, "   diffuse   %f %f %f (%s)\n", diffuse.r, diffuse.g, diffuse.b, diffuse_tex.C_Str());
		fprintf(stderr, "   specular  %f %f %f (%s)\n", specular.r, specular.g, specular.b, specular_tex.C_Str());
		fprintf(stderr, "   shininess %f (%s)\n", shininess, shininess_tex.C_Str());
		fprintf(stderr, "   strength  %f\n", shininess_strength);
		fprintf(stderr, "   opacity   %f\n", opacity);
		fprintf(stderr, "   ior       %f\n", ior);
  #endif

	return out_material;
}

void load_assimp(const char* filename, MeshStorage& out_mesh, const std::vector<std::string>& dirs, std::vector<std::string>& scene_dirs)
{
	// Create an instance of the Importer class
	Assimp::Importer importer;

	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll
	// probably to request more postprocessing than we do in this example.
	const aiScene* scene = importer.ReadFile( std::string(filename),
		aiProcess_CalcTangentSpace       |
		aiProcess_Triangulate            |
		aiProcess_JoinIdenticalVertices  |
		aiProcess_SortByPType			 |
		aiProcess_GenNormals			 |
		aiProcess_PreTransformVertices);

	// If the import failed, report it
	if( !scene)
		throw MeshException(std::string(importer.GetErrorString()));

	fprintf(stderr, "  %u cameras found\n", scene->mNumCameras);
	fprintf(stderr, "  %u lights found\n", scene->mNumLights);

	/*
	// Now we can access the file's contents.
	out_mesh.alloc_materials( scene->mNumMaterials );

	for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
	{
		aiMaterial* material = scene->mMaterials[i];

		out_mesh.m_materials.set(i, convert(material));
	}
	*/
	for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
	{
		aiMesh* mesh = scene->mMeshes[m];

		// skip unsupported meshes (e.g. points, lines and quads)
		if (mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE)
			continue;

		MeshStorage temp_mesh;

		// alloc all the necessary vertex storage
		temp_mesh.alloc(
			mesh->mNumFaces, 
			mesh->mNumVertices,
			mesh->HasNormals()        ? mesh->mNumVertices : 0u,
			mesh->HasTextureCoords(0) ? mesh->mNumVertices : 0u,
			1u);													// number of groups

		// copy the mesh name
		aiString mesh_name = mesh->mName;
		temp_mesh.getGroupName(0) = mesh_name.C_Str();

		// aiMesh's have always one material
		temp_mesh.alloc_materials( 1u );
		{
			// attach the material
			const uint32 material_idx = mesh->mMaterialIndex;
			aiMaterial* material = scene->mMaterials[material_idx];

			temp_mesh.m_materials.set(0, convert(material, temp_mesh));
		}
		// set the material indices
		memset( temp_mesh.getMaterialIndices(), 0x00, sizeof(uint32)*mesh->mNumFaces );

		// setup (trivial) group offsets
		int* offsets = temp_mesh.getGroupOffsets();
		offsets[0] = 0;
		offsets[1] = mesh->mNumFaces;

		MeshStorage::vertex_triangle* indices = reinterpret_cast<MeshStorage::vertex_triangle*>( temp_mesh.getVertexIndices() );
		float* vertices  = temp_mesh.getVertexData();
		float* normals   = temp_mesh.getNormalData();
		float2* textures = reinterpret_cast<float2*>( temp_mesh.getTextureCoordinateData() );

		unsigned int DEFAULT_TRIANGLE_MASK = 0u;

		// copy vertex indices
		for (uint32 i = 0; i < mesh->mNumFaces; ++i)
		{
			if (mesh->mFaces[i].mNumIndices == 3)
			{
				indices[i].x = mesh->mFaces[i].mIndices[0];
				indices[i].y = mesh->mFaces[i].mIndices[1];
				indices[i].z = mesh->mFaces[i].mIndices[2];
				indices[i].w = DEFAULT_TRIANGLE_MASK;
			}
			else
				throw MeshException(std::string("unsupported mesh face"));
		}

		// TODO: copy normal and texture indices!!

		// copy vertices
		memcpy( vertices, mesh->mVertices, sizeof(float)*3*mesh->mNumVertices );

		// copy normals
		if (mesh->HasNormals())
			memcpy( normals, mesh->mNormals, sizeof(float)*3*mesh->mNumVertices );
		else
			add_per_triangle_normals( temp_mesh );

		// copy texture coords
		if (mesh->HasTextureCoords(0))
		{
			for (uint32 i = 0; i < mesh->mNumVertices; ++i)
			{
				textures[i].x = mesh->mTextureCoords[0][i].x;
				textures[i].y = mesh->mTextureCoords[0][i].y;
			}
		}
		else
			add_per_triangle_texture_coordinates( temp_mesh );

		merge( out_mesh, temp_mesh );
	}
	// We're done. Everything will be cleaned up by the importer destructor
}
