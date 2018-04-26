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

#include "MeshStorage.h"
#include <cugar/linalg/matrix.h>
#include <files.h>
#include <stack>


void load_scene(const char* filename, MeshStorage& mesh, const std::vector<std::string>& dirs, std::vector<std::string>& scene_dirs)
{
	char complete_name[2048];
	strcpy(complete_name, filename);

	if (!find_file(complete_name, dirs))
	{
		throw MeshException(std::string("unable to find file: ") + filename);
		return;
	}

	ScopedFile file(complete_name, "r");
	if (!file)
	{
		throw MeshException(std::string("unable to open file: ") + filename);
		return;
	}

	int default_material = -1;

	if (strlen(filename) >= 3 && strcmp(filename + strlen(filename) - 3, ".fa") == 0)
	{
		char command[1024];

		std::stack<cugar::Matrix4x4f> transform_stack;

		transform_stack.push(cugar::Matrix4x4f::one());

		for (;;)
		{
			if (fscanf(file, "%s", command) <= 0)
				break;

			if (command[0] == '#')
			{
				// skip the line
				fgets(command,1024,file);
			}
			else if (strcmp(command, "Begin") == 0)
			{
				transform_stack.push( transform_stack.top() );
			}
			else if (strcmp(command, "End") == 0)
			{
				transform_stack.pop();
			}
			else if (strcmp(command, "Transform") == 0)
			{
				float m[16];
				if (fscanf(file, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
					m, m+1, m+2, m+3, m+4, m+5, m+6, m+7, m+8, m+9, m+10, m+11, m+12, m+13, m+14, m+15) < 16)
					throw MeshException("Transform: insufficient number of arguments");

				transform_stack.top() = cugar::Matrix4x4f( m ) * transform_stack.top();
			}
			else if (strcmp(command, "Translate") == 0)
			{
				cugar::Vector3f t;
				if (fscanf(file, "%f %f %f", &t.x, &t.y, &t.z) < 3)
					throw MeshException("Translate: insufficient number of arguments");

				transform_stack.top() = cugar::translate( t ) * transform_stack.top();
			}
			else if (strcmp(command, "Scale") == 0)
			{
				cugar::Vector3f s;
				if (fscanf(file, "%f %f %f", &s.x, &s.y, &s.z) < 3)
					throw MeshException("Scale: insufficient number of arguments");

				cugar::Matrix4x4f m = cugar::Matrix4x4f::one();
				m[0][0] *= s.x;
				m[1][1] *= s.y;
				m[2][2] *= s.z;

				transform_stack.top() = m * transform_stack.top();
			}
			else if (strcmp(command, "RotateX") == 0)
			{
				float angle;
				fscanf(file, "%f", &angle);

				transform_stack.top() = cugar::rotation_around_X( angle * M_PIf/180.0f ) * transform_stack.top();
			}
			else if (strcmp(command, "RotateY") == 0)
			{
				float angle;
				fscanf(file, "%f", &angle);

				transform_stack.top() = cugar::rotation_around_Y( angle * M_PIf/180.0f ) * transform_stack.top();
			}
			else if (strcmp(command, "RotateZ") == 0)
			{
				float angle;
				fscanf(file, "%f", &angle);

				transform_stack.top() = cugar::rotation_around_Z( angle * M_PIf/180.0f ) * transform_stack.top();
			}
			else if (strcmp(command, "LoadScene") == 0 ||
					 strcmp(command, "LoadMesh") == 0)
			{
				char name[2048];
				fscanf(file, "%s", name);

				if (!find_file(name, dirs))
				{
					char error_string[1024];
					sprintf(error_string, "unable to find file \"%s\"", name);
					throw MeshException(error_string);
				}

				// add the local path of this scene to the scene directory
				{
					char local_path[2048];
					extract_path(name, local_path);
					scene_dirs.push_back(local_path);
				}

				fprintf(stderr, "  loading mesh file %s... started\n", name);
				MeshStorage other;

				load_scene(name, other, dirs, scene_dirs); // recurse
				fprintf(stderr, "  loading mesh file %s... done (%d triangles, %d materials, %d groups)\n", name, other.getNumTriangles(), other.getNumMaterials(), other.getNumGroups());

				transform( other, &transform_stack.top()[0][0] );

				const int triangle_offset = mesh.getNumTriangles();
				const int num_materials   = mesh.getNumMaterials();

				merge( mesh, other );

				// replace the default material
				if (default_material != -1)
				{
					// NOTE: here, we assume that each loaded mesh has a default material placet at index 0, and that merging
					// will cause it to be moved at index 'num_materials (the number of materials in 'mesh before merging).
					// This is somewhat dirty, and will need to be substantially cleaned up together with the entire (non-)design
					// of the .obj/.ply mesh loader classes.
					for (int i = 0; i < other.getNumTriangles(); ++i)
					{
						if (mesh.m_material_indices[triangle_offset + i] == num_materials)
							mesh.m_material_indices.set( triangle_offset + i, default_material );
					}
				}
			}
			else if (strcmp(command, "LoadMaterials") == 0)
			{
				char name[2048];
				fscanf(file, "%s", name);

				if (!find_file(name, dirs))
				{
					char error_string[1024];
					sprintf(error_string, "unable to find file \"%s\"", name);
					throw MeshException(error_string);
				}

				loadMaterials( name, mesh );
			}
			else if (strcmp(command, "SetMaterial") == 0)
			{
				char name[2048];
				fscanf(file, "%s", name);

				// find the material in the current mesh
				for (int i = mesh.getNumMaterials()-1; i >= 0; --i)
				{
					if (strcmp(mesh.getMaterialName(i), name) == 0)
					{
						default_material = i;
						break;
					}
				}
			}
		}
	}
	else
	{
		// let's try with the other loader
		loadModel(complete_name, mesh);
	}
}