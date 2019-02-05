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

#include <pbrt_parser.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <vector>
#include <stack>
#include <string>
#include <map>

class MeshStorage;
struct MeshMaterial;
struct Camera;
struct DirectionalLight;

namespace pbrt {

struct FilmOptions
{
	FilmOptions() : gamma(2.2f), exposure(1.0f) {}

	float gamma;
	float exposure;
};

struct FermatImporter : public Importer
{
	FermatImporter(const char* filename, MeshStorage* mesh, Camera* camera, std::vector<DirectionalLight>* dir_lights, std::vector<std::string>* scene_dirs);
	~FermatImporter();

	virtual void identity();
	virtual void transform(const Value& floats);
	virtual void rotate(const float angle, const float x, const float y, const float z);
	virtual void scale(const float x, const float y, const float z);
	virtual void translate(const float x, const float y, const float z);
	virtual void look_at(
		const float ex, const float ey, const float ez,
		const float lx, const float ly, const float lz,
		const float ux, const float uy, const float uz);

	virtual void integrator(const char* name, const ParameterList& params) {}
	virtual void sampler(const char* name, const ParameterList& params) {}
	virtual void pixel_filter(const char* name, const ParameterList& params) {}
	virtual void film(const char* name, const ParameterList& params);
	virtual void camera(const char* name, const ParameterList& params);

	virtual void world_begin();
	virtual void world_end();

	virtual void attribute_begin();
	virtual void attribute_end();

	virtual void transform_begin();
	virtual void transform_end();

	virtual void texture(const char* name, const char* texel_type, const char* texture_type, const ParameterList& params);

	virtual void make_named_medium(const char* name, const ParameterList& params);
	virtual void make_named_material(const char* name, const ParameterList& params);
	virtual void named_material(const char* name);
	virtual void medium_interface(const char* name1, const char* name2);
	virtual void material(const char* type, const ParameterList& params);
	virtual void area_light_source(const char* type, const ParameterList& params);
	virtual void light_source(const char* type, const ParameterList& params);

	virtual void shape(const char* type, const ParameterList& params);

	void build_material(const char* type, const ParameterList& params, MeshMaterial& material);
	void finish();


	typedef std::map<std::string, uint32>	texture_map_type;
	typedef std::map<std::string, uint32>	material_map_type;

	FilmOptions						m_film;
	MeshStorage*					m_mesh;
	Camera*							m_camera;
	std::vector<DirectionalLight>*	m_dir_lights;
	std::vector<std::string>&		m_dirs;
	texture_map_type				m_texture_map;
	material_map_type				m_material_map;
	std::vector<MeshMaterial>		m_materials;
	std::vector<std::string>		m_material_names;
	std::stack<cugar::Matrix4x4f>	m_transform_stack;
	std::stack<int>					m_material_stack;
	std::stack<cugar::Vector3f>		m_emission_stack;
	int								m_default_material;
};

} // namespace pbrt
