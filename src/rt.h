/*
 * Fermat
 *
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <types.h>
#include <ray.h>

#include <internal/optix_declarations.h>

///@addtogroup Fermat
///@{

///@defgroup RTModule Ray-Tracing Module
///  This module defines the core interfaces for ray tracing queries.
///@{

enum ERayGenPrograms
{
	NULL_RAY_GEN					= 0,
    TMIN_INTERSECTION_RAY_GEN		= 1,
	MASKED_INTERSECTION_RAY_GEN		= 2
};

struct RTContextImpl;

/// Class defining core ray tracing functionality, ranging from geometry setup to
/// performing actual ray tracing queries.
///
struct FERMAT_API RTContext
{
	RTContext();
	~RTContext();

	void create_geometry(
		const uint32	triCount,
		const int*		index_ptr,
		const uint32	vertex_count,
		const float*	vertex_ptr,
		const int*		normal_index_ptr,
		const float*	normal_vertex_ptr,
		const int*		tex_index_ptr,
		const float*	tex_vertex_ptr,
		const int*		material_index_ptr);

	void bind_buffer(
		const char*		name,
		const uint32	size,
		const uint32	element_size,
		void*			ptr,
		const RTformat	format);

	void bind_var(
		const char*		name,
		const uint32	size,
		void*			ptr);

	template <typename T>
	void bind_var(const char* name, const T value) { bind_var(name, sizeof(T), (void*)&value); }

	void bind_var(const char* name, const int32 value);
	void bind_var(const char* name, const uint32 value);
	void bind_var(const char* name, const float value);

	uint32 create_program(
		const char*		filename,
		const char*		program_name);

	uint32 add_ray_generation_program(const uint32 program);

	void launch(const uint32 index, const uint32 width);
	void launch(const uint32 index, const uint32 width, const uint32 height);

	void trace(const uint32 count, const Ray* rays, Hit* hits);
	void trace(const uint32 count, const MaskedRay* rays, Hit* hits);
	void trace_shadow(const uint32 count, const MaskedRay* rays, Hit* hits);
	void trace_shadow(const uint32 count, const MaskedRay* rays, uint32* binary_hits);

	RTContextImpl* impl;
};

///@} RTModule
///@} Fermat