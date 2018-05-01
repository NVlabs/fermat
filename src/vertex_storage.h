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

#include <cugar/linalg/vector.h>

///@addtogroup Fermat
///@{

///@addtogroup VertexGeometryModule
///@{

enum class VertexSampling
{
	kAll = 0,
	kEnd = 1
};

enum class VertexOrdering
{
	kRandomOrdering = 0,
	kPathOrdering	= 1
};

struct VertexStorageView
{
	VertexStorageView() :
		vertex(NULL),
		vertex_path_id(NULL),
		vertex_gbuffer(NULL),
		vertex_pos(NULL),
		vertex_input(NULL),
		vertex_weights(NULL),
		vertex_counts(NULL),
		vertex_counter(NULL) {}

	VPL*			vertex;
	uint32*			vertex_path_id;
	uint4*			vertex_gbuffer;
	float4*			vertex_pos;
	uint2*			vertex_input;
	float2*			vertex_weights;
	uint32*			vertex_counts;
	uint32*			vertex_counter;
};

struct VertexStorage
{
	void alloc(const uint32 n_paths, const uint32 n_vertices)
	{
		vertex.alloc(n_vertices);
		vertex_path_id.alloc(n_vertices);
		vertex_pos.alloc(n_vertices);
		vertex_gbuffer.alloc(n_vertices);
		vertex_input.alloc(n_vertices);
		vertex_weights.alloc(n_vertices);
		vertex_counts.alloc(n_paths);
		vertex_counter.alloc(1);
	}

	VertexStorageView view()
	{
		VertexStorageView r;
		r.vertex			= vertex.ptr();
		r.vertex_path_id	= vertex_path_id.ptr();
		r.vertex_pos		= vertex_pos.ptr();
		r.vertex_gbuffer	= vertex_gbuffer.ptr();
		r.vertex_input		= vertex_input.ptr();
		r.vertex_weights	= vertex_weights.ptr();
		r.vertex_counts		= vertex_counts.ptr();
		r.vertex_counter	= vertex_counter.ptr();
		return r;
	}

	DomainBuffer<CUDA_BUFFER, VPL>		vertex;			// VPL descriptor: needed for local surface and BSDF reconstruction
	DomainBuffer<CUDA_BUFFER, uint32>	vertex_path_id;	// vertex path id: (path # | vertex #)
	DomainBuffer<CUDA_BUFFER, float4>	vertex_pos;		// vertex position + normal (useful for tree building)
	DomainBuffer<CUDA_BUFFER, uint4>	vertex_gbuffer;	// bsdf parameters
	DomainBuffer<CUDA_BUFFER, uint2>	vertex_input;	// incident direction / radiance
	DomainBuffer<CUDA_BUFFER, float2>	vertex_weights;	// see 'PathWeights' struct
	DomainBuffer<CUDA_BUFFER, uint32>	vertex_counts;	// vertex counts, by depth
	DomainBuffer<CUDA_BUFFER, uint32>	vertex_counter;	// vertex counts, by depth
};

///@} VertexGeometryModule
///@} Fermat
