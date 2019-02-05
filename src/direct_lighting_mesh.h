/*
 * Fermat
 *
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <lights.h>

///@addtogroup Fermat
///@{

///@addtogroup PTLib
///@{

/// A model of \ref TPTDirectLightingSampler, implementing NEE sample generation from an emissive mesh
///
struct DirectLightingMesh
{
	static const uint32	INVALID_SLOT	= 0xFFFFFFFF;
	static const uint32	INVALID_SAMPLE	= 0xFFFFFFFF;

	/// empty constructor
	///
	FERMAT_HOST_DEVICE
	DirectLightingMesh() {}

	/// constructor
	///
	FERMAT_HOST_DEVICE
	DirectLightingMesh(const MeshLight _mesh_light) : mesh_light(_mesh_light) {}

	/// preprocess a path vertex and return a hash slot used for NEE
	///
	FERMAT_DEVICE
	uint32 preprocess_vertex(
		const RenderingContextView&	renderer,
		const EyeVertex&	ev,
		const uint32		pixel,
		const uint32		bounce,
		const bool			is_secondary_diffuse,
		const float			cone_radius,
		const cugar::Bbox3f	scene_bbox)
	{
		return INVALID_SLOT;
	}

	/// sample a light vertex at a given slot
	///
	FERMAT_DEVICE
	uint32 sample(
		const uint32		nee_slot,
		const float			z[3],
		VertexGeometryId*	light_vertex,
		VertexGeometry*		light_vertex_geom,
		float*				light_pdf,
		Edf*				light_edf)
	{
		mesh_light.sample(z, &light_vertex->prim_id, &light_vertex->uv, light_vertex_geom, light_pdf, light_edf);
		return INVALID_SAMPLE;
	}

	/// map a light vertex at the slot given at the previous vertex
	///
	FERMAT_DEVICE
	void map(
		const uint32			prev_nee_slot,
		const uint32			triId,
		const cugar::Vector2f	uv,
		const VertexGeometry	light_vertex_geom,
		float*					light_pdf,
		Edf*					light_edf)
	{
		mesh_light.map(triId, uv, light_vertex_geom, light_pdf, light_edf);
	}

	/// update with the resulting NEE sample
	///
	FERMAT_DEVICE
	void update(
		const uint32			nee_slot,
		const uint32			nee_sample,
		const cugar::Vector3f	w,
		const bool				occluded)
	{}

	MeshLight	mesh_light;
};

///@} PTLib
///@} Fermat
