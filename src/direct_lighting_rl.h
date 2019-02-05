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
#include <clustered_rl.h>
#include <vtl_mesh_view.h>
#include <spatial_hash.h>

///@addtogroup Fermat
///@{

///@addtogroup PTLib
///@{

/// A model of \ref TPTDirectLightingSampler, implementing NEE sample generation from an emissive mesh using
/// the novel adaptively clustered Reinforcement-Learning algorithm implemented in \ref ClusteredRLModule
///
struct DirectLightingRL
{
	typedef AdaptiveClusteredRLView	RLMap;

	static const uint32	INVALID_SLOT	= 0xFFFFFFFF;
	static const uint32	INVALID_SAMPLE	= 0xFFFFFFFF;

	/// empty constructor
	///
	FERMAT_HOST_DEVICE
	DirectLightingRL() {}

	/// constructor
	///
	FERMAT_HOST_DEVICE
	DirectLightingRL(
		RLMap		_rl,
		VTLMeshView	_vtls) :
		vtl_rl(_rl), vtls(_vtls) {}

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
		// compute a spatial hash
		const float cone_scale   = 32.0f;
		const float filter_scale = is_secondary_diffuse ? 0.2f : 1.5f;

		const uint32 base_dim   = (is_secondary_diffuse ? 0 : renderer.instance) * 6;
		const uint32 random_set = cugar::hash(pixel + renderer.res_x * renderer.res_y * bounce);

		const float jitter[6] = {
			cugar::randfloat( base_dim + 0, random_set ),
			cugar::randfloat( base_dim + 1, random_set ),
			cugar::randfloat( base_dim + 2, random_set ),
			cugar::randfloat( base_dim + 3, random_set ),
			cugar::randfloat( base_dim + 4, random_set ),
			cugar::randfloat( base_dim + 5, random_set ),
		};

		const float bbox_delta = cugar::max_comp( scene_bbox[1] - scene_bbox[0] );

		const uint64 shading_key = spatial_hash(
			pixel,
			ev.geom.position,
			dot(ev.in, ev.geom.normal_s) > 0.0f ? ev.geom.normal_s : -ev.geom.normal_s,
			ev.geom.tangent,
			ev.geom.binormal,
			scene_bbox,
			jitter,
			cugar::min( cone_radius * cone_scale, bbox_delta * 0.05f ),
			//cone_radius * cone_scale,
			filter_scale);

		return vtl_rl.find_slot(shading_key);
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
		// sample the light source surface
		//
		if (nee_slot != INVALID_SLOT)
		{
			uint32 vtl_cluster;

			// 1. use the clustered-RL to sample a VPL
			float  vtl_pdf;
			uint32 vtl_idx = vtl_rl.sample( nee_slot, z[2], &vtl_pdf, &vtl_cluster );

			// 2. sample the selected VTL uniformly
			vtls.sample( vtl_idx, cugar::Vector2f(z[0],z[1]), &light_vertex->prim_id, &light_vertex->uv, light_vertex_geom, light_pdf, light_edf );
			
			// 3. multiply by the RL-sampling pdf
			*light_pdf *= vtl_pdf;

			return vtl_cluster;
		}
		else
		{
			const uint32 vtl_idx = cugar::quantize( z[2], vtls.vtl_count() );

			// sample the selected VTL uniformly
			vtls.sample( vtl_idx, cugar::Vector2f(z[0],z[1]), &light_vertex->prim_id, &light_vertex->uv, light_vertex_geom, light_pdf, light_edf );

			return INVALID_SAMPLE;
		}
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
		const uint32 vtl_idx = vtls.map(triId, uv, light_vertex_geom, light_pdf, light_edf);
		if (prev_nee_slot != uint32(-1) && vtl_idx != uint32(-1))
		{
			// multiply by the clustered-RL pdf
			*light_pdf *= vtl_rl.pdf( prev_nee_slot, vtl_idx );
		}
	}

	/// update with the resulting NEE sample
	///
	FERMAT_DEVICE
	void update(
		const uint32			nee_slot,
		const uint32			nee_cluster,
		const cugar::Vector3f	w,
		const bool				occluded)
	{
		// update the NEE RL pdfs
		if (nee_cluster != INVALID_SAMPLE)
		{
			const float new_value = cugar::max_comp(w.xyz()) * (occluded == false ? 1.0f : 0.0f);

			vtl_rl.update(nee_slot, nee_cluster, new_value);
		}
	}

	RLMap		vtl_rl;
	VTLMeshView	vtls;
};

///@} PTLib
///@} Fermat
