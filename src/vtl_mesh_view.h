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

#include <mesh/MeshView.h>
#include <cugar/basic/vector.h>
#include <cugar/spherical/mappings.h>
#include <cugar/bvh/bvh_node.h>
#include <uv_bvh_view.h>

/// This class provides utilities to view a VTL-mesh, i.e. a collection of mesh-based VTLs.
/// Particularly, it provides methods to sample a given VTL, and methods to locate/map a VTL from a given mesh point.
///
struct VTLMeshView
{
	FERMAT_HOST_DEVICE
	VTLMeshView() :
		n_vtls(0), vtls(NULL), uvbvh(), mesh(), textures(NULL) {}

	FERMAT_HOST_DEVICE
	VTLMeshView(const uint32 _n_vtls, const VTL* _vtls, const UVBvhView& _uvbvh, const MeshView& _mesh, const MipMapView* _textures) :
		n_vtls(_n_vtls), vtls(_vtls), uvbvh(_uvbvh), mesh(_mesh), textures(_textures) {}


	/// sample a point on a given VTL
	///
	FERMAT_HOST_DEVICE
	void sample(
		const uint32			vtl_idx,
		const cugar::Vector2f	vtl_uv,
		uint32_t*				prim_id,
		cugar::Vector2f*		uv,
		VertexGeometry*			geom,
		float*					pdf,
		Edf*					edf) const
	{
		//const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

		// sample one of the VTLs
		*prim_id = vtls[vtl_idx].prim_id;
		*uv		 = vtls[vtl_idx].interpolate_uv( vtl_uv );
		*pdf	 = 1.0f / vtls[vtl_idx].area;

		FERMAT_ASSERT(*prim_id < uint32(mesh.num_triangles));
		setup_differential_geometry(mesh, *prim_id, uv->x, uv->y, geom);

		const int material_id = mesh.material_indices[*prim_id];
		MeshMaterial material = mesh.materials[material_id];

		material.emissive = cugar::Vector4f(material.emissive) * texture_lookup(geom->texture_coords, material.emissive_map, textures, cugar::Vector4f(1.0f));

		*edf = Edf(material);
	}

	/// map a (prim,uv) pair and its surface element to the corresponding VTL/EDF/pdf
	/// NOTE:
	///   the returned pdf here is only _relative_ to the VTL, i.e. it doesn't account for the probability of sampling that VTL,
	///   since the sampling policy is not controlled by this class and intentionally left outside of it.
	///
	FERMAT_HOST_DEVICE
	uint32 map(const uint32_t prim_id, const cugar::Vector2f& uv, const VertexGeometry& geom, float* pdf, Edf* edf) const
	{
		if (n_vtls)
		{
			FERMAT_ASSERT(prim_id < uint32(mesh.num_triangles));
			const int material_id = mesh.material_indices[prim_id];
			MeshMaterial material = mesh.materials[material_id];

			material.emissive = cugar::Vector4f(material.emissive) * texture_lookup(geom.texture_coords, material.emissive_map, textures, cugar::Vector4f(1.0f));

			*edf = Edf(material);

			// find the VTL containing this uv
			const uint32 vtl_idx = locate( uvbvh, vtls, prim_id, uv );
			*pdf = vtl_idx != uint32(-1) ?  1.0f / vtls[vtl_idx].area : 0.0f;
			return vtl_idx;
		}
		else
		{
			*pdf = 1.0f;
			*edf = Edf();
			return uint32(-1);
		}
	}

	FERMAT_HOST_DEVICE
	uint32 vtl_count() const { return n_vtls; }

	FERMAT_HOST_DEVICE
	VTL get_vtl(const uint32 i) const { return vtls[i]; }

	uint32				n_vtls;
	const VTL*			vtls;
	UVBvhView			uvbvh;
	MeshView			mesh;
	const MipMapView*	textures;
};
