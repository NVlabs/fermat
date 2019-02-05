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

#include <lights.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/vector.h>
#include <cugar/spherical/mappings.h>
#include <cugar/bvh/bvh_node.h>
#include <uv_bvh.h>
#include <vtl_mesh_view.h>

struct RenderingContext;

struct MeshLightsStorageImpl
{
	MeshLightsStorageImpl() {}

	void init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance = 0);
	void init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance = 0);

	MeshLight view(const bool use_vpls) const
	{
		if (use_vpls)
			return MeshLight(uint32(mesh_cdf.size()), cugar::raw_pointer(mesh_cdf), cugar::raw_pointer(mesh_inv_area), mesh, textures, uint32(vpls.size()), cugar::raw_pointer(vpl_cdf), cugar::raw_pointer(vpls), normalization_coeff);
		else
			return MeshLight(uint32(mesh_cdf.size()), cugar::raw_pointer(mesh_cdf), cugar::raw_pointer(mesh_inv_area), mesh, textures, 0, 0, NULL, normalization_coeff);
	}

	uint32 get_bvh_nodes_count() const  { return uint32(bvh_nodes.size()); }
	uint32 get_bvh_clusters_count() const { return uint32(bvh_clusters.size()); }
	const cugar::Bvh_node_3d*	get_bvh_nodes() const  { return cugar::raw_pointer(bvh_nodes); }
	const uint32*				get_bvh_parents() const { return cugar::raw_pointer(bvh_parents); }
	const uint2*				get_bvh_ranges() const { return cugar::raw_pointer(bvh_ranges); }
	const uint32*				get_bvh_clusters() const { return cugar::raw_pointer(bvh_clusters); }
	const uint32*				get_bvh_cluster_offsets() const { return cugar::raw_pointer(bvh_cluster_offsets); }

	cugar::vector<cugar::device_tag, float>	 mesh_cdf;
	cugar::vector<cugar::device_tag, float>	 mesh_inv_area;
	MeshView								 mesh;
	const MipMapView*						 textures;
	cugar::vector<cugar::device_tag, float>	 vpl_cdf;
	cugar::vector<cugar::device_tag, VPL>	 vpls;
	float									 normalization_coeff;

	cugar::vector<cugar::device_tag, cugar::Bvh_node_3d>	bvh_nodes;
	cugar::vector<cugar::device_tag, uint32>				bvh_leaves;
	cugar::vector<cugar::device_tag, uint32>				bvh_parents;
	cugar::vector<cugar::device_tag, uint2>					bvh_ranges;
	cugar::vector<cugar::device_tag, uint32>				bvh_clusters;
	cugar::vector<cugar::device_tag, uint32>				bvh_cluster_offsets;
};

struct MeshVTLStorageImpl
{
	MeshVTLStorageImpl() {}

	void init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance = 0);
	void init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance = 0);

	VTLMeshView view() const
	{
		return VTLMeshView(
			uint32( vtls.size() ),
			cugar::raw_pointer( vtls ),
			uvbvh.view(),
			mesh,
			textures );
	}

	uint32 get_bvh_nodes_count() const  { return uint32(bvh_nodes.size()); }
	uint32 get_bvh_clusters_count() const { return uint32(bvh_clusters.size()); }
	const cugar::Bvh_node_3d*	get_bvh_nodes() const  { return cugar::raw_pointer(bvh_nodes); }
	const uint32*				get_bvh_parents() const { return cugar::raw_pointer(bvh_parents); }
	const uint2*				get_bvh_ranges() const { return cugar::raw_pointer(bvh_ranges); }
	const uint32*				get_bvh_clusters() const { return cugar::raw_pointer(bvh_clusters); }
	const uint32*				get_bvh_cluster_offsets() const { return cugar::raw_pointer(bvh_cluster_offsets); }

	MeshView								mesh;
	const MipMapView*						textures;

	cugar::vector<cugar::device_tag, VTL>	vtls;
	float									normalization_coeff;

	DeviceUVBvh								uvbvh;

	cugar::vector<cugar::device_tag, cugar::Bvh_node_3d>	bvh_nodes;
	cugar::vector<cugar::device_tag, uint32>				bvh_parents;
	cugar::vector<cugar::device_tag, uint2>					bvh_ranges;
	cugar::vector<cugar::device_tag, uint32>				bvh_clusters;
	cugar::vector<cugar::device_tag, uint32>				bvh_cluster_offsets;
};
