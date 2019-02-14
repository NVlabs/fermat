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
#include <mesh/MeshView.h>
#include <cugar/basic/vector.h>
#include <cugar/bvh/bvh_node.h>
#include <vtl_mesh_view.h>

struct RenderingContext;

struct MeshLightsStorageImpl;
struct MeshVTLStorageImpl;

struct FERMAT_API MeshLightsStorage
{
	MeshLightsStorage();

	void init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance = 0);
	void init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance = 0);

	MeshLight view(const bool use_vpls) const;

	uint32 get_vpl_count() const;
	VPL* get_vpls() const;

	uint32 get_bvh_nodes_count() const;
	uint32 get_bvh_clusters_count() const;

	const cugar::Bvh_node_3d*	get_bvh_nodes() const;
	const uint32*				get_bvh_parents() const;
	const uint2*				get_bvh_ranges() const;
	const uint32*				get_bvh_clusters() const;
	const uint32*				get_bvh_cluster_offsets() const;

	MeshLightsStorageImpl* m_impl;
};

struct FERMAT_API MeshVTLStorage
{
	MeshVTLStorage();

	void init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance = 0);
	void init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance = 0);

	VTLMeshView view() const;

	uint32 get_vtl_count() const;
	VTL* get_vtls() const;

	uint32 get_bvh_nodes_count() const;
	uint32 get_bvh_clusters_count() const;

	const cugar::Bvh_node_3d*	get_bvh_nodes() const;
	const uint32*				get_bvh_parents() const;
	const uint2*				get_bvh_ranges() const;
	const uint32*				get_bvh_clusters() const;
	const uint32*				get_bvh_cluster_offsets() const;

	MeshVTLStorageImpl* m_impl;
};
