//
//Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
//
//NVIDIA Corporation and its licensors retain all intellectual property and
//proprietary rights in and to this software, related documentation and any
//modifications thereto.  Any use, reproduction, disclosure or distribution of
//this software and related documentation without an express license agreement
//from NVIDIA Corporation is strictly prohibited.
//
//TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
//OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
//MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
//NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
//CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
//LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
//INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGES
//

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
