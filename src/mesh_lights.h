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

#include <lights.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/vector.h>
#include <cugar/spherical/mappings.h>

struct MeshLightsStorage
{
	MeshLightsStorage() {}

	void init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance = 0);

	MeshLight view(const bool use_vpls)
	{
		if (use_vpls)
			return MeshLight(uint32(mesh_cdf.size()), cugar::raw_pointer(mesh_cdf), cugar::raw_pointer(mesh_inv_area), mesh, textures, uint32(vpls.size()), cugar::raw_pointer(vpl_cdf), cugar::raw_pointer(vpls), normalization_coeff);
		else
			return MeshLight(uint32(mesh_cdf.size()), cugar::raw_pointer(mesh_cdf), cugar::raw_pointer(mesh_inv_area), mesh, textures, 0, 0, NULL, normalization_coeff);
	}

	cugar::vector<cugar::device_tag, float>	 mesh_cdf;
	cugar::vector<cugar::device_tag, float>	 mesh_inv_area;
	MeshView								 mesh;
	const MipMapView*						 textures;
	cugar::vector<cugar::device_tag, float>	 vpl_cdf;
	cugar::vector<cugar::device_tag, VPL>	 vpls;
	float									 normalization_coeff;
};
