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

#include <mesh_lights.h>
#include <cugar/sampling/lfsr.h>

void MeshLightsStorage::init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance)
{
	cugar::vector<cugar::host_tag, float> h_mesh_cdf(h_mesh.num_triangles);
	cugar::vector<cugar::host_tag, float> h_mesh_inv_area(h_mesh.num_triangles);

	float sum = 0.0f;

	cugar::LFSRGeneratorMatrix generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS);
	cugar::LFSRRandomStream random(&generator,1u,cugar::hash(1351u + instance));

	for (uint32 i = 0; i < (uint32)h_mesh.num_triangles; ++i)
	{
		const int3 tri = reinterpret_cast<const int3*>(h_mesh.vertex_indices)[i];
		const cugar::Vector3f vp0 = reinterpret_cast<const float3*>(h_mesh.vertex_data)[tri.x];
		const cugar::Vector3f vp1 = reinterpret_cast<const float3*>(h_mesh.vertex_data)[tri.y];
		const cugar::Vector3f vp2 = reinterpret_cast<const float3*>(h_mesh.vertex_data)[tri.z];

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		const float area = 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));

		const uint32 material_id = h_mesh.material_indices[i];
		const MeshMaterial material = h_mesh.materials[material_id];

		if (material.emissive_map.is_valid() && h_textures[material.emissive_map.texture].n_levels)
		{
			// estimate the triangle's emissive energy using filtered texture lookups
			//

			// estimate the triangle area in texture space
			const int3 tri = reinterpret_cast<const int3*>(h_mesh.texture_indices)[i];
			const cugar::Vector2f vt0 = tri.x >= 0 ? reinterpret_cast<const float2*>(h_mesh.texture_data)[tri.x] : cugar::Vector2f(1.0f, 0.0f);
			const cugar::Vector2f vt1 = tri.y >= 0 ? reinterpret_cast<const float2*>(h_mesh.texture_data)[tri.y] : cugar::Vector2f(0.0f, 1.0f);
			const cugar::Vector2f vt2 = tri.z >= 0 ? reinterpret_cast<const float2*>(h_mesh.texture_data)[tri.z] : cugar::Vector2f(0.0f, 0.0f);

			cugar::Vector2f dst_du = vt0 - vt2;
			cugar::Vector2f dst_dv = vt1 - vt2;

			float n_samples = 10;

			const MipMapView mipmap = h_textures[material.emissive_map.texture];

			// compute how large is the filter footprint in texels
			float max_edge = cugar::max(
				cugar::max(fabsf(dst_du.x), fabsf(dst_dv.x)) * material.emissive_map.scaling.x * mipmap.levels[0].res_x,
				cugar::max(fabsf(dst_du.y), fabsf(dst_dv.y)) * material.emissive_map.scaling.y * mipmap.levels[0].res_y);

			// adjust for the sampling density
			max_edge /= sqrtf(n_samples);

			// compute the lod as the log2 of max_edge
			const uint32 lod = cugar::min( cugar::log2( uint32(max_edge) ), mipmap.n_levels-1 );

			const TextureView texture = mipmap.levels[lod];

			cugar::Vector4f avg(0.0f);

			// take a random set of points on the triangle
			for (uint32 i = 0; i < n_samples; ++i)
			{
				float u = random.next();
				float v = random.next();

				if (u + v > 1.0f)
				{
					u = 1.0f - u;
					v = 1.0f - v;
				}

				const cugar::Vector2f st = cugar::mod( (vt2 * (1.0f - u - v) + vt0 *u + vt1 * v) * cugar::Vector2f( material.emissive_map.scaling ), 1.0f );

				const uint32 x = cugar::min(uint32(st.x * texture.res_x), texture.res_x - 1);
				const uint32 y = cugar::min(uint32(st.y * texture.res_y), texture.res_y - 1);

				avg += cugar::Vector4f( texture(x, y) );
			}

			avg /= float(n_samples);

			const float E = VPL::pdf(cugar::Vector4f(material.emissive) * avg);

			sum += E * area;
		}
		else
		{
			const float E = VPL::pdf( material.emissive );

			sum += E * area;
		}

		h_mesh_cdf[i] = sum;
		h_mesh_inv_area[i] = 1.0f / area;
	}

	// normalize the CDF
	if (sum)
	{
		fprintf(stderr, "    total emission: %f\n", sum);
		float inv_sum = 1.0f / sum;
		for (uint32 i = 0; i < (uint32)h_mesh_cdf.size(); ++i)
			h_mesh_cdf[i] *= inv_sum;

		// go backwards in the cdf, and fix the trail of values that should be one but aren't
		if (h_mesh_cdf.last() != 1.0f)
		{
			const float last = h_mesh_cdf.last();

			for (int32 i = (int32)h_mesh_cdf.size() - 1; i >= 0; --i)
			{
				if (h_mesh_cdf[i] == last)
					h_mesh_cdf[i] = 1.0f;
				else
					break;
			}
		}

		// Use a uniform CDF
		//for (uint32 i = 0; i < (uint32)h_mesh_cdf.size(); ++i)
		//	h_mesh_cdf[i] = float(i+1) / h_mesh_cdf.size();
	}
	else
	{
		// no emissive surfaces
		for (uint32 i = 0; i < (uint32)h_mesh_cdf.size(); ++i)
			h_mesh_cdf[i] = float(i+1)/float(h_mesh_cdf.size());

		mesh_cdf = h_mesh_cdf;
		mesh	 = d_mesh;
		textures = d_textures;

		fprintf(stderr, "\nwarning: no emissive surfaces found!\n\n");
		return;
	}

	// keep track of the normalization coefficient of our pdf, i.e. the area measure integral of emission
	normalization_coeff = 0.0f;

	cugar::vector<cugar::host_tag, VPL> h_vpls(n_vpls);

	const float one = nexttowardf(1.0f, 0.0f);

	for (uint32 i = 0; i < n_vpls; ++i)
	{
		const float r = (i + random.next()) / float(n_vpls);

		const uint32 tri_id = cugar::min( cugar::upper_bound_index( cugar::min( r, one ), h_mesh_cdf.begin(), (uint32)h_mesh_cdf.size() ), (uint32)h_mesh_cdf.size()-1 );

		float u = random.next();
		float v = random.next();

		if (u + v > 1.0f)
		{
			u = 1.0f - u;
			v = 1.0f - v;
		}

		// Use the same exact code executed by the actual light source class
		VertexGeometry geom;
		float          pdf;

		setup_differential_geometry(h_mesh, tri_id, u, v, &geom, &pdf);

		pdf *= h_mesh_cdf[tri_id] - (tri_id ? h_mesh_cdf[tri_id - 1] : 0.0f);

		const int material_id = h_mesh.material_indices[tri_id];
		assert(material_id < h_mesh.num_materials);
		MeshMaterial material = h_mesh.materials[material_id];

		material.emissive = cugar::Vector4f(material.emissive) * texture_lookup(geom.texture_coords, material.emissive_map, h_textures, cugar::Vector4f(1.0f));

		cugar::Vector4f E(material.emissive);

		E /= pdf;

		h_vpls[i].prim_id = tri_id;
		h_vpls[i].uv      = make_float2(u,v);
		h_vpls[i].E       = VPL::pdf(E);

		// keep track of the normalization coefficient of our pdf, i.e. the area measure integral of emission
		normalization_coeff += h_vpls[i].E;
	}

	// keep track of the normalization coefficient of our pdf, i.e. the area measure integral of emission
	normalization_coeff /= n_vpls;

	// rebuild the CDF across the VPLs
	cugar::vector<cugar::host_tag, float> h_vpl_cdf(n_vpls);
	{
		float sum = 0.0f;
		for (uint32 i = 0; i < n_vpls; ++i)
		{
			// convert E to the area measure PDF
			h_vpls[i].E /= normalization_coeff;

			sum += h_vpls[i].E / float(n_vpls);

			h_vpl_cdf[i] = sum;
		}
	}

	// resample our VPLs so as to distribute them exactly according to emission
	cugar::vector<cugar::host_tag, VPL> h_resampled_vpls(n_vpls);

	for (uint32 i = 0; i < n_vpls; ++i)
	{
		const float r = (float(i) + random.next()) / float(n_vpls);

		const uint32 vpl_id = cugar::min( cugar::upper_bound_index( cugar::min(r, one), h_vpl_cdf.begin(), (uint32)h_vpl_cdf.size()), n_vpls - 1u );
		//fprintf(stderr, "%u\n", vpl_id);

		h_resampled_vpls[i] = h_vpls[vpl_id];
	}

	// adjust the mesh CDF dividing it by the area
	{
	}

	mesh_cdf		= h_mesh_cdf;
	mesh_inv_area	= h_mesh_inv_area;
	mesh			= d_mesh;
	textures		= d_textures;
	vpl_cdf			= h_vpl_cdf;
	vpls			= h_resampled_vpls;
}
