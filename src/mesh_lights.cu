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

#include <mesh_lights.h>
#include <mesh_lights_impl.h>
#include <renderer.h>
#include <cugar/sampling/lfsr.h>
#include <cugar/basic/primitives.h>
#include <cugar/bvh/cuda/lbvh_builder.h>
#include <cugar/tree/cuda/reduce.h>
#include <cugar/bintree/bintree_visitor.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/gather.h>
#include <queue>

namespace {

struct floar4tovec3
{
	typedef cugar::Vector3f result_type;
	typedef float4			argument_type;

	FERMAT_HOST_DEVICE
	cugar::Vector3f operator() (const float4 v) const { return cugar::Vector3f(v.x,v.y,v.z); }
};

struct leaf_index_functor
{
	typedef uint32					result_type;
	typedef cugar::Bvh_node_3d		argument_type;

	FERMAT_HOST_DEVICE
	result_type operator() (const argument_type node) const { return node.get_child_index(); }
};

struct is_leaf_functor
{
	typedef bool					result_type;
	typedef cugar::Bvh_node_3d		argument_type;

	FERMAT_HOST_DEVICE
	result_type operator() (const argument_type node) const { return node.is_leaf() ? true : false; }
};

struct Bbox_merge_functor
{
    // merge two points
    CUGAR_HOST_DEVICE
	cugar::Bbox3f operator() (
        const float4 pnt1,
        const float4 pnt2) const
    {
        // Build a bbox for each of the two points
        cugar::Bbox3f bbox1(
            cugar::Vector3f( pnt1.x - pnt1.w, pnt1.y - pnt1.w, pnt1.z - pnt1.w ),
            cugar::Vector3f( pnt1.x + pnt1.w, pnt1.y + pnt1.w, pnt1.z + pnt1.w ) );
        cugar::Bbox3f bbox2(
            cugar::Vector3f( pnt2.x - pnt2.w, pnt2.y - pnt2.w, pnt2.z - pnt2.w ),
            cugar::Vector3f( pnt2.x + pnt2.w, pnt2.y + pnt2.w, pnt2.z + pnt2.w ) );

        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }

    // merge a bbox and a point
    CUGAR_HOST_DEVICE
	cugar::Bbox3f operator() (
        const cugar::Bbox3f	bbox1,
        const float4		pnt2) const
    {
        cugar::Bbox3f bbox2(
            cugar::Vector3f( pnt2.x - pnt2.w, pnt2.y - pnt2.w, pnt2.z - pnt2.w ),
            cugar::Vector3f( pnt2.x + pnt2.w, pnt2.y + pnt2.w, pnt2.z + pnt2.w ) );

        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }

    // merge two bboxes
    CUGAR_HOST_DEVICE cugar::Bbox3f operator() (
        const cugar::Bbox3f bbox1,
        const cugar::Bbox3f	bbox2) const
    {
        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }
};

struct BvhNodeLess
{
	bool operator() (const cugar::Bvh_node_3d* n1, const cugar::Bvh_node_3d* n2) const
	{
		return n1->get_range_size() < n2->get_range_size();
		//return cugar::area( n1->bbox ) < cugar::area( n2->bbox );
	}
};

void check_ranges(const uint32 node_index, const cugar::Bvh_node_3d* nodes, const uint2* ranges)
{
	const cugar::Bvh_node_3d* node = nodes + node_index;

	if (!node->is_leaf())
	{
		const uint2 range = ranges[node_index];
		const uint2 range0 = ranges[node->get_child(0)];
		const uint2 range1 = ranges[node->get_child(1)];

		if (range0.x != range.x || range1.y != range.y || range0.y != range1.x)
		{
			fprintf(stderr, "range mismatch at node %u : %u, %u:\n  r[%u,%u)\n  r0[%u,%u), r1[%u,%u)\n",
				node_index, node->get_child(0), node->get_child(1),
				range.x, range.y,
				range0.x, range0.y,
				range1.x, range1.y);
			exit(1);
		}

		check_ranges( node->get_child(0), nodes, ranges );
		check_ranges( node->get_child(1), nodes, ranges );
	}
}

} // anonymous namespace

// ---- MeshLightsStorageImpl ------------------------------------------------------------------------------------------------------------------------------------------- //

void MeshLightsStorageImpl::init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance)
{
	// initialize the mesh lights sampler
	init( n_vpls, renderer.get_host_mesh().view(), renderer.get_device_mesh().view(), renderer.get_host_texture_views(), renderer.get_device_texture_views() );
}

void MeshLightsStorageImpl::init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance)
{
	cugar::vector<cugar::host_tag, float> h_mesh_cdf(h_mesh.num_triangles);
	cugar::vector<cugar::host_tag, float> h_mesh_inv_area(h_mesh.num_triangles);

	float sum = 0.0f;

	cugar::LFSRGeneratorMatrix generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS);
	cugar::LFSRRandomStream random(&generator,1u,cugar::hash(1351u + instance));

	for (uint32 i = 0; i < (uint32)h_mesh.num_triangles; ++i)
	{
		const MeshStorage::vertex_triangle tri = reinterpret_cast<const MeshStorage::vertex_triangle*>(h_mesh.vertex_indices)[i];
		const cugar::Vector3f vp0 = load_vertex(h_mesh, tri.x);
		const cugar::Vector3f vp1 = load_vertex(h_mesh, tri.y);
		const cugar::Vector3f vp2 = load_vertex(h_mesh, tri.z);

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
			const MeshStorage::texture_triangle tri = reinterpret_cast<const MeshStorage::texture_triangle*>(h_mesh.texture_indices)[i];
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
	}
	else
	{
		// no emissive surfaces
		for (uint32 i = 0; i < (uint32)h_mesh_cdf.size(); ++i)
			h_mesh_cdf[i] = float(i+1)/float(h_mesh_cdf.size());

		mesh_cdf		= h_mesh_cdf;
		mesh_inv_area	= h_mesh_inv_area;
		mesh			= d_mesh;
		textures		= d_textures;

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
	cugar::vector<cugar::host_tag, float4> h_resampled_pos(n_vpls);

	cugar::Bbox3f bbox;

	for (uint32 i = 0; i < n_vpls; ++i)
	{
		const float r = (float(i) + random.next()) / float(n_vpls);

		const uint32 vpl_id = cugar::min( cugar::upper_bound_index( cugar::min(r, one), h_vpl_cdf.begin(), (uint32)h_vpl_cdf.size()), n_vpls - 1u );
		//fprintf(stderr, "%u\n", vpl_id);

		h_resampled_vpls[i] = h_vpls[vpl_id];
		h_resampled_pos[i] = cugar::Vector4f( interpolate_position( h_mesh, h_resampled_vpls[i] ), 0.0f );

		bbox.insert( cugar::Vector3f(h_resampled_pos[i].x, h_resampled_pos[i].y, h_resampled_pos[i].z) );
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

	// build a bvh
	cugar::vector<cugar::device_tag, float4>	bvh_points = h_resampled_pos;
	cugar::vector<cugar::device_tag, uint32>	bvh_index;

	cugar::cuda::LBVH_builder<uint64,cugar::Bvh_node_3d> bvh_builder( &bvh_nodes, &bvh_index, NULL, NULL, &bvh_parents, NULL, &bvh_ranges );

	bvh_builder.build( bbox, thrust::make_transform_iterator(bvh_points.begin(), floar4tovec3()), thrust::make_transform_iterator(bvh_points.end(), floar4tovec3()), 16u );

	const uint32 node_count = bvh_builder.m_node_count;
	const uint32 leaf_count = bvh_builder.m_leaf_count;

	// build its bounding boxes
	cugar::Bintree_visitor<cugar::Bvh_node_3d> bvh;
    bvh.set_node_count( node_count );
    bvh.set_leaf_count( leaf_count );
    bvh.set_nodes( cugar::raw_pointer( bvh_nodes ) );
    bvh.set_parents( cugar::raw_pointer( bvh_parents ) );

    cugar::Bvh_node_3d_bbox_iterator bbox_iterator( cugar::raw_pointer( bvh_nodes ) );

    cugar::cuda::tree_reduce(
        bvh,
        thrust::make_permutation_iterator( bvh_points.begin(), bvh_index.begin() ),
        bbox_iterator,
        Bbox_merge_functor(),
        cugar::Bbox3f() );

	// reorder the vpls according to the tree's index
	{
		cugar::vector<cugar::device_tag, VPL> sorted_vpls(vpls.size());

		thrust::gather(bvh_index.begin(), bvh_index.end(), vpls.begin(), sorted_vpls.begin());

		vpls.swap( sorted_vpls );
	}

	// extract the leaves
	bvh_leaves.resize( leaf_count );
	{
		cugar::vector<cugar::device_tag,uint8> temp_storage;

		assert( leaf_count == cugar::cuda::copy_flagged(
			node_count,
			thrust::make_transform_iterator( bvh_nodes.begin(), leaf_index_functor() ),
			thrust::make_transform_iterator( bvh_nodes.begin(), is_leaf_functor() ),
			bvh_leaves.begin(),
			temp_storage ) );
	}

	cugar::vector<cugar::host_tag, cugar::Bvh_node_3d>	h_bvh_nodes   = bvh_nodes;
	cugar::vector<cugar::host_tag, uint2>				h_bvh_ranges  = bvh_ranges;

	// perform some debugging...
	check_ranges(0u, &h_bvh_nodes[0], &h_bvh_ranges[0]);


	// collect the initial clusters
	const uint32 target_clusters = 256;

	cugar::vector<cugar::host_tag, uint32> h_clusters;
	cugar::vector<cugar::host_tag, uint32> h_cluster_offsets;
	{
		// starting from the root, split the nodes until we reach the desired quota,
		// using their surface area to prioritize the splitting process
		const cugar::Bvh_node_3d* root = &h_bvh_nodes[0];

		std::priority_queue<const cugar::Bvh_node_3d*, std::vector<const cugar::Bvh_node_3d*>, BvhNodeLess> queue;
		queue.push( root );

		while (!queue.empty() && (queue.size() + h_clusters.size() < target_clusters))
		{
			const cugar::Bvh_node_3d* node = queue.top();
			queue.pop();

			if (node->is_leaf())
			{
				const uint32 node_index = uint32(node - root);

				h_clusters.push_back(node_index);
				h_cluster_offsets.push_back(h_bvh_ranges[node_index].x);
			}
			else
			{
				queue.push( root + node->get_child(0u) );
				queue.push( root + node->get_child(1u) );
			}
		}

		// collect all nodes in the queue
		while (queue.size())
		{
			const cugar::Bvh_node_3d* node = queue.top();
			queue.pop();

			const uint32 node_index = uint32(node - root);

			h_clusters.push_back(node_index);
			h_cluster_offsets.push_back(h_bvh_ranges[node_index].x);
		}
	}

	const uint32 cluster_count = uint32( h_clusters.size() );

	fprintf(stderr, "    nodes    : %u\n", node_count);
	fprintf(stderr, "    leaves   : %u\n", leaf_count);
	fprintf(stderr, "    clusters : %u\n", cluster_count);

	bvh_clusters		= h_clusters;
	bvh_cluster_offsets	= h_cluster_offsets;

	// extract the sorted cluster offsets
	bvh_cluster_offsets.resize( cluster_count+1 );
	{
		cugar::vector<cugar::device_tag,uint8> temp_storage;

		cugar::radix_sort( cluster_count, bvh_cluster_offsets.begin(), bvh_clusters.begin(), temp_storage );
	}
	bvh_cluster_offsets[cluster_count] = n_vpls;
}

// ---- MeshLightsStorage --------------------------------------------------------------------------------------------------------------------------------------------- //

MeshLightsStorage::MeshLightsStorage() { m_impl = new MeshLightsStorageImpl(); }

void MeshLightsStorage::init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance)
{
	m_impl->init( n_vpls, h_mesh, d_mesh, h_textures, d_textures, instance );
}
void MeshLightsStorage::init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance)
{
	m_impl->init( n_vpls, renderer, instance );
}

MeshLight MeshLightsStorage::view(const bool use_vpls) const
{
	return m_impl->view( use_vpls );
}

uint32 MeshLightsStorage::get_vpl_count() const { return uint32( m_impl->vpls.size() ); }
VPL* MeshLightsStorage::get_vpls() const { return cugar::raw_pointer( m_impl->vpls ); }

uint32 MeshLightsStorage::get_bvh_nodes_count() const		{ return m_impl->get_bvh_nodes_count(); }
uint32 MeshLightsStorage::get_bvh_clusters_count() const	{ return m_impl->get_bvh_clusters_count(); }

const cugar::Bvh_node_3d*	MeshLightsStorage::get_bvh_nodes() const			{ return m_impl->get_bvh_nodes(); }
const uint32*				MeshLightsStorage::get_bvh_parents() const			{ return m_impl->get_bvh_parents(); }
const uint2*				MeshLightsStorage::get_bvh_ranges() const			{ return m_impl->get_bvh_ranges(); }
const uint32*				MeshLightsStorage::get_bvh_clusters() const			{ return m_impl->get_bvh_clusters(); }
const uint32*				MeshLightsStorage::get_bvh_cluster_offsets() const	{ return m_impl->get_bvh_cluster_offsets(); }

// ---- MeshVTLStorageImpl -------------------------------------------------------------------------------------------------------------------------------------------- //

struct Node
{
	Node() {}
	Node(VTL _vtl, float _E) : vtl(_vtl), E(_E) {}

	VTL		vtl;
	float	E;
};

struct NodeLess
{
	bool operator() (Node& a, Node& b) const { return a.E < b.E; }
};

float compute_E(const VTL vtl, MeshView h_mesh, const MipMapView* h_textures, cugar::LFSRRandomStream& random)
{
	// fetch the original triangle
	cugar::Vector3f vp0, vp1, vp2;
	vtl.interpolate_positions( h_mesh, vp0, vp1, vp2 );

	const cugar::Vector3f dp_du = vp0 - vp2;
	const cugar::Vector3f dp_dv = vp1 - vp2;
	const float area = 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));

	const uint32 material_id = h_mesh.material_indices[vtl.prim_id];
	const MeshMaterial material = h_mesh.materials[material_id];

	if (material.emissive_map.is_valid() && h_textures[material.emissive_map.texture].n_levels)
	{
		// estimate the triangle's emissive energy using filtered texture lookups
		//
		cugar::Vector2f vt0, vt1, vt2;
		vtl.interpolate_tex_coords( h_mesh, vt0, vt1, vt2 );

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

		return E * area;
	}
	else
	{
		const float E = VPL::pdf( material.emissive );

		return E * area;
	}
}

void MeshVTLStorageImpl::init(const uint32 n_target_vtls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance)
{
	std::priority_queue<Node, std::vector<Node>, NodeLess>	queue;

	cugar::LFSRGeneratorMatrix generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS);
	cugar::LFSRRandomStream random(&generator,1u,cugar::hash(1351u + instance));

	for (uint32 i = 0; i < (uint32)h_mesh.num_triangles; ++i)
	{
		const MeshStorage::vertex_triangle tri = reinterpret_cast<const MeshStorage::vertex_triangle*>(h_mesh.vertex_indices)[i];
		const cugar::Vector3f vp0 = load_vertex(h_mesh, tri.x);
		const cugar::Vector3f vp1 = load_vertex(h_mesh, tri.y);
		const cugar::Vector3f vp2 = load_vertex(h_mesh, tri.z);

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		const float area = 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));

		const uint32 material_id = h_mesh.material_indices[i];
		const MeshMaterial material = h_mesh.materials[material_id];

		if (cugar::max3(material.emissive.x, material.emissive.y, material.emissive.z) > 0.0f)
		{
			VTL vtl;
			vtl.uv0		= cugar::Vector2f(0.0f,0.0f);
			vtl.uv1		= cugar::Vector2f(1.0f,0.0f);
			vtl.uv2		= cugar::Vector2f(0.0f,1.0f);
			vtl.prim_id = i;
			vtl.area	= area;

			const float E = compute_E(vtl,h_mesh,h_textures,random);
			if (E > 0.0f)
				queue.push( Node(vtl, E) );
		}
	}

	if (queue.empty())
	{
		fprintf(stderr, "\nwarning: no emissive surfaces found!\n\n");
		return;
	}

	while (queue.size() < n_target_vtls)
	{
		// fetch the most energetic VTL and split it
		VTL parent = queue.top().vtl;
		queue.pop();

		// split, with care to keep the parent's winding
		const cugar::Vector2f m01 = (parent.uv0 + parent.uv1)*0.5f;
		const cugar::Vector2f m02 = (parent.uv0 + parent.uv2)*0.5f;
		const cugar::Vector2f m12 = (parent.uv1 + parent.uv2)*0.5f;
		VTL vtl0( parent.prim_id, parent.uv0, m01, m02, parent.area * 0.25f);
		VTL vtl1( parent.prim_id, parent.uv1, m12, m01, parent.area * 0.25f);
		VTL vtl2( parent.prim_id, parent.uv2, m02, m12, parent.area * 0.25f);
		VTL vtl3( parent.prim_id, m02, m01, m12, parent.area * 0.25f);

		queue.push( Node(vtl0, compute_E(vtl0, h_mesh, h_textures, random)) );
		queue.push( Node(vtl1, compute_E(vtl1, h_mesh, h_textures, random)) );
		queue.push( Node(vtl2, compute_E(vtl2, h_mesh, h_textures, random)) );
		queue.push( Node(vtl3, compute_E(vtl3, h_mesh, h_textures, random)) );
	}

	cugar::Bbox3f bbox;

	cugar::vector<cugar::host_tag, VTL>		h_vtls( queue.size() );
	cugar::vector<cugar::host_tag, float4>	h_centroids( queue.size() );
	uint32 n_vtls = 0;
	while (queue.size())
	{
		VTL vtl = queue.top().vtl;

		cugar::Vector3f c = vtl.centroid(h_mesh);

		h_vtls[n_vtls] = vtl;
		h_centroids[n_vtls] = cugar::Vector4f(c, 0.0f);

		bbox.insert(c);

		n_vtls++;

		queue.pop();
	}

	mesh			= d_mesh;
	textures		= d_textures;
	vtls			= h_vtls;

	// nothing to normalize...
	normalization_coeff = 1.0f;

	// build a bvh
	cugar::vector<cugar::device_tag, float4>	bvh_points( h_centroids );
	cugar::vector<cugar::device_tag, uint32>	bvh_index;

	cugar::cuda::LBVH_builder<uint64,cugar::Bvh_node_3d> bvh_builder( &bvh_nodes, &bvh_index, NULL, NULL, &bvh_parents, NULL, &bvh_ranges );

	const uint32 target_clusters = 256;
	//const uint32 leaf_size = (n_vtls + target_clusters - 1u) / target_clusters;
	const uint32 leaf_size = 1u;

	bvh_builder.build( bbox, thrust::make_transform_iterator(bvh_points.begin(), floar4tovec3()), thrust::make_transform_iterator(bvh_points.end(), floar4tovec3()), leaf_size );

	const uint32 node_count = bvh_builder.m_node_count;
	const uint32 leaf_count = bvh_builder.m_leaf_count;

	// build its bounding boxes
	cugar::Bintree_visitor<cugar::Bvh_node_3d> bvh;
    bvh.set_node_count( node_count );
    bvh.set_leaf_count( leaf_count );
    bvh.set_nodes( cugar::raw_pointer( bvh_nodes ) );
    bvh.set_parents( cugar::raw_pointer( bvh_parents ) );

    cugar::Bvh_node_3d_bbox_iterator bbox_iterator( cugar::raw_pointer( bvh_nodes ) );

    cugar::cuda::tree_reduce(
        bvh,
        thrust::make_permutation_iterator( bvh_points.begin(), bvh_index.begin() ),
        bbox_iterator,
        Bbox_merge_functor(),
        cugar::Bbox3f() );

	// reorder the vpls according to the tree's index
	{
		cugar::vector<cugar::device_tag, VTL> sorted_vtls(vtls.size());

		thrust::gather(bvh_index.begin(), bvh_index.end(), vtls.begin(), sorted_vtls.begin());

		vtls.swap( sorted_vtls );
	}

	cugar::vector<cugar::host_tag, cugar::Bvh_node_3d>	h_bvh_nodes   = bvh_nodes;
	cugar::vector<cugar::host_tag, uint2>				h_bvh_ranges  = bvh_ranges;

	// perform some debugging...
	check_ranges(0u, &h_bvh_nodes[0], &h_bvh_ranges[0]);

	// collect the initial clusters
	cugar::vector<cugar::host_tag, uint32> h_clusters;
	cugar::vector<cugar::host_tag, uint32> h_cluster_offsets;
	{
		// starting from the root, split the nodes until we reach the desired quota,
		// using their surface area to prioritize the splitting process
		const cugar::Bvh_node_3d* root = &h_bvh_nodes[0];

		std::priority_queue<const cugar::Bvh_node_3d*, std::vector<const cugar::Bvh_node_3d*>, BvhNodeLess> queue;
		queue.push( root );

		while (!queue.empty() && (queue.size() + h_clusters.size() < target_clusters))
		{
			const cugar::Bvh_node_3d* node = queue.top();
			queue.pop();

			if (node->is_leaf())
			{
				const uint32 node_index = uint32(node - root);

				h_clusters.push_back(node_index);
				h_cluster_offsets.push_back(h_bvh_ranges[node_index].x);
			}
			else
			{
				queue.push( root + node->get_child(0u) );
				queue.push( root + node->get_child(1u) );
			}
		}

		// collect all nodes in the queue
		while (queue.size())
		{
			const cugar::Bvh_node_3d* node = queue.top();
			queue.pop();

			const uint32 node_index = uint32(node - root);

			h_clusters.push_back(node_index);
			h_cluster_offsets.push_back(h_bvh_ranges[node_index].x);
		}
	}
  #if 1
	const uint32 cluster_count = uint32( h_clusters.size() );

	fprintf(stderr, "    nodes    : %u\n", node_count);
	fprintf(stderr, "    leaves   : %u\n", leaf_count);
	fprintf(stderr, "    clusters : %u\n", cluster_count);

	bvh_clusters		= h_clusters;
	bvh_cluster_offsets	= h_cluster_offsets;

	// extract the sorted cluster offsets
	bvh_cluster_offsets.resize( cluster_count+1 );
	{
		cugar::vector<cugar::device_tag,uint8> temp_storage;

		cugar::radix_sort( cluster_count, bvh_cluster_offsets.begin(), bvh_clusters.begin(), temp_storage );
	}
	bvh_cluster_offsets[cluster_count] = n_vtls;
	//fprintf(stderr, "clusters[ ");
	//for (uint32 i = 0; i < cluster_count; ++i)
	//	fprintf(stderr, "%u, ", uint32(bvh_cluster_offsets[i+1] - bvh_cluster_offsets[i]));
	//fprintf(stderr, "]\n");
  #else
	fprintf(stderr, "    nodes    : %u\n", node_count);
	fprintf(stderr, "    leaves   : %u\n", leaf_count);

	// extract the leaves
	bvh_clusters.resize(leaf_count);
	bvh_cluster_offsets.resize( leaf_count+1 );
	{
		cugar::vector<cugar::device_tag,uint8> temp_storage;

		assert( leaf_count == cugar::cuda::copy_flagged(
			node_count,
			thrust::make_counting_iterator(uint32(0u)),
			thrust::make_transform_iterator( bvh_nodes.begin(), is_leaf_functor() ),
			bvh_clusters.begin(),
			temp_storage ) );

		assert( leaf_count == cugar::cuda::copy_flagged(
			node_count,
			thrust::make_transform_iterator( bvh_nodes.begin(), leaf_index_functor() ),
			thrust::make_transform_iterator( bvh_nodes.begin(), is_leaf_functor() ),
			bvh_cluster_offsets.begin(),
			temp_storage ) );

		// and sort them
		cugar::radix_sort( leaf_count, bvh_cluster_offsets.begin(), bvh_clusters.begin(), temp_storage );
	}
	bvh_cluster_offsets[leaf_count] = n_vtls;
  #endif

	// build the UV bvh
	{
		// copy the sorted vtls back to the host
		h_vtls = vtls;

		HostUVBvh h_uvbvh;
		build( &h_uvbvh, h_vtls );

		uvbvh = h_uvbvh;

		fprintf(stderr, "    uv-nodes : %u\n", uint32(h_uvbvh.nodes.size()));
	}
}

void MeshVTLStorageImpl::init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance)
{
	// initialize the mesh lights sampler
	init( n_vpls, renderer.get_host_mesh().view(), renderer.get_device_mesh().view(), renderer.get_host_texture_views(), renderer.get_device_texture_views() );
}

// ---- MeshVTLStorage ----------------------------------------------------------------------------------------------------------------------------------------------- //

MeshVTLStorage::MeshVTLStorage() { m_impl = new MeshVTLStorageImpl(); }

void MeshVTLStorage::init(const uint32 n_vpls, MeshView h_mesh, MeshView d_mesh, const MipMapView* h_textures, const MipMapView* d_textures, const uint32 instance)
{
	m_impl->init( n_vpls, h_mesh, d_mesh, h_textures, d_textures, instance );
}
void MeshVTLStorage::init(const uint32 n_vpls, RenderingContext& renderer, const uint32 instance)
{
	m_impl->init( n_vpls, renderer, instance );
}

VTLMeshView MeshVTLStorage::view() const
{
	return m_impl->view();
}

uint32 MeshVTLStorage::get_vtl_count() const { return uint32( m_impl->vtls.size() ); }
VTL* MeshVTLStorage::get_vtls() const { return cugar::raw_pointer( m_impl->vtls ); }

uint32 MeshVTLStorage::get_bvh_nodes_count() const		{ return m_impl->get_bvh_nodes_count(); }
uint32 MeshVTLStorage::get_bvh_clusters_count() const	{ return m_impl->get_bvh_clusters_count(); }

const cugar::Bvh_node_3d*	MeshVTLStorage::get_bvh_nodes() const			{ return m_impl->get_bvh_nodes(); }
const uint32*				MeshVTLStorage::get_bvh_parents() const			{ return m_impl->get_bvh_parents(); }
const uint2*				MeshVTLStorage::get_bvh_ranges() const			{ return m_impl->get_bvh_ranges(); }
const uint32*				MeshVTLStorage::get_bvh_clusters() const		{ return m_impl->get_bvh_clusters(); }
const uint32*				MeshVTLStorage::get_bvh_cluster_offsets() const	{ return m_impl->get_bvh_cluster_offsets(); }
