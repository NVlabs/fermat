/*
 * Fermat
 *
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <uv_bvh.h>
#include <cugar/basic/numbers.h>
#include <algorithm>

// expand bboxes
cugar::Bbox2f refit(cugar::Bvh<2>& bvh, cugar::Bvh_builder<2>& bvh_builder, const cugar::Bbox2f* uv_bboxes, const uint32_t index = 0)
{
    // fetch node
    cugar::Bvh_node node = bvh.m_nodes[index];

    cugar::Bbox2f bbox;

    if (node.is_leaf())
    {
        // compute leaf's bbox
		const uint2 range =  node.get_leaf_range();
        for (uint32_t i = range.x; i < range.y; ++i)
            bbox.insert( uv_bboxes[bvh_builder.index(i)] );
    }
    else
    {
        // recurse left
        bbox.insert( refit( bvh, bvh_builder, uv_bboxes, node.get_child(0) ) );

        // recurse right
        bbox.insert( refit( bvh, bvh_builder, uv_bboxes, node.get_child(1) ) );
    }

    // rewrite bbox
    bvh.m_bboxes[index] = bbox;
    return bbox;
}

void build(HostUVBvh* uvbvh, const MeshStorage& mesh)
{
    // loop through all groups in the mesh, and build a Bvh for the UV-triangles in each of them
    //
    const uint32_t num_groups = mesh.getNumGroups();

    const MeshStorage::texture_triangle* texture_indices = reinterpret_cast<const MeshStorage::texture_triangle*>(mesh.getTextureCoordinateIndices());
    const cugar::Vector2f*	texture_data    = reinterpret_cast<const cugar::Vector2f*>(mesh.getTextureCoordinateData());
    const int*				group_offsets   = mesh.getGroupOffsets();

    cugar::Bvh<2>          bvh;
    cugar::Bvh_builder<2>  bvh_builder;

    cugar::make_room(uvbvh->index, mesh.getNumTriangles());   // reserve space for object indices
    cugar::make_room(uvbvh->nodes,  num_groups);              // reserve space for bvh roots
    cugar::make_room(uvbvh->bboxes, num_groups);              // reserve space for bvh roots

    uint32_t max_group_size = 0;
    for (uint32_t g = 0; g < num_groups; ++g)
    {
        const uint32_t group_begin = group_offsets[g];
        const uint32_t group_end   = group_offsets[g+1];
        const uint32_t group_size  = group_end - group_begin;

        max_group_size = cugar::max( max_group_size, group_size );
    }

    cugar::vector<cugar::host_tag, cugar::Vector2f> uv_points( max_group_size );
    cugar::vector<cugar::host_tag, cugar::Bbox2f>   uv_bboxes( max_group_size );

    //fprintf(stderr, "   # groups: %u\n", num_groups);

    for (uint32_t g = 0; g < num_groups; ++g)
    {
        const uint32_t group_begin = group_offsets[g];
        const uint32_t group_end   = group_offsets[g+1];
        const uint32_t group_size  = group_end - group_begin;

        //fprintf(stderr, "     group[%u] : %u triangles\n", g, group_size);

        // fetch the UV-triangle centroids
        //

        for (uint32_t t = 0; t < group_size; ++t)
        {
            const int4 tri = texture_indices[group_begin + t];

            uv_points[t] = (texture_data[tri.x] +
                            texture_data[tri.y] +
                            texture_data[tri.z]) / 3.0f;

            uv_bboxes[t].clear();
            uv_bboxes[t].insert( texture_data[tri.x] );
            uv_bboxes[t].insert( texture_data[tri.y] );
            uv_bboxes[t].insert( texture_data[tri.z] );
        }

        //fprintf(stderr, "     building bvh... started\n");

        // compute the bvh
        //
        bvh_builder.build( uv_points.begin(), uv_points.begin() + group_size, &bvh );

        //fprintf(stderr, "     building bvh... done\n");
        //fprintf(stderr, "       nodes  : %u\n", (uint32_t)bvh.m_nodes.size());
        //fprintf(stderr, "       leaves : %u\n", (uint32_t)bvh.m_leaves.size());

        //fprintf(stderr, "     compressing bvh... begin\n");

        // adapt the bvh
        //
        const uint32_t cur_node = uint32_t( uvbvh->nodes.size() );
        const uint32_t cur_bbox = uint32_t( uvbvh->bboxes.size() );

        cugar::make_room(uvbvh->nodes,  uint32_t(cur_node  + bvh.m_nodes.size() - 1) );
        cugar::make_room(uvbvh->bboxes, uint32_t(cur_bbox + bvh.m_bboxes.size() - 1) );

		std::vector<uint32> skip_nodes( bvh.m_nodes.size() );
		cugar::build_skip_nodes( &bvh.m_nodes[0], &skip_nodes[0] );

        // write the root node
        const cugar::Bvh_node root_node = bvh.m_nodes[0];

        if (root_node.is_leaf())
        {
			const uint32 leaf_index = root_node.get_leaf_begin();
			const uint32 leaf_size  = root_node.get_leaf_size();

            uvbvh->nodes[g] = UVBvh_node(
                UVBvh_node::kLeaf,
                leaf_size,
                group_begin + leaf_index,
                UVBvh_node::kInvalid );
        }
        else
        {
            uvbvh->nodes[g] = UVBvh_node(
                UVBvh_node::kInternal,
                2u,
                root_node.get_child_index() + cur_node - 1,
                UVBvh_node::kInvalid );
        }

        // rewrite internal nodes
        for (uint32_t n = 1; n < bvh.m_nodes.size(); ++n)
        {
            const cugar::Bvh_node in_node = bvh.m_nodes[n];

            const uint32_t skip_node = skip_nodes[n] == UVBvh_node::kInvalid ?
                UVBvh_node::kInvalid : skip_nodes[n] + cur_node - 1;

            if (in_node.is_leaf())
            {
				const uint32 leaf_index = in_node.get_leaf_begin();
				const uint32 leaf_size  = in_node.get_leaf_size();

                uvbvh->nodes[cur_node + n - 1] = UVBvh_node(
                    UVBvh_node::kLeaf,
                    leaf_size,
                    group_begin + leaf_index,
                    skip_node );
            }
            else
            {
                uvbvh->nodes[cur_node + n - 1] = UVBvh_node(
                    UVBvh_node::kInternal,
                    2u,
                    in_node.get_child_index() + cur_node - 1,
                    skip_node );
            }
        }

        // rewrite indices
        for (uint32_t t = 0; t < group_size; ++t)
            uvbvh->index[group_begin + t] = group_begin + bvh_builder.index(t);

        // refit bboxes to the actual triangle bounds
        refit( bvh, bvh_builder, cugar::raw_pointer( uv_bboxes ) );

        // append bboxes
        uvbvh->bboxes[g] = bvh.m_bboxes[0];
        for (uint32_t n = 1; n < bvh.m_bboxes.size(); ++n)
            uvbvh->bboxes[cur_node + n - 1] = bvh.m_bboxes[n];

        //fprintf(stderr, "     compressing bvh... done\n");
    }
}

void build(HostUVBvh* uvbvh, const cugar::vector<cugar::host_tag,VTL>& vtls)
{
	const uint32 n_vtls = uint32(vtls.size());

	cugar::Bvh<2>          bvh;
    cugar::Bvh_builder<2>  bvh_builder;

    cugar::make_room(uvbvh->index,  vtls.size());		// reserve space for object indices
    cugar::make_room(uvbvh->nodes,  1);					// reserve space for bvh roots
    cugar::make_room(uvbvh->bboxes, 1);					// reserve space for bvh roots

    cugar::vector<cugar::host_tag, cugar::Vector2f> uv_points( n_vtls );
    cugar::vector<cugar::host_tag, cugar::Bbox2f>   uv_bboxes( n_vtls );

    //fprintf(stderr, "   # groups: %u\n", num_groups);

    for (uint32_t t = 0; t < n_vtls; ++t)
    {
		const VTL tri = vtls[t];

		// shift the vtl by the prim-id along the U axis
		const cugar::Vector2f shift( float(tri.prim_id), 0.0f );

        uv_points[t] = tri.uv_centroid() + shift;

        uv_bboxes[t].clear();
        uv_bboxes[t].insert( tri.uv0 + shift );
        uv_bboxes[t].insert( tri.uv1 + shift );
        uv_bboxes[t].insert( tri.uv2 + shift );
	}

	// compute the bvh
    //
    bvh_builder.build( uv_points.begin(), uv_points.begin() + n_vtls, &bvh );

    //fprintf(stderr, "     building bvh... done\n");
    //fprintf(stderr, "       nodes  : %u\n", (uint32_t)bvh.m_nodes.size());
    //fprintf(stderr, "       leaves : %u\n", (uint32_t)bvh.m_leaves.size());

    //fprintf(stderr, "     compressing bvh... begin\n");

    // adapt the bvh
    //
    cugar::make_room(uvbvh->nodes,  uint32_t(bvh.m_nodes.size()) );
    cugar::make_room(uvbvh->bboxes, uint32_t(bvh.m_bboxes.size()) );

	std::vector<uint32> skip_nodes( bvh.m_nodes.size() );
	cugar::build_skip_nodes( &bvh.m_nodes[0], &skip_nodes[0] );

    // rewrite nodes
    for (uint32_t n = 0; n < bvh.m_nodes.size(); ++n)
    {
        const cugar::Bvh_node in_node = bvh.m_nodes[n];

        const uint32_t skip_node = skip_nodes[n];

        if (in_node.is_leaf())
        {
			const uint32 leaf_index = in_node.get_leaf_begin();
			const uint32 leaf_size  = in_node.get_leaf_size();

            uvbvh->nodes[n] = UVBvh_node(
                UVBvh_node::kLeaf,
                leaf_size,
                leaf_index,
                skip_node );
        }
        else
        {
            uvbvh->nodes[n] = UVBvh_node(
                UVBvh_node::kInternal,
                2u,
                in_node.get_child_index(),
                skip_node );
        }
    }

    // copy the indices
    for (uint32_t t = 0; t < n_vtls; ++t)
        uvbvh->index[t] = bvh_builder.index(t);

    // refit bboxes to the actual triangle bounds
    refit( bvh, bvh_builder, cugar::raw_pointer( uv_bboxes ) );

    // copy the bboxes
    for (uint32_t n = 0; n < bvh.m_bboxes.size(); ++n)
        uvbvh->bboxes[n] = bvh.m_bboxes[n];
}
