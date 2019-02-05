//
//Copyright (c) 2015 NVIDIA Corporation.  All rights reserved.
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

#include <uv_fix.h>
#include <mesh_utils.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/utils.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/bbox.h>
#include <cugar/image/tga.h>
#include <algorithm>
#include <map>
#include <vector>
#include <stack>
//#include <texpacker/GuillotineBinPack.h>

#define NULL_AREA_TRIANGLE		1.0e-9f
#define CHECK_NULL_TRIANGLES	1

namespace  { // anonymous namespace

// a simple edge type, connecting two vertices
//
typedef int2 edge_type;

// a double edge type, used to represent associated edge pairs in the vertex- and UV- graphs
//
typedef int4 double_edge_type;

struct lt_edge
{
    bool operator()(const edge_type a, const edge_type b) const
    {
        return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
    }

    bool operator()(const double_edge_type a, const double_edge_type b) const
    {
        if (a.x < b.x) return true;
        if (a.x > b.x) return false;

        if (a.y < b.y) return true;
        if (a.y > b.y) return false;

        if (a.z < b.z) return true;
        if (a.z > b.z) return false;

        return (a.w < b.w);
    }
};

typedef std::multimap<double_edge_type, int, lt_edge> edge_tri_map_type;

// make undirected edge
//
edge_type make_undirected_edge(const int v1, const int v2)
{
    return v1 < v2 ? make_int2( v1, v2 ) : make_int2( v2, v1 );
}

// make undirected edge
//
double_edge_type make_undirected_edge(const int v1, const int v2, const int uv1, const int uv2)
{
    int2 e    = make_undirected_edge( v1, v2 );
    int2 uv_e = make_undirected_edge( uv1, uv2 );

    return make_int4( e.x, e.y, uv_e.x, uv_e.y );
}

template <int INDEX>
double_edge_type triangle_edge(const MeshStorage::vertex_triangle tri, const MeshStorage::texture_triangle uv_tri)
{
    return (INDEX == 0) ? make_undirected_edge(tri.x, tri.y, uv_tri.x, uv_tri.y) :
           (INDEX == 1) ? make_undirected_edge(tri.y, tri.z, uv_tri.y, uv_tri.z) :
                          make_undirected_edge(tri.z, tri.x, uv_tri.z, uv_tri.x);
}

// calculate the 3 coefficients of a 2d edge equation
//
float3 edge_equation(const float2 a, const float2 b)
{
    return make_float3( a.y - b.y, b.x - a.x, a.x*b.y - b.x*a.y );
}

// evaluate an edge equation
//
float eval_edge_equation(const float3 e, const float2 p)
{
    return e.x * p.x + e.y * p.y + e.z;
}

// evaluate an edge equation
//
float triangle_area(const float2 a, const float2 b, const float2 c)
{
    return (b.x - a.x)*(c.y - a.y) - (c.x - a.x)*(b.y - a.y);
}

struct uv_chart_builder
{
	typedef MeshStorage::vertex_triangle	vertex_triangle;
	typedef MeshStorage::texture_triangle	texture_triangle;

	int								num_triangles;
    int								num_vertices;
    int								num_uv_vertices;
    vertex_triangle*				vertex_indices;
    const MeshView::vertex_type*	vertex_data;
    texture_triangle*				texture_indices;
    const cugar::Vector2f*			texture_data;
    edge_tri_map_type				edge_map;
    std::vector<bool>				triangle_bitmap;
    std::vector<int>				output_index;
    std::vector<int>				output_offsets;

    // merge the UV coordinates
    //
    void merge_uvs()
    {
        cugar::Bbox2f bbox;
        for (int i = 0; i < num_uv_vertices; ++i)
            bbox.insert( texture_data[i] );

        typedef uint32_t int_type;

        typedef std::map<int_type,int> vertex_map_type;
        typedef std::map<int,int>      index_map_type;

        vertex_map_type vertex_map;
        index_map_type  index_map;

        const uint32_t GRID_BITS = 16;
        const uint32_t GRID_SIZE = 1u << GRID_BITS;

        for (int i = 0; i < num_uv_vertices; ++i)
        {
            const cugar::Vector2f vertex = (texture_data[i] - bbox[0]) / (bbox[1] - bbox[0]);
            const int_type vertex_int = (int_type(cugar::quantize(vertex.x, GRID_SIZE))) +
                                        (int_type(cugar::quantize(vertex.y, GRID_SIZE)) << GRID_BITS);

            const vertex_map_type::const_iterator vertex_it = vertex_map.find(vertex_int);
            if (vertex_it == vertex_map.end())
            {
                vertex_map.insert( std::make_pair(vertex_int, i) );
                index_map.insert( std::make_pair(i, i) );
            }
            else
                index_map.insert( std::make_pair(i, vertex_it->second) );
        }
        for (int i = 0; i < num_triangles; ++i)
        {
            texture_triangle& tri = texture_indices[i];
            tri.x = index_map[tri.x];
            tri.y = index_map[tri.y];
            tri.z = index_map[tri.z];
        }
    }
    // merge the vertices
    //
    void merge_vertices()
    {
        cugar::Bbox3f bbox;
        for (int i = 0; i < num_vertices; ++i)
            bbox.insert( vertex_comp( vertex_data[i] ) );

        typedef uint64_t int_type;

        typedef std::map<int_type,int> vertex_map_type;
        typedef std::map<int,int>      index_map_type;

        vertex_map_type vertex_map;
        index_map_type  index_map;

        const uint32_t GRID_BITS = 16;
        const uint32_t GRID_SIZE = 1u << GRID_BITS;

        for (int i = 0; i < num_vertices; ++i)
        {
            const cugar::Vector3f vertex = (vertex_comp( vertex_data[i] ) - bbox[0]) / (bbox[1] - bbox[0]);
            const int_type vertex_int = (int_type(cugar::quantize(vertex.x, GRID_SIZE)) << (0*GRID_BITS)) +
                                        (int_type(cugar::quantize(vertex.y, GRID_SIZE)) << (1*GRID_BITS)) +
                                        (int_type(cugar::quantize(vertex.z, GRID_SIZE)) << (2*GRID_BITS));

            const vertex_map_type::const_iterator vertex_it = vertex_map.find(vertex_int);
            if (vertex_it == vertex_map.end())
            {
                vertex_map.insert( std::make_pair(vertex_int, i) );
                index_map.insert( std::make_pair(i, i) );
            }
            else
                index_map.insert( std::make_pair(i, vertex_it->second) );
        }
        for (int i = 0; i < num_triangles; ++i)
        {
            vertex_triangle& tri = vertex_indices[i];
            tri.x = index_map[tri.x];
            tri.y = index_map[tri.y];
            tri.z = index_map[tri.z];
        }
    }

    void add_triangle(const int t)
    {
        output_index.push_back( t );
        triangle_bitmap[t] = false;
    }

    void grow_chart(const int t)
    {
        // skip previously charted triangles
        if (triangle_bitmap[t] == false)
            return;

        std::vector<double_edge_type> in_edge_frontier;
        std::vector<double_edge_type> out_edge_frontier;

        const vertex_triangle tri     = vertex_indices[t];
        const texture_triangle uv_tri = texture_indices[t];

        in_edge_frontier.push_back( triangle_edge<0>(tri,uv_tri) );
        in_edge_frontier.push_back( triangle_edge<1>(tri,uv_tri) );
        in_edge_frontier.push_back( triangle_edge<2>(tri,uv_tri) );

        add_triangle(t);
	#if CHECK_NULL_TRIANGLES
        // check if the UV-triangle has null area
        if (fabsf( triangle_area( texture_data[uv_tri.x], texture_data[uv_tri.y], texture_data[uv_tri.z] ) ) < NULL_AREA_TRIANGLE)
        {
			// mark the end of this chart
			output_offsets.push_back( (int)output_index.size() );
			return;
        }
	#endif	
		while (in_edge_frontier.empty() == false)
        {
            // sort the input edge frontier, so as to allow efficient search
            std::sort(in_edge_frontier.begin(), in_edge_frontier.end(), lt_edge());

            // loop through all the edges in the frontier and find their neighboring triangles.
            //
            // NOTE: by considering the double edges in the vertex- and UV- graphs, we automatically
            // skip triangles crossing a seam in UV space.
            //

            // empty the output
            out_edge_frontier.erase( out_edge_frontier.begin(), out_edge_frontier.end() );

            for (size_t i = 0; i < in_edge_frontier.size(); ++i)
            {
                const double_edge_type in_edge = in_edge_frontier[i];
                const size_t n_tris = edge_map.count( in_edge );

                const edge_type in_uv_edge = make_undirected_edge( in_edge.z, in_edge.w );

                edge_tri_map_type::const_iterator tri_begin = edge_map.find( in_edge );

                int in_tri = 0;
                {
                    // loop through the tris to find the "input" triangle we came from
                    // (note: this is only correct for manifold surfaces, where edges are shared by at most 2 triangles)
                    edge_tri_map_type::const_iterator it = tri_begin;
                    for (size_t n = 0; n < n_tris; ++n, ++it)
                    {
                        const int tri_id = it->second;

                        if (triangle_bitmap[tri_id] == false)
                        {
                            in_tri = tri_id;
                            break;
                        }
                    }
                }

                // compute the edge equation coefficients of the input edge
                const cugar::Vector2f in_uv0 = texture_data[ in_edge.z ];
                const cugar::Vector2f in_uv1 = texture_data[ in_edge.w ];
                const float3 in_edge_eq = edge_equation( in_uv0, in_uv1 );

                // evaluate the edge equation at the opposite vertex
                const cugar::Vector2f in_uv = opposite_uv_vertex( in_tri, in_uv_edge );
                //const float in_sgn = eval_edge_equation( in_edge_eq, in_uv );
                const float in_sgn = triangle_area( in_uv0, in_uv1, in_uv );

                // restart the loop
                edge_tri_map_type::const_iterator it = tri_begin;

                for (size_t n = 0; n < n_tris; ++n, ++it)
                {
                    const int tri_id = it->second;

                    // skip triangles which have already been used in a chart
                    if (triangle_bitmap[tri_id] == false)
                        continue;

                    const vertex_triangle out_tri    = vertex_indices[tri_id];
                    const texture_triangle out_uv_tri = texture_indices[tri_id];

                    // Check whether the output (uv) triangle overlaps the input one, i.e. if the UV parameterization folds on itself.
                    // In 2D, this is equivalent to cheking whether the vertices opposite to the given edge lie on the same side of the edge.

                    // evaluate the edge equation at the opposite vertex of the candidate triangle
                    const cugar::Vector2f out_uv = opposite_uv_vertex( tri_id, in_uv_edge );
                    //const float out_sgn = eval_edge_equation( in_edge_eq, out_uv );
                    const float out_sgn = triangle_area( in_uv0, in_uv1, out_uv );

                    // check whether the triangle has null area
                    if (fabsf( triangle_area( in_uv0, in_uv1, out_uv ) ) < NULL_AREA_TRIANGLE)
                        continue;

                    // check whether the signs are opposite - if not, this triangle shouldn't be included in the current chart
					//if (out_sgn * in_sgn >= 0.0f)
                    //    continue;
                    if ((out_sgn > 0.0f && in_sgn > 0.0f) ||
						(out_sgn < 0.0f && in_sgn < 0.0f))
                        continue;

                    // add this triangle to the current chart
                    add_triangle( tri_id );

                    // loop through its edges and add the ones which are not already in the input frontier to the output
                    const double_edge_type out_edge1 = triangle_edge<0>( out_tri, out_uv_tri );
                    const double_edge_type out_edge2 = triangle_edge<1>( out_tri, out_uv_tri );
                    const double_edge_type out_edge3 = triangle_edge<2>( out_tri, out_uv_tri );

                    if (std::binary_search( in_edge_frontier.begin(), in_edge_frontier.end(), out_edge1, lt_edge() ) == false) out_edge_frontier.push_back( out_edge1 );
                    if (std::binary_search( in_edge_frontier.begin(), in_edge_frontier.end(), out_edge2, lt_edge() ) == false) out_edge_frontier.push_back( out_edge2 );
                    if (std::binary_search( in_edge_frontier.begin(), in_edge_frontier.end(), out_edge3, lt_edge() ) == false) out_edge_frontier.push_back( out_edge3 );
                }
            }

            std::swap( in_edge_frontier, out_edge_frontier );
        }

        // mark the end of this chart
        output_offsets.push_back( (int)output_index.size() );
    }

    // return the UV vertex opposite to a given edge
    cugar::Vector2f opposite_uv_vertex(const int tri_id, const edge_type edge) const
    {
        const MeshStorage::texture_triangle tri = texture_indices[tri_id];

        if (make_undirected_edge( tri.x, tri.y ) == edge)
            return texture_data[ tri.z ];
        if (make_undirected_edge( tri.y, tri.z ) == edge)
            return texture_data[ tri.x ];

        return texture_data[ tri.y ];
    }
};

} // anonymous namespace

float pack(const uint32 n, const cugar::Vector2f* rects, cugar::Vector2f* transforms, uint8* flips);

cugar::Vector2f optional_flip(const bool flip, const cugar::Vector2f uv) { return flip ? uv.yx() : uv; }

void uv_fix(MeshStorage& mesh)
{
    // loop through all groups in the mesh, and break them in subgroups (charts) with non-overlapping UVs
    //
    const uint32_t num_triangles = mesh.getNumTriangles();
    const uint32_t num_groups = mesh.getNumGroups();

    const int* group_offsets   = mesh.getGroupOffsets();

    uint32_t max_group_size = 0;
    for (uint32_t g = 0; g < num_groups; ++g)
    {
        const uint32_t group_begin = group_offsets[g];
        const uint32_t group_end   = group_offsets[g+1];
        const uint32_t group_size  = group_end - group_begin;

        max_group_size = cugar::max( max_group_size, group_size );
    }

    //cugar::vector<cugar::host_tag, cugar::Vector2f> uv_points( max_group_size );
    //cugar::vector<cugar::host_tag, cugar::Bbox2f>   uv_bboxes( max_group_size );

    //fprintf(stderr, "   # groups: %u\n", num_groups);

    uv_chart_builder uv_charts;

    uv_charts.num_triangles   = num_triangles;
    uv_charts.num_vertices    = mesh.getNumVertices();
    uv_charts.num_uv_vertices = mesh.getNumTextureCoordinates();
    uv_charts.triangle_bitmap.resize( num_triangles, true );
    uv_charts.vertex_indices  = reinterpret_cast<MeshStorage::vertex_triangle*>(mesh.getVertexIndices());
    uv_charts.vertex_data     = reinterpret_cast<const MeshView::vertex_type*>(mesh.getVertexData());
    uv_charts.texture_indices = reinterpret_cast<MeshStorage::texture_triangle*>(mesh.getTextureCoordinateIndices());
    uv_charts.texture_data    = reinterpret_cast<const cugar::Vector2f*>(mesh.getTextureCoordinateData());
    uv_charts.output_index.reserve( max_group_size );
    uv_charts.output_offsets.reserve( max_group_size );

	// no texture coordinates, bail out
	if (!uv_charts.texture_indices ||
		!uv_charts.num_uv_vertices)
		return;

    uv_charts.output_offsets.resize(1);
    uv_charts.output_offsets[0] = 0;

    //uv_charts.merge_uvs();
    uv_charts.merge_vertices();

    for (uint32_t g = 0; g < num_groups; ++g)
    {
        const uint32_t group_begin = group_offsets[g];
        const uint32_t group_end   = group_offsets[g+1];
        const uint32_t group_size  = group_end - group_begin;
		FERMAT_ASSERT(group_begin <= group_end);
		FERMAT_ASSERT(group_end <= num_triangles);

        // build a map (edge -> triangles)
        //
        typedef int2 edge_type;
        typedef std::map<edge_type, uint32_t, lt_edge> edge_tri_map_type;

        uv_charts.edge_map.clear();

        for (uint32_t t = 0; t < group_size; ++t)
        {
			FERMAT_ASSERT(group_begin + t < num_triangles);
            const MeshStorage::vertex_triangle tri     = uv_charts.vertex_indices[group_begin + t];
            const MeshStorage::texture_triangle uv_tri = uv_charts.texture_indices[group_begin + t];
			FERMAT_ASSERT(uv_tri.x < uv_charts.num_uv_vertices);
			FERMAT_ASSERT(uv_tri.y < uv_charts.num_uv_vertices);
			FERMAT_ASSERT(uv_tri.z < uv_charts.num_uv_vertices);

            const double_edge_type edge0 = triangle_edge<0>( tri, uv_tri );
            const double_edge_type edge1 = triangle_edge<1>( tri, uv_tri );
            const double_edge_type edge2 = triangle_edge<2>( tri, uv_tri );

            uv_charts.edge_map.insert( std::make_pair( edge0, group_begin + t ) );
            uv_charts.edge_map.insert( std::make_pair( edge1, group_begin + t ) );
            uv_charts.edge_map.insert( std::make_pair( edge2, group_begin + t ) );
        }

        // decompose the current group into multiple charts
        for (uint32_t t = 0; t < group_size; ++t)
        {
            // starting from the given triangle, grow a chart by expanding its frontier until either a seam or an overlap is found
            uv_charts.grow_chart( group_begin + t );
        }

        // 
        /*
        for (uint32_t t = 0; t < group_size; ++t)
        {
            const texture_triangle tri = texture_indices[group_begin + t];

            uv_points[t] = (texture_data[tri.x] +
                            texture_data[tri.y] +
                            texture_data[tri.z]) / 3.0f;

            uv_bboxes[t].clear();
            uv_bboxes[t].insert( texture_data[tri.x] );
            uv_bboxes[t].insert( texture_data[tri.y] );
            uv_bboxes[t].insert( texture_data[tri.z] );
        }*/
    }

	float surface_area = 0.0f;
	cugar::Bbox3f scene_bbox;
	for (uint32 i = 0; i < num_triangles; ++i)
	{
		const MeshStorage::vertex_triangle tri = uv_charts.vertex_indices[i];
		const cugar::Vector3f vp0 = vertex_comp( uv_charts.vertex_data[tri.x] );
		const cugar::Vector3f vp1 = vertex_comp( uv_charts.vertex_data[tri.y] );
		const cugar::Vector3f vp2 = vertex_comp( uv_charts.vertex_data[tri.z] );

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;
		const float area = 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));

		surface_area += area;

		scene_bbox.insert(vp0);
		scene_bbox.insert(vp1);
		scene_bbox.insert(vp2);
	}

	const uint32 num_charts = uint32( uv_charts.output_offsets.size() - 1 );

	fprintf(stderr, "  # charts: %u\n", num_charts);

	// count how many distinct vertices we'll need, noticing that triangles in different groups cannot share vertices...
	typedef std::map<std::pair<uint32,uint32>, uint32> vertex_map_type;
	vertex_map_type					vertex_map;
	std::vector<float2>				lm_uvs;
	std::vector<MeshStorage::lightmap_triangle> lm_indices(num_triangles);
	std::vector<uint32>				lm_chart_offsets(num_charts+1);
	std::vector<cugar::Vector2f>	lm_rect_offsets(num_charts);
	std::vector<cugar::Vector2f>	lm_rect_widths(num_charts);
	std::vector<float>				lm_rect_extents(num_charts);
	std::vector<cugar::Vector2f>	lm_rect_trs(num_charts);
	std::vector<uint8>				lm_rect_flips(num_charts);

	// set the first offset
	lm_chart_offsets[0] = 0;

	for (uint32 g = 0; g < num_charts; ++g)
	{
        const uint32_t group_begin = uv_charts.output_offsets[g];
        const uint32_t group_end   = uv_charts.output_offsets[g+1];
        const uint32_t group_size  = group_end - group_begin;
		FERMAT_ASSERT(group_begin <= group_end);
		FERMAT_ASSERT(group_end <= num_triangles);

		cugar::Bbox2f uv_bbox;
		cugar::Bbox3f spatial_bbox;
		float         area = 0.0f;

        for (uint32_t t = 0; t < group_size; ++t)
        {
			FERMAT_ASSERT(group_begin + t < num_triangles);
			const uint32 tidx = uv_charts.output_index[group_begin + t];
			const MeshStorage::texture_triangle uv_tri = uv_charts.texture_indices[tidx];

			const MeshStorage::vertex_triangle tri = uv_charts.vertex_indices[tidx];
			const cugar::Vector3f vp0 = vertex_comp( uv_charts.vertex_data[tri.x] );
			const cugar::Vector3f vp1 = vertex_comp( uv_charts.vertex_data[tri.y] );
			const cugar::Vector3f vp2 = vertex_comp( uv_charts.vertex_data[tri.z] );
			spatial_bbox.insert(vp0);
			spatial_bbox.insert(vp1);
			spatial_bbox.insert(vp2);

			const cugar::Vector3f dp_du = vp0 - vp2;
			const cugar::Vector3f dp_dv = vp1 - vp2;

			area += 0.5f * cugar::length(cugar::cross(dp_du, dp_dv));

		#if CHECK_NULL_TRIANGLES
			// check if the UV-triangle has null area
			if (fabsf( triangle_area( uv_charts.texture_data[uv_tri.x], uv_charts.texture_data[uv_tri.y], uv_charts.texture_data[uv_tri.z] ) ) < NULL_AREA_TRIANGLE)
			{
				// 1. take the actual triangle coordinates, 
				// 2. project them to the triangle plane,
				// 3. scale them so as to match the relative surface area / edge lengths

				// build the local frame
				const cugar::Vector3f N        = cugar::normalize(cugar::cross(dp_du, dp_dv));
				const cugar::Vector3f tangent  = cugar::normalize(dp_du);
				const cugar::Vector3f binormal = cugar::cross(N, tangent);

				// project the triangle
				cugar::Vector2f uv2(0.0f);
				cugar::Vector2f uv0(dot(dp_du,tangent),0.0f);
				cugar::Vector2f uv1(dot(dp_dv,tangent),dot(dp_dv,binormal));

				// and rescale it
				const float u_edge_length = cugar::length(dp_du) / cugar::length(cugar::extents(scene_bbox));
				const float v_edge_length = cugar::length(dp_dv) / cugar::length(cugar::extents(scene_bbox));
				
				if (u_edge_length && v_edge_length)
				{
					uv0 *= u_edge_length / cugar::length(uv0);
					uv1 *= v_edge_length / cugar::length(uv1);
				}

				const int offset = (int)lm_uvs.size();
				lm_uvs.push_back(uv0);
				lm_uvs.push_back(uv1);
				lm_uvs.push_back(uv2);
				uv_bbox.insert(uv0);
				uv_bbox.insert(uv1);
				uv_bbox.insert(uv2);

				lm_indices[tidx] = MeshStorage::make_lightmap_triangle(offset,offset+1,offset+2);
			}
			else
		#endif
			{
				if (vertex_map.find(std::make_pair(g,uv_tri.x)) == vertex_map.end())
				{
					vertex_map.insert(std::make_pair(std::make_pair(g,uv_tri.x), uint32(lm_uvs.size())));
					const float2 uv = uv_charts.texture_data[uv_tri.x];
					lm_uvs.push_back(uv);

					uv_bbox.insert(uv);
				}
				if (vertex_map.find(std::make_pair(g,uv_tri.y)) == vertex_map.end())
				{
					vertex_map.insert(std::make_pair(std::make_pair(g,uv_tri.y), uint32(lm_uvs.size())));
					const float2 uv = uv_charts.texture_data[uv_tri.y];
					lm_uvs.push_back(uv);

					uv_bbox.insert(uv);
				}
				if (vertex_map.find(std::make_pair(g,uv_tri.z)) == vertex_map.end())
				{
					vertex_map.insert(std::make_pair(std::make_pair(g,uv_tri.z), uint32(lm_uvs.size())));
					const float2 uv = uv_charts.texture_data[uv_tri.z];
					lm_uvs.push_back(uv);

					uv_bbox.insert(uv);
				}

				vertex_map_type::const_iterator v0 = vertex_map.find(std::make_pair(g,uv_tri.x));
				vertex_map_type::const_iterator v1 = vertex_map.find(std::make_pair(g,uv_tri.y));
				vertex_map_type::const_iterator v2 = vertex_map.find(std::make_pair(g,uv_tri.z));

				lm_indices[tidx] = MeshStorage::make_lightmap_triangle(
					v0->second,
					v1->second,
					v2->second);
			}
		}

		lm_rect_offsets[g] = uv_bbox[0];
		lm_rect_widths[g]  = uv_bbox[1] - uv_bbox[0];

	//#define AREA_DENSITY
	#define EDGE_DENSITY

	#if defined(AREA_DENSITY)
		lm_rect_extents[g] = sqrtf( area );
	#else
		lm_rect_extents[g] = cugar::length(cugar::extents(spatial_bbox));
	#endif

		// mark the end of this chart
		lm_chart_offsets[g+1] = (uint32)lm_uvs.size();
	}
    fprintf(stderr, "  # verts: %u -> %u\n", (int)mesh.getNumTextureCoordinates(), (int)lm_uvs.size());

	// resize the charts appropriately so that they match their spatial extents
	for (uint32 g = 0; g < num_charts; ++g)
	{
		// as a simple heuristic we want to make the diagonal of the uv bbox proportional to the diagonal of the spatial bounding bbox
	#if defined(AREA_DENSITY)
		if (lm_rect_widths[g].x * lm_rect_widths[g].y > 1.0e-6f)
			lm_rect_extents[g] /= sqrtf( lm_rect_widths[g].x * lm_rect_widths[g].y );
		else
	#elif defined(EDGE_DENSITY)
		if (cugar::length( lm_rect_widths[g] ) > 1.0e-6f)
			lm_rect_extents[g] /= cugar::length( lm_rect_widths[g] );
		else
	#endif
			lm_rect_extents[g] = 1.0f;

		lm_rect_widths[g] *= lm_rect_extents[g];
	}

	const float scaling_factor = pack( num_charts, &lm_rect_widths[0], &lm_rect_trs[0], &lm_rect_flips[0] );

  #if 0
	fprintf(stderr, "rasterizing atlas... started\n");
	const uint32 res = 4096;
	std::vector<uchar4> rgba(res*res, make_uchar4(0,0,0,0));
	for (uint32 g = 0; g < num_charts; ++g)
	{
		uint32 x = cugar::quantize( lm_rect_trs[g].x / scaling_factor, res );
		uint32 y = cugar::quantize( lm_rect_trs[g].y / scaling_factor, res );

		uint32 w = cugar::quantize( lm_rect_widths[g].x / scaling_factor, res );
		uint32 h = cugar::quantize( lm_rect_widths[g].y / scaling_factor, res );

		if (lm_rect_flips[g])
			std::swap(w,h);

		const uchar4 c = make_uchar4(
			cugar::quantize(cugar::randfloat(g,0),255u),
			cugar::quantize(cugar::randfloat(g,1),255u),
			cugar::quantize(cugar::randfloat(g,2),255u),
			0 );

		for (uint32 j = y; j < cugar::min(y + h, res-1u); ++j)
			for (uint32 i = x; i < cugar::min(x + w, res-1u); ++i)
				rgba[j*res + i] = c;
	}
	cugar::write_tga("atlas.tga", res, res, (unsigned char*)&rgba[0], cugar::TGAPixels::RGBA);
	fprintf(stderr, "rasterizing atlas... done\n");
  #endif

	// apply the final chart packing transformations to each group
	for (uint32 g = 0; g < num_charts; ++g)
	{
        const uint32_t group_begin = lm_chart_offsets[g];
        const uint32_t group_end   = lm_chart_offsets[g+1];
		FERMAT_ASSERT(group_begin <= group_end);

		const bool flip = lm_rect_flips[g];

		const cugar::Vector2f offset      = optional_flip( flip, lm_rect_offsets[g] );
		const float           scaling     = lm_rect_extents[g];
		const cugar::Vector2f translation = lm_rect_trs[g];

		// loop through the vertices in this group/chart
        for (uint32_t i = group_begin; i < group_end; ++i)
        {
			// flip, offset and scale
			cugar::Vector2f uv = (optional_flip( flip, lm_uvs[i] ) - offset) * scaling;
			
			// translate and apply global scaling
			uv = (uv + translation) / scaling_factor;

			assert(uv.x >= 0.0f || uv.x <= 1.0f + 1.0e-6f);
			assert(uv.y >= 0.0f || uv.y <= 1.0f + 1.0e-6f);

			lm_uvs[i] = uv;
		}
	}

	mesh.alloc_lightmap((int)lm_uvs.size());
	mesh.m_lightmap_indices.copy_from( MeshStorage::LIGHTMAP_TRIANGLE_SIZE*lm_indices.size(), HOST_BUFFER, &(lm_indices[0].x) );
	mesh.m_lightmap_data.copy_from( 2*lm_uvs.size(), HOST_BUFFER, &(lm_uvs[0].x) );

    // reorder the mesh
    //mesh.reorder_triangles( &uv_charts.output_index[0] );
    //mesh.reset_groups( (int)uv_charts.output_offsets.size()-1, &uv_charts.output_offsets[0] );
}


struct Node
{
	static const uint32 INVALID = uint32(-1);

	Node() : child(INVALID), depth(0), parent(INVALID) {}

	Node(const cugar::Vector2f o, const cugar::Vector2f w, uint32 d, uint32 p = INVALID) : child(INVALID), org(o), width(w), depth(d), parent(p), max_free_area(w.x * w.y), max_free_width(w) {}

	uint32			child;
	cugar::Vector2f org;
	cugar::Vector2f width;
	uint32			depth;
	uint32			parent;
	float			max_free_area;
	cugar::Vector2f	max_free_width;
};

void update_free_area(uint32 node_id, std::vector<Node>& nodes)
{
	while (node_id != Node::INVALID)
	{
		Node& node = nodes[node_id];

		node.max_free_area		= cugar::max(nodes[node.child].max_free_area, nodes[node.child+1].max_free_area);
		node.max_free_width.x	= cugar::max(nodes[node.child].max_free_width.x, nodes[node.child+1].max_free_width.x);
		node.max_free_width.y	= cugar::max(nodes[node.child].max_free_width.y, nodes[node.child+1].max_free_width.y);

		node_id = node.parent;
	}
}

void split_leaf(const uint32 id, const cugar::Vector2f& rect, const uint32 node_id, std::vector<Node>& nodes)
{
	Node& node = nodes[node_id];

	// fetch the box in local registers
	const cugar::Vector2f node_org   = node.org;
	const cugar::Vector2f node_width = node.width;
	const uint32          depth      = node.depth;
	assert(rect.x <= node_width.x && rect.y <= node_width.y);

	// calculate the difference in extents
	const cugar::Vector2f delta = node_width - rect;

	const uint32 child_offset = node.child = uint32( nodes.size() );

	nodes.resize( child_offset + 2 );

	bool splitHorizontally = rect.x * delta.y > delta.x * rect.y;
							//(delta.x > delta.y);
	{
		// form the two new free rectangles.
		const cugar::Vector2f o_L(	node_org.x,
									node_org.y + rect.y );
		const cugar::Vector2f w_L(	splitHorizontally ? node_width.x : rect.x,
									node_width.y - rect.y );

		const cugar::Vector2f o_R(	node_org.x + rect.x,
									node_org.y );
		const cugar::Vector2f w_R(	node_width.x - rect.x,
									splitHorizontally ? rect.y : node_width.y );

		nodes[child_offset+0] = Node( o_L, w_L, depth+1, node_id );
		nodes[child_offset+1] = Node( o_R, w_R, depth+1, node_id );
		//assert((nodes[child_offset+0].org.x >= node_org.x) && (nodes[child_offset+0].org.x + nodes[child_offset+0].width.x <= node_org.x + node_width.x));
		//assert((nodes[child_offset+0].org.y >= node_org.y) && (nodes[child_offset+0].org.y + nodes[child_offset+0].width.y <= node_org.y + node_width.y));
		//assert((nodes[child_offset+1].org.x >= node_org.x) && (nodes[child_offset+1].org.x + nodes[child_offset+1].width.x <= node_org.x + node_width.x));
		//assert((nodes[child_offset+1].org.y >= node_org.y) && (nodes[child_offset+1].org.y + nodes[child_offset+1].width.y <= node_org.y + node_width.y));
	}

	// update the amount of free area
	update_free_area(node_id, nodes);

#if 0
	// sort the two children by area
	if (nodes[child_offset + 0].max_free_area > nodes[child_offset + 1].max_free_area)
		std::swap(nodes[child_offset + 0], nodes[child_offset + 1]);
#endif
}

typedef std::pair<uint32, float> scored_node;

scored_node find_best_node(const uint32 id, const cugar::Vector2f& rect, uint32 node_id, const std::vector<Node>& nodes, std::vector<uint32>& stack)
{
	const scored_node bad_node(Node::INVALID, 1.0e8f);

	// reset the stack
	stack.erase( stack.begin(), stack.end() );

	// push a sentinel
	stack.push_back(Node::INVALID);

	scored_node best_node = bad_node;

	while (node_id != Node::INVALID)
	{
		const Node& node = nodes[node_id];

		// fetch the box in local registers
		const cugar::Vector2f node_width = node.width;

		// calculate the difference in extents
		const cugar::Vector2f delta = node_width - rect;

		// check whether we need to discard this node
		bool discard;

		// discard this node if too small
		discard = delta.x < 0.0f || delta.y < 0.0f;

		// discard if there is no chance of finding enough free space
		discard = discard || (rect.x * rect.y > node.max_free_area ||
							  rect.x > node.max_free_width.x ||
							  rect.y > node.max_free_width.y);

		// prune this branch if there is no chance of beating the current best
		const float score = cugar::max(delta.x,delta.y);
		discard = discard || (score > best_node.second);

		// pop the stack if we need to discard this branch
		if (discard)
		{
			// pop the stack
			node_id = stack.back();
			stack.pop_back();
			continue;
		}

		if (node.child != Node::INVALID)
		{
			// descend into the first child and push the other on the stack
			node_id = node.child+0;
			stack.push_back(node.child+1);
		}
		else
		{
			// found an empty leaf
			if (score < best_node.second)
				best_node = scored_node(node_id,score);

			// pop the stack
			node_id = stack.back();
			stack.pop_back();
		}
	}
	return best_node;
}

struct RectCompare
{
	bool operator() (const std::pair<cugar::Vector2f,uint32> r1, const std::pair<cugar::Vector2f,uint32> r2) const
	{
		return r1.first.x + r1.first.y > r2.first.x + r2.first.y;
		//return cugar::max(r1.first.x, r1.first.y) > cugar::max(r2.first.x, r2.first.y);
	}
};

// return a global scale factor
float pack(const uint32 n, const cugar::Vector2f* rects, cugar::Vector2f* transforms, uint8* flips)
{
	float max_width = 0.0f;
	float min_width = 1.0e8f;
	float total_area = 0.0f;
	for (uint32 i = 0; i < n; ++i)
	{
		total_area += rects[i].x * rects[i].y;
		max_width = cugar::max3(max_width, rects[i].x, rects[i].y);
		if (rects[i].x > 0.0f && 
			rects[i].y > 0.0f)
			min_width = cugar::min3(min_width, rects[i].x, rects[i].y);
	}

	float slack = 1.2f;
	float width = cugar::max( sqrtf(total_area), max_width ) * slack;

	std::vector< std::pair<cugar::Vector2f,uint32> > ordered_rects(n);
	for (uint32 i = 0; i < n; ++i)
		ordered_rects[i] = std::make_pair(rects[i],i);

	std::sort(ordered_rects.begin(), ordered_rects.end(), RectCompare());

	// assume the total resolution was 512 x 512 and add a border pixel
	const float border = width / 512u;

#if 0
	// assume the total resolution was 4k
	uint32 n_pixels = 0;
	for (uint32 i = 0; i < n; ++i)
	{
		uint32 wx = cugar::max( cugar::quantize( rects[i].x / width, 4096u ), 1u );
		uint32 wy = cugar::max( cugar::quantize( rects[i].y / width, 4096u ), 1u );

		n_pixels += wx * wy;
	}
    fprintf(stderr, "  n_pixels: %u\n", n_pixels);

	rbp::GuillotineBinPack gpacker(4096u, 4096u);
	for (uint32 i = 0; i < n; ++i)
	{
		uint32 wx = cugar::max( cugar::quantize( ordered_rects[i].first.x / width, 4096u ), 1u );
		uint32 wy = cugar::max( cugar::quantize( ordered_rects[i].first.y / width, 4096u ), 1u );

		rbp::Rect r = gpacker.Insert(wx,wy,true,rbp::GuillotineBinPack::RectBestShortSideFit,rbp::GuillotineBinPack::SplitMinimizeArea);
		if (r.width == 0 && r.height == 0)
		{
			fprintf(stderr, "failed to pack %u\n", i);
			exit(0);
		}
	}
	fprintf(stderr, "  guillotine: %f\n", gpacker.Occupancy());
#endif
	std::vector<Node> nodes;

	uint32 n_bins = 1;
	uint32 offset = 0;

	std::vector<uint32> stack;

	while (1)
	{
	    fprintf(stderr, "  width: %f (max: %f; min: %f)\n", width, max_width, min_width);
		nodes.resize(1);
		nodes[0] = Node( cugar::Vector2f(0.0f), cugar::Vector2f(width), 0u );

		float used_area = 0.0f;

		bool success = true;
		for (uint32 i = offset; i < n; ++i)
		{
			cugar::Vector2f rect = cugar::max(ordered_rects[i].first, min_width);

			// leave some border space
			rect.x += border;
			rect.y += border;

			const uint32 id = ordered_rects[i].second;

			// try to pack the rectangle unflipped
			scored_node score_normal  = find_best_node( id, rect, 0u, nodes, stack );
			scored_node score_flipped = find_best_node( id, cugar::Vector2f(rect.y, rect.x), 0u, nodes, stack );

			uint32 node_id = score_normal.first;
			bool   flip    = false;
			if (score_flipped.second < score_normal.second)
			{
				node_id = score_flipped.first;
				flip    = true;
			}

			if (node_id != Node::INVALID)
			{
				// mark the placement offset of this rectangle
				transforms[id] = nodes[node_id].org + cugar::Vector2f(width * (n_bins-1),0);

			#if 0
				if (transforms[id].x < 0.0f || transforms[id].x + optional_flip( flip, rect ).x - width > width * 1.0e-3f ||
					transforms[id].y < 0.0f || transforms[id].y + optional_flip( flip, rect ).y - width > width * 1.0e-3f)
				{
					fprintf(stderr, "%u (%u) : n[%f, %f][%f, %f], r[%f, %f]\n  node[%u], flip[%u]\n", i, id, 
						nodes[node_id].org.x, nodes[node_id].org.y,
						nodes[node_id].width.x, nodes[node_id].width.y,
						flip ? rect.y : rect.x,
						flip ? rect.x : rect.y,
						node_id,
						uint32(flip));
					fgetc(stdin);
				}
			#endif
				assert(
					transforms[id].x >= width * (n_bins-1) || transforms[id].x + optional_flip( flip, rect ).x - width * (n_bins-1) <= width * 1.0e-3f ||
					transforms[id].y >= 0.0f || transforms[id].y + optional_flip( flip, rect ).y - width <= width * 1.0e-3f);

				// mark the flipping bit
				flips[id] = flip;

				// keep track of the used area
				used_area += rect.x * rect.y;

				// and split the empty leaf
				split_leaf( id, flip ? cugar::Vector2f(rect.y, rect.x) : rect, node_id, nodes );
			}
			else
			{
			    fprintf(stderr, "  packed [%u,%u)\n    max free area: %f, w: %f, h: %f\n    used area: %f\n", offset, i, nodes[0].max_free_area, nodes[0].max_free_width.x, nodes[0].max_free_width.y, used_area / (width*width));
				// restart
				success = false;
				offset = i;
				++n_bins;
				break;
			}
		}

		if (success)
		{
			fprintf(stderr, "  packed [%u,%u)\n    max free area: %f, w: %f, h: %f\n    used area: %f\n", offset, n, nodes[0].max_free_area, nodes[0].max_free_width.x, nodes[0].max_free_width.y, used_area / (width*width));
			break;
		}

		//width *= slack;
	}
	// repack the bins so as to form a square, with ceil(sqrt(n_bins)) on each side
	if (n_bins > 1)
	{
		const uint32 n_bins_per_side = (uint32)ceilf(sqrtf(float(n_bins)));
		for (uint32 i = 0; i < n; ++i)
		{
			const uint32 in_bin = uint32(transforms[i].x / width);
			const uint32 out_bin_x = in_bin % n_bins_per_side;
			const uint32 out_bin_y = in_bin / n_bins_per_side;

			transforms[i].x = transforms[i].x - width * in_bin + width * out_bin_x;
			transforms[i].y = transforms[i].y + width * out_bin_y;
		}
		fprintf(stderr, "  # bins: %u -> %u x %u\n", n_bins, n_bins_per_side, n_bins_per_side);
		return width * n_bins_per_side;
	}
	return width;
}