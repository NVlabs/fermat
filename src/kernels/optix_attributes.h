/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#pragma once

#include "optix_payload.h"
#include <MeshView.h>
#include <texture_view.h>

rtDeclareVariable( unsigned int, model_id,, );

rtDeclareVariable( float2, barycentrics, attribute rtTriangleBarycentrics, );

rtBuffer<int4>		g_index_buffer;
#if 0
rtBuffer<float3>	g_vertex_buffer;
rtBuffer<int3>		g_normal_index_buffer;
rtBuffer<float3>	g_normal_buffer;
rtBuffer<int3>		g_texture_index_buffer;
rtBuffer<float2>	g_texture_buffer;
rtBuffer<int>		g_material_index_buffer;

rtBuffer<MeshMaterial>	g_materials;
rtBuffer<MipMapView>	g_textures;

// setup the differential geometry
//
__device__
void setup_differential_geometry(const uint32 tri_id, const float u, const float v, VertexGeometry* geom, float* pdf = 0)
{
	// setup the geometric normal
	{
		const int4 tri = g_index_buffer[tri_id];
		const cugar::Vector3f vp0 = g_vertex_buffer[tri.x];
		const cugar::Vector3f vp1 = g_vertex_buffer[tri.y];
		const cugar::Vector3f vp2 = g_vertex_buffer[tri.z];

		const cugar::Vector3f dp_du = vp0 - vp2;
		const cugar::Vector3f dp_dv = vp1 - vp2;

		geom->normal_g = cugar::normalize(cugar::cross(dp_du, dp_dv));

		if (pdf)
			*pdf = 2.0f / cugar::length(cugar::cross(dp_du, dp_dv));
	}

	if (g_normal_index_buffer.size() && g_normal_buffer.size())
	{
		const int3 tri = g_normal_index_buffer[tri_id];
		const cugar::Vector3f vn0 = tri.x >= 0 ? g_normal_buffer[tri.x] : geom->normal_g;
		const cugar::Vector3f vn1 = tri.y >= 0 ? g_normal_buffer[tri.y] : geom->normal_g;
		const cugar::Vector3f vn2 = tri.z >= 0 ? g_normal_buffer[tri.z] : geom->normal_g;

		geom->normal_s = cugar::normalize(vn2 * (1.0f - u - v) + vn0 * u + vn1 * v);
		geom->tangent  = cugar::orthogonal(geom->normal_s);
		geom->binormal = cugar::cross(geom->normal_s, geom->tangent);
	}
	else
	{
		geom->normal_s = geom->normal_g;
		geom->tangent  = cugar::orthogonal(geom->normal_g);
		geom->binormal = cugar::cross(geom->normal_g, geom->tangent);
	}

	if (g_texture_index_buffer.size() && g_texture_buffer.size())
	{
		const int3 tri = g_texture_index_buffer[tri_id];
		const cugar::Vector2f vt0 = tri.x >= 0 ? g_texture_buffer[tri.x] : make_float2(1,0);
		const cugar::Vector2f vt1 = tri.y >= 0 ? g_texture_buffer[tri.y] : make_float2(0,1);
		const cugar::Vector2f vt2 = tri.z >= 0 ? g_texture_buffer[tri.z] : make_float2(0,0);

		const cugar::Vector2f st = vt2 * (1.0f - u - v) + vt0 * u + vt1 * v;
		geom->texture_coords = cugar::Vector4f(st.x, st.y, 0.0f, 0.0f);
	}
	else
		geom->texture_coords = cugar::Vector4f(u, v, 0.0f, 0.0f);
}
#endif