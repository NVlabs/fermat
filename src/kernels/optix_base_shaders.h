/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "optix_payload.h"
#include "optix_common_variables.h"
#include "optix_attributes.h"


rtDeclareVariable(Payload,	g_prd,	rtPayload,      );

//-----------------------------------------------------------------------------
// Closest hit
//-----------------------------------------------------------------------------
__device__
void base_closest_hit()
{
	const int primIdx = rtGetPrimitiveIndex();

	g_prd.set_triangle_id( primIdx );
    //g_prd.set_model_id( model_id );

	// NOTE: we are switching triangle parameterization
	const float bx = barycentrics.x;
	const float by = barycentrics.y;

	const float u = 1.0f - bx - by;
	const float v = bx;

	g_prd.set_uv( u, v );
	g_prd.set_t( g_hit_t );

	//// calculate the intersection point
	//g_prd.geom.position = g_ray.origin + g_ray.direction * g_hit_t;
	//
	//// setup the differential geometry
	//setup_differential_geometry( primIdx, u, v, &g_prd.geom );
}

//-----------------------------------------------------------------------------
// Any hit
//-----------------------------------------------------------------------------
__device__
void base_any_hit()
{
	const int primIdx = rtGetPrimitiveIndex();

	const uint32 triangleMask = g_index_buffer[primIdx].w;

	if (g_prd.mask() & triangleMask)
        rtIgnoreIntersection();
}

//-----------------------------------------------------------------------------
// Miss
//-----------------------------------------------------------------------------
__device__
void base_miss()
{
    g_prd.set_triangle_id( -1 );
    //g_prd.set_model_id( -1 );
	g_prd.set_uv( 0.0f, 0.0f );
	g_prd.set_t( -1.0f );
}
