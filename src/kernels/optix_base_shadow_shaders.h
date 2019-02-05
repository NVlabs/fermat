/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "optix_payload.h"
#include "optix_common_variables.h"
#include "optix_attributes.h"


rtDeclareVariable(ShadowPayload,	g_prd,	rtPayload,      );

//-----------------------------------------------------------------------------
// Closest hit
//-----------------------------------------------------------------------------
__device__
void base_shadow_closest_hit()
{
	g_prd.set_hit( true );
}

//-----------------------------------------------------------------------------
// Any hit
//-----------------------------------------------------------------------------
__device__
void base_shadow_any_hit()
{
	const int primIdx = rtGetPrimitiveIndex();

	const uint32 triangleMask = g_index_buffer[primIdx].w;

	if (g_prd.mask() & triangleMask)
        rtIgnoreIntersection();

	g_prd.set_hit( true );
	rtTerminateRay();
}

//-----------------------------------------------------------------------------
// Miss
//-----------------------------------------------------------------------------
__device__
void base_shadow_miss()
{
	g_prd.set_hit( false );
}
