/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#pragma once

#include "optix_payload.h"
#include "renderer_view.h"

rtDeclareVariable(unsigned int,         g_launch_index,    rtLaunchIndex,  );
rtDeclareVariable(uint2,                g_launch_index_2d, rtLaunchIndex,  );
rtDeclareVariable(rtObject,             g_top_object,                ,  );

rtDeclareVariable(optix::Ray,           g_ray,          rtCurrentRay,   );
rtDeclareVariable(float,				g_hit_t,        rtIntersectionDistance,   );

rtDeclareVariable(RenderingContextView,	g_renderer,		,   );

//rtBuffer<Ray,1> g_in_ray_buffer;
//rtBuffer<Hit,1> g_out_hit_buffer;
