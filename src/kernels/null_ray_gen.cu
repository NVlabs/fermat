/******************************************************************************
 * Copyright 2018 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "optix_common_variables.h"


//-----------------------------------------------------------------------------
// Ray generation
//-----------------------------------------------------------------------------
RT_PROGRAM
void program_ray_generation()
{
	printf("executing launch index %u\n", g_launch_index );
}
