/*
 * Fermat
 *
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
