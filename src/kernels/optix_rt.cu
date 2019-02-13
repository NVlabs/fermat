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

#include <ray.h>

//-----------------------------------------------------------------------------
// Ray generation program: plain ray tracing kernel, without masking
//-----------------------------------------------------------------------------

rtBuffer<float4>	g_ray_buffer;
rtBuffer<float4>	g_hit_buffer;

RT_PROGRAM
void program_tmin_ray_generation()
{
	const uint32 idx = g_launch_index;

	const float4 f1 = g_ray_buffer[idx*2 + 0];
	const float4 f2 = g_ray_buffer[idx*2 + 1];
	
	Ray ray;
	ray.origin.x = f1.x;
	ray.origin.y = f1.y;
	ray.origin.z = f1.z;
	ray.tmin     = f1.w;

	ray.dir.x = f2.x;
	ray.dir.y = f2.y;
	ray.dir.z = f2.z;
	ray.tmax  = f2.w;

	// trace the ray generated at the previous bounce
	Payload payload(
		-1.0f,	// t,
		-1,		// triangle id
		0.0f,	// u
		0.0f,	// v
		0x0u );	// mask

	rtTrace( g_top_object, optix::make_Ray(ray.origin, ray.dir, 2u /* ray type 2: no masking */, ray.tmin, ray.tmax), payload );

	const Hit hit = Hit(payload);

	g_hit_buffer[idx] = make_float4(
		hit.t,
		__uint_as_float(hit.triId),
		hit.u,
		hit.v
	);
}


//-----------------------------------------------------------------------------
// Ray generation program: plain ray tracing kernel, with masking
//-----------------------------------------------------------------------------

RT_PROGRAM
void program_masked_ray_generation()
{
	const uint32 idx = g_launch_index;

	const float4 f1 = g_ray_buffer[idx*2 + 0];
	const float4 f2 = g_ray_buffer[idx*2 + 1];
	
	MaskedRay ray;
	ray.origin.x = f1.x;
	ray.origin.y = f1.y;
	ray.origin.z = f1.z;
	ray.mask     = __float_as_uint(f1.w);

	ray.dir.x = f2.x;
	ray.dir.y = f2.y;
	ray.dir.z = f2.z;
	ray.tmax  = f2.w;

	// trace the ray generated at the previous bounce
	Payload payload(
		-1.0f,	// t,
		-1,		// triangle id
		0.0f,	// u
		0.0f,	// v
		ray.mask );	// mask

	rtTrace( g_top_object, optix::make_Ray(ray.origin, ray.dir, 0u /* ray type 0: masking */, 0.0f, ray.tmax), payload );

	const Hit hit = Hit(payload);

	g_hit_buffer[idx] = make_float4(
		hit.t,
		__uint_as_float(hit.triId),
		hit.u,
		hit.v
	);
}

//-----------------------------------------------------------------------------
// Ray generation program: shadow ray tracing kernel, with masking, reporting
// the first valid hit (not necessarily the closest)
//-----------------------------------------------------------------------------

RT_PROGRAM
void program_masked_shadow_ray_generation()
{
	const uint32 idx = g_launch_index;

	const float4 f1 = g_ray_buffer[idx*2 + 0];
	const float4 f2 = g_ray_buffer[idx*2 + 1];
	
	MaskedRay ray;
	ray.origin.x = f1.x;
	ray.origin.y = f1.y;
	ray.origin.z = f1.z;
	ray.mask     = __float_as_uint(f1.w);

	ray.dir.x = f2.x;
	ray.dir.y = f2.y;
	ray.dir.z = f2.z;
	ray.tmax  = f2.w;

	// trace the ray generated at the previous bounce
	ShadowPayload hit( ray.mask, false );

	rtTrace( g_top_object, optix::make_Ray(ray.origin, ray.dir, 1u /* ray type 1: shadow w/ masking */, 0.0f, ray.tmax), hit );

	// TODO / FIXME: for now all of the fields are invalid, as they just fake a valid intersection
	g_hit_buffer[idx] = make_float4(
		hit ? 1.0f : -1.0f,
		hit ? __uint_as_float(1u) : __uint_as_float(-1u),
		0.0f,
		0.0f
	);
}

//-----------------------------------------------------------------------------
// Ray generation program: shadow ray tracing kernel, with masking, reporting
// just a binary hit / no hit bit
//-----------------------------------------------------------------------------

rtDeclareVariable(uint32*, g_binary_hits, ,   );

RT_PROGRAM
void program_masked_shadow_binary_ray_generation()
{
	const uint32 idx = g_launch_index;

	const float4 f1 = g_ray_buffer[idx*2 + 0];
	const float4 f2 = g_ray_buffer[idx*2 + 1];
	
	MaskedRay ray;
	ray.origin.x = f1.x;
	ray.origin.y = f1.y;
	ray.origin.z = f1.z;
	ray.mask     = __float_as_uint(f1.w);

	ray.dir.x = f2.x;
	ray.dir.y = f2.y;
	ray.dir.z = f2.z;
	ray.tmax  = f2.w;

	// trace the ray generated at the previous bounce
	ShadowPayload hit( ray.mask, false );

	rtTrace( g_top_object, optix::make_Ray(ray.origin, ray.dir, 1u /* ray type 1: shadow w/ masking */, 0.0f, ray.tmax), hit );

	const uint32 word_idx = idx >> 5;
	const uint32 word_bit = idx & 31u;

	const uint32 word_mask = 1u << word_bit;

	if (hit) atomicOr(  g_binary_hits + word_idx,  word_mask );
	else     atomicAnd( g_binary_hits + word_idx, ~word_mask );
}
