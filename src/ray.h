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

#pragma once

#include "types.h"

#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>

///@addtogroup Fermat
///@{

/// 
///  Ray struct
///
struct Ray
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

  float3 origin;
  float  tmin;
  float3 dir;
  float  tmax;
};

/// 
///  Ray struct
///
struct MaskedRay
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_MASK_DIRECTION_TMAX;

  float3 origin;
  uint32 mask;
  float3 dir;
  float  tmax;
};

///
/// Ray-tracing hit
///
struct Hit
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;

  float t;
  int   triId;
  float u;
  float v;
};

///
/// Ray-tracing hit, with instancing support
///
struct HitInstancing
{
  static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

  float t;
  int   triId;
  int   instId;
  float u;
  float v;
};

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Ray make_ray(
	const float3 origin,
	const float3 dir,
	const float tmin,
	const float tmax)
{
	Ray r;
	r.origin	= origin;
	r.tmin		= tmin;
	r.dir		= dir;
	r.tmax		= tmax;
	return r;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
MaskedRay make_ray(
	const float3 origin,
	const float3 dir,
	const uint32 mask,
	const float tmax)
{
	MaskedRay r;
	r.origin	= origin;
	r.mask		= mask;
	r.dir		= dir;
	r.tmax		= tmax;
	return r;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
Hit make_hit(
	const float t,
	const int   triId,
	const float u,
	const float v)
{
	Hit r;
	r.t		= t;
	r.triId	= triId;
	r.u		= u;
	r.v		= v;
	return r;
}

///@} Fermat
