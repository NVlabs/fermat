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

#pragma once

#ifndef OPTIX_COMPILATION
#define OPTIX_COMPILATION
#endif

#ifndef WIN32
#define WIN32
#endif

#include <vector_types.h>

#include <vertex.h>
#include <ray.h>

/// Hit structure currently used by Fermat
///
struct Payload
{
	uint4 packed;
 
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	Payload() {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	Payload(const float _t, const int32 _tri_id, const float _u, const float _v, const uint8 _mask)
	{
		set_t( _t );
		set_triangle_id( _tri_id );
		set_uv( _u, _v );
		set_mask( _mask );
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_t(const float _t) { packed.x = __float_as_uint(_t); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float t() const { return __uint_as_float(packed.x); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_triangle_id(const int32 _tri_id) { packed.y = uint32(_tri_id); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	int32 triangle_id() const { return int32(packed.y); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_uv(const float _u, const float _v) { packed.z = cugar::binary_cast<uint32>( __floats2half2_rn(_u,_v) ); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float2 uv() const { return __half22float2( cugar::binary_cast<__half2>( packed.z ) ); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float t(const float _t) { return __uint_as_float(packed.x); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_mask(const uint8 _mask) { packed.w = uint32(_mask); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	uint32 mask() const { return uint32(packed.w); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	operator Hit() const
	{
		return make_hit( t(), triangle_id(), uv().x, uv().y );
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	operator bool() const { return t() >= 0.0f; }
};

/// Shadow payload structure currently used by Fermat
///
struct ShadowPayload
{
	uint2 packed;

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	ShadowPayload() {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	ShadowPayload(const uint32 _mask, const bool _hit)
	{
		set_mask( _mask );
		set_hit( _hit );
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	uint32 mask() const { return uint32(packed.x); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_mask(const uint8 _mask) { packed.x = _mask; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void set_hit(const bool _hit) { packed.y =  _hit ? 1u : 0u; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	operator bool() const { return packed.y ? true : false; }
};
