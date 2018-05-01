/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

// ------------------------------------------------------------------------- //
//
// Declaration of classes used to store intersections.
//
// ------------------------------------------------------------------------- //

#include <types.h>
#include <cugar/linalg/vector.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/spherical/mappings.h>
#include <cuda_fp16.h>

///@addtogroup Fermat
///@{

///@addtogroup VertexGeometryModule
///@{

///
/// Vertex geometry.
///
struct VertexGeometry : public cugar::DifferentialGeometry
{
	cugar::Vector3f	position;
	cugar::Vector4f	texture_coords;
};

struct VertexGeometryId
{
	cugar::Vector2f uv;
	uint32			prim_id;

	FERMAT_HOST_DEVICE
	VertexGeometryId() {}

	FERMAT_HOST_DEVICE
	VertexGeometryId(const uint32 _prim_id, const float _u, const float _v) : prim_id(_prim_id), uv(_u, _v) {}

	FERMAT_HOST_DEVICE
	VertexGeometryId(const uint32 _prim_id, const cugar::Vector2f _uv) : prim_id(_prim_id), uv(_uv) {}
};

/// pack a normalized direction vector in 32-bits
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
uint32 pack_direction(const cugar::Vector3f& dir)
{
	const cugar::Vector2f sdir = cugar::uniform_sphere_to_square(dir);

	return cugar::quantize(sdir.x, 0xFFFFu) + (cugar::quantize(sdir.y, 0xFFFFu) << 16);
}

/// unpack a 32-bits normalized direction vector
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f unpack_direction(const uint32 packed_dir)
{
	const cugar::Vector2f sdir(
		float((packed_dir >>  0) & 0xFFFFu) / float(0xFFFFu),
		float((packed_dir >> 16) & 0xFFFFu) / float(0xFFFFu));

	return cugar::uniform_square_to_sphere( sdir );
}

/// pack a normalized direction vector in 32-bits
///
FERMAT_DEVICE FERMAT_FORCEINLINE
uint32 pack_hemispherical_direction(const cugar::Vector3f& dir)
{
	// Spheremap Transform
	cugar::Vector2f enc = cugar::normalize(dir.xy()) * sqrtf(-dir.z*0.5f + 0.5f);
	//enc = enc*0.5f + cugar::Vector2f(0.5f);	// compress from [-1:1] to [0:1]
	half2 h = __floats2half2_rn(enc.x, enc.y);
	return cugar::binary_cast<uint32>(h);
}
/// unpack a 32-bits normalized direction vector
///
FERMAT_DEVICE FERMAT_FORCEINLINE
cugar::Vector3f unpack_hemispherical_direction(const uint32 packed_dir)
{
	// Inverse Spheremap Transform
	float2 u = __half22float2( cugar::binary_cast<half2>(packed_dir) );

	//u.x = 2 * u.x - 1;						// decompress from [0:1] to [-1:1]
	//u.y = 2 * u.y - 1;

	const float l = 1 - u.x*u.x - u.y*u.y;
	const float m = 2 * sqrtf(l);

	return cugar::Vector3f(u.x * m, u.y * m, l * 2 - 1);
}

///@} VertexGeometryModule
///@} Fermat
