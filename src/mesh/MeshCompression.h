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

#include <MeshView.h>
#include <cugar/basic/types.h>
#include <cuda_fp16.h>

FERMAT_HOST_DEVICE inline
uint32 compress_tex_coord(const float2 t, const float2 tbias, const float2 tscale)
{
	const float2 tn = make_float2(
		(t.x - tbias.x) / tscale.x,
		(t.y - tbias.y) / tscale.y );
  #if TEX_COORD_COMPRESSION_MODE == TEX_COORD_COMPRESSION_HALF
	const half2 h = __floats2half2_rn(tn.x, tn.y);
	const __half2_raw hr = h;
  #else
	return uint32(tn.x * (1u << 15)) | (uint32(tn.y * (1u << 15)) << 16);
  #endif
	return uint32(hr.x) | uint32(hr.y) << 16;
}

#if defined(__CUDACC__)

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float2 decompress_tex_coord(const MeshView& mesh, const uint32 packed_tex_coord)
{
#if TEX_COORD_COMPRESSION_MODE == TEX_COORD_COMPRESSION_FIXED
	const float dn = 0.000030517578125f; // 1 / (1u << 15)
	const float2 tn = make_float2(
		float(packed_tex_coord & 0xFFFF) * dn,
		float(packed_tex_coord >> 16)	 * dn
		);
#else
	const half2 h = cugar::binary_cast<__half2_raw>(packed_tex_coord);
	const float2 tn = __half22float2(h);
#endif
	return make_float2(
		tn.x * mesh.tex_scale.x + mesh.tex_bias.x,
		tn.y * mesh.tex_scale.y + mesh.tex_bias.y );
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float2 decompress_lightmap_coord(const MeshView& mesh, const uint32 packed_tex_coord)
{
#if TEX_COORD_COMPRESSION_MODE == TEX_COORD_COMPRESSION_FIXED
	const float dn = 0.000030517578125f; // 1 / (1u << 15)
	const float2 tn = make_float2(
		float(packed_tex_coord & 0xFFFF) * dn,
		float(packed_tex_coord >> 16)	 * dn
		);
#else
	const half2 h = cugar::binary_cast<__half2_raw>(packed_tex_coord);
	const float2 tn = __half22float2(h);
#endif
	return make_float2(
		tn.x * mesh.lm_scale.x + mesh.lm_bias.x,
		tn.y * mesh.lm_scale.y + mesh.lm_bias.y );
}

#else

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float2 decompress_tex_coord(const MeshView& mesh, const uint32 packed_tex_coord)
{
	assert(0);
	return make_float2(0,0);
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float2 decompress_lightmap_coord(const MeshView& mesh, const uint32 packed_tex_coord)
{
	assert(0);
	return make_float2(0,0);
}

#endif