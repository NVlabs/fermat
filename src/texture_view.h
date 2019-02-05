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
#include "texture_reference.h"

#include <cugar/basic/types.h>
#include <cugar/linalg/vector.h>

///@addtogroup Fermat
///@{

///@addtogroup TexturesModule
///@{

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float4 texture_load(const float4* tex)
{
#ifdef FERMAT_DEVICE_COMPILATION
	return __ldg(tex);
#else
	return *tex;
#endif
}

/// A \ref TextureStorage "Texture" view object to be used within CUDA kernels
///
struct TextureView
{
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 pixel)		  { return c[pixel]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 pixel) const { return texture_load(c + pixel); }
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 x, const uint32 y)			{ return c[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 x, const uint32 y) const	{ return texture_load(c + y*res_x + x); }
	FERMAT_HOST_DEVICE const float4* ptr() const { return c;  }
	FERMAT_HOST_DEVICE       float4* ptr()       { return c; }

	float4* c;
	uint32	res_x;
	uint32	res_y;
};

/// A \ref MipMapStorageAnchor "Mip-Map" view object to be used within CUDA kernels
///
struct MipMapView
{
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 pixel, const uint32 lod)       { return levels[lod](pixel); }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 pixel, const uint32 lod) const { return reinterpret_cast<const TextureView*>(levels)[lod](pixel); }
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 x, const uint32 y, const uint32 lod)       { return levels[lod](x,y); }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 x, const uint32 y, const uint32 lod) const { return reinterpret_cast<const TextureView*>(levels)[lod](x,y); }

	TextureView* levels;
	uint32		 n_levels;
	uint32		 res_x;
	uint32		 res_y;
};

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector4f texture_lookup(float4 st, const TextureReference texture_ref, const MipMapView* textures, const float4 default_value)
{
	if (texture_ref.texture == uint32(-1) || textures[texture_ref.texture].n_levels == 0)
		return default_value;

	st.x *= texture_ref.scaling.x;
	st.y *= texture_ref.scaling.y;

	st.x = cugar::mod(st.x, 1.0f);
	st.y = cugar::mod(st.y, 1.0f);

	const MipMapView texture = textures[texture_ref.texture];
	if (texture.n_levels > 0)
	{
		const uint32 x = cugar::min(uint32(st.x * texture.res_x), texture.res_x - 1);
		const uint32 y = cugar::min(uint32(st.y * texture.res_y), texture.res_y - 1);
		return texture(x, y, 0); // access LOD 0
	}
	else
		return default_value;
}


FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector4f bilinear_texture_lookup_unscaled(cugar::Vector4f st, const TextureReference texture_ref, const MipMapView* textures, const float4 default_value)
{
	if (texture_ref.texture == uint32(-1) || textures[texture_ref.texture].n_levels == 0)
		return default_value;

	const MipMapView texture = textures[texture_ref.texture];
	if (texture.n_levels > 0)
	{
		const uint32 x = uint32(st.x) % texture.res_x;
		const uint32 y = uint32(st.y) % texture.res_y;

		const uint32 xx = (x + 1) % texture.res_x;
		const uint32 yy = (y + 1) % texture.res_y;

		const float u = cugar::mod(st.x, 1.0f);
		const float v = cugar::mod(st.y, 1.0f);

		cugar::Vector4f c00 = texture(x,  y,  0);
		cugar::Vector4f c10 = texture(xx, y,  0);
		cugar::Vector4f c01 = texture(x,  yy, 0);
		cugar::Vector4f c11 = texture(xx, yy, 0);

		return
			(c00 * (1-u) + c10 * u) * (1-v) +
			(c01 * (1-u) + c11 * u) * v;
	}
	else
		return default_value;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector4f bilinear_texture_lookup(cugar::Vector4f st, const TextureReference texture_ref, const MipMapView* textures, const float4 default_value)
{
	if (texture_ref.texture == uint32(-1) || textures[texture_ref.texture].n_levels == 0)
		return default_value;

	st.x *= texture_ref.scaling.x;
	st.y *= texture_ref.scaling.y;

	st.x = cugar::mod(st.x, 1.0f);
	st.y = cugar::mod(st.y, 1.0f);

	const MipMapView texture = textures[texture_ref.texture];
	if (texture.n_levels > 0)
	{
		const uint32 x = cugar::min(uint32(st.x * texture.res_x), texture.res_x - 1);
		const uint32 y = cugar::min(uint32(st.y * texture.res_y), texture.res_y - 1);

		const uint32 xx = (x + 1) % texture.res_x;
		const uint32 yy = (y + 1) % texture.res_y;

		const float u = cugar::mod(st.x, 1.0f);
		const float v = cugar::mod(st.y, 1.0f);

		cugar::Vector4f c00 = texture(x,  y,  0);
		cugar::Vector4f c10 = texture(xx, y,  0);
		cugar::Vector4f c01 = texture(x,  yy, 0);
		cugar::Vector4f c11 = texture(xx, yy, 0);

		return
			(c00 * (1-u) + c10 * u) * (1-v) +
			(c01 * (1-u) + c11 * u) * v;
	}
	else
		return default_value;
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
cugar::Vector2f diff_texture_lookup(cugar::Vector4f st, const TextureReference texture_ref, const MipMapView* textures, const float2 default_value)
{
	if (texture_ref.texture == uint32(-1) || textures[texture_ref.texture].n_levels == 0)
		return default_value;

	st.x *= texture_ref.scaling.x;
	st.y *= texture_ref.scaling.y;

	st.x = cugar::mod(st.x, 1.0f);
	st.y = cugar::mod(st.y, 1.0f);

	const MipMapView texture = textures[texture_ref.texture];
	if (texture.n_levels > 0)
	{
		// unnormalize the coordinates
		st *= cugar::Vector4f(float(texture.res_x), float(texture.res_y), 1, 1);

		cugar::Vector4f c  = bilinear_texture_lookup_unscaled( st, texture_ref, textures, cugar::Vector4f(0.0f) );
		cugar::Vector4f cu = bilinear_texture_lookup_unscaled( st + cugar::Vector4f(1,0,0,0), texture_ref, textures, cugar::Vector4f(0.0f) );
		cugar::Vector4f cv = bilinear_texture_lookup_unscaled( st + cugar::Vector4f(0,1,0,0), texture_ref, textures, cugar::Vector4f(0.0f) );

		const float i  = (c.x + c.y + c.z);
		const float iu = (cu.x + cu.y + cu.z);
		const float iv = (cv.x + cv.y + cv.z);
		return cugar::Vector2f(iu - i, iv - i);
	}
	else
		return default_value;
}

///@} TexturesModule
///@} Fermat
