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

#include "types.h"
#include "buffers.h"
#include "texture_reference.h"

#include <cugar/basic/types.h>
#include <cugar/linalg/vector.h>
#include <vector>
#include <memory>
#include <string>

///@addtogroup Fermat
///@{

///@addtogroup TexturesModule
///@{

/// A \ref TextureStorage "Texture" view object to be used within CUDA kernels
///
struct TextureView
{
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 pixel)		  { return c[pixel]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 pixel) const { return c[pixel]; }
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 x, const uint32 y)			{ return c[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 x, const uint32 y) const	{ return c[y*res_x + x]; }
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
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 pixel, const uint32 lod) const { return levels[lod](pixel); }
	FERMAT_HOST_DEVICE float4&       operator()(const uint32 x, const uint32 y, const uint32 lod)       { return levels[lod](x,y); }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32 x, const uint32 y, const uint32 lod) const { return levels[lod](x,y); }

	TextureView* levels;
	uint32		 n_levels;
	uint32		 res_x;
	uint32		 res_y;
};

/// Texture storage
///
/// \anchor TextureStorage
///
template <BufferType TYPE>
struct TextureStorage
{
	TextureStorage() : res_x(0), res_y(0) {}

	template <BufferType UTYPE>
	TextureStorage(const TextureStorage<UTYPE>& other) : res_x(other.res_x), res_y(other.res_y), c(other.c) {}

	template <BufferType UTYPE>
	TextureStorage& operator=(const TextureStorage<UTYPE>& other)
	{
		res_x = other.res_x;
		res_y = other.res_y;
		c = other.c;
		return *this;
	}

	const float4* ptr() const { return c.ptr(); }
	      float4* ptr()		  { return c.ptr(); }

	void resize(const uint32 _res_x, const uint32 _res_y)
	{
		res_x = _res_x;
		res_y = _res_y;

		c.alloc(res_x * res_y);
	}

	size_t size() const { return res_x * res_y; }

	uint2 res() const { return make_uint2(res_x, res_y); }

	void clear()
	{
		c.clear(0);
	}

	TextureView view()
	{
		TextureView out;
		out.res_x = res_x;
		out.res_y = res_y;
		out.c	  = c.ptr();
		return out;
	}

public:
	uint32 res_x;
	uint32 res_y;

	DomainBuffer<TYPE, float4>	c;				// color
};

/// Mip-Map storage
///
/// \anchor MipMapStorage
///
template <BufferType TYPE>
struct MipMapStorage
{
	typedef TextureStorage<TYPE>					TextureType;
	typedef std::shared_ptr<TextureStorage<TYPE> >	TexturePtr;

	uint32_t res_x;
	uint32_t res_y;
	uint32_t n_levels;

	std::vector<TexturePtr>				levels;
	DomainBuffer<TYPE, TextureView>		level_views;

	MipMapStorage() : res_x(0), res_y(0), n_levels(0) {}

	template <BufferType UTYPE>
	MipMapStorage(const MipMapStorage<UTYPE>& other) : res_x(0), res_y(0), n_levels(0)
	{
		this->operator=(other);
	}

	template <BufferType UTYPE>
	MipMapStorage& operator=(const MipMapStorage<UTYPE>& other)
	{
		res_x	 = other.res_x;
		res_y	 = other.res_y;
		n_levels = other.n_levels;

		levels.resize( n_levels );
		for (uint32 l = 0; l < n_levels; ++l)
		{
			levels[l] = TexturePtr(new TextureType);
			*levels[l] = *other.levels[l];
		}

		level_views.alloc(n_levels);
		for (uint32 l = 0; l < n_levels; ++l)
			level_views.set(l, levels[l]->view());

		return *this;
	}

	void set(TexturePtr texture)
	{
		res_x = texture->res().x;
		res_y = texture->res().y;

		// count the number of levels needed
		n_levels = 0;
		{
			uint32 l_res_x = res_x;
			uint32 l_res_y = res_y;
			while (l_res_x >= 1 && l_res_y >= 1)
			{
				n_levels++;
				l_res_x /= 2;
				l_res_y /= 2;
			}
		}
		levels.resize(n_levels);

		if (n_levels)
		{
			levels[0] = texture;

			generate_mips();
		}

		level_views.alloc(n_levels);
		for (uint32 l = 0; l < n_levels; ++l)
			level_views.set(l, levels[l]->view());
	}

	void resize(const uint32_t _res_x, const uint32_t _res_y)
	{
		res_x = _res_x;
		res_y = _res_y;

		// count levels
		n_levels = 0;
		{
			uint32 l_res_x = res_x;
			uint32 l_res_y = res_y;
			while (l_res_x >= 1 && l_res_y >= 1)
			{
				n_levels++;
				l_res_x /= 2;
				l_res_y /= 2;
			}
		}
		levels.resize(n_levels);

		// allocate levels
		n_levels = 0;
		{
			uint32 l_res_x = res_x;
			uint32 l_res_y = res_y;
			while (l_res_x >= 1 && l_res_y >= 1)
			{
				levels[n_levels] = TexturePtr(new TextureType);
				levels[n_levels]->resize(l_res_x, l_res_y);

				n_levels++;
				l_res_x /= 2;
				l_res_y /= 2;
			}
		}

		level_views.resize(n_levels);
		for (uint32 l = 0; l < n_levels; ++l)
			level_views.set(l, levels[l]->view());
	}

	void generate_mips()
	{
		uint32 level = 1;

		uint32 l_res_x = res_x/2;
		uint32 l_res_y = res_y/2;
		while (l_res_x >= 1 && l_res_y >= 1)
		{
			levels[level] = TexturePtr(new TextureType);
			levels[level]->resize(l_res_x, l_res_y);

			downsample(levels[level - 1], levels[level]);

			level++;
			l_res_x /= 2;
			l_res_y /= 2;
		}
	}

	static void downsample(TexturePtr src, TexturePtr dst)
	{
		TextureView src_view = src->view();
		TextureView dst_view = dst->view();

		for (uint32 y = 0; y < dst_view.res_y; ++y)
		{
			for (uint32 x = 0; x < dst_view.res_x; ++x)
			{
				cugar::Vector4f t(0.0f);
				for (uint32 j = 0; j < 2; ++j)
					for (uint32 i = 0; i < 2; ++i)
						t += cugar::Vector4f( src_view(x * 2 + i, y * 2 + j) );

				dst_view(x, y) = t / 4.0f;
			}
		}
	}

	uint32 level_count() const
	{
		return uint32(n_levels);
	}

	MipMapView view()
	{
		MipMapView r;
		r.n_levels = n_levels;
		r.res_x = res_x;
		r.res_y = res_y;
		r.levels = level_views.ptr();
		return r;
	}
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

///@} TexturesModule
///@} Fermat
