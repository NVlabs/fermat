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
#include "buffers.h"
#include "texture_reference.h"
#include "texture_view.h"

#include <cugar/basic/types.h>
#include <cugar/linalg/vector.h>
#include <vector>
#include <memory>
#include <string>

///@addtogroup Fermat
///@{

///@addtogroup TexturesModule
///@{

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

	DomainBuffer<HOST_BUFFER, TexturePtr>			levels;
	DomainBuffer<TYPE,        TextureView>			level_views;

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

		levels.alloc( n_levels );
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
		levels.alloc(n_levels);

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
		levels.alloc(n_levels);

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

FERMAT_API_EXTERN template class FERMAT_API DomainBuffer<HOST_BUFFER, MipMapStorage<HOST_BUFFER>::TexturePtr>;
FERMAT_API_EXTERN template class FERMAT_API DomainBuffer<HOST_BUFFER, MipMapStorage<CUDA_BUFFER>::TexturePtr>;
FERMAT_API_EXTERN template class FERMAT_API DomainBuffer<HOST_BUFFER, TextureView>;
FERMAT_API_EXTERN template class FERMAT_API DomainBuffer<CUDA_BUFFER, TextureView>;

FERMAT_API_EXTERN template struct FERMAT_API TextureStorage<HOST_BUFFER>;
FERMAT_API_EXTERN template struct FERMAT_API TextureStorage<CUDA_BUFFER>;

FERMAT_API_EXTERN template struct FERMAT_API MipMapStorage<HOST_BUFFER>;
FERMAT_API_EXTERN template struct FERMAT_API MipMapStorage<CUDA_BUFFER>;


///@} TexturesModule
///@} Fermat
