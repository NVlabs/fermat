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

#include <cugar/basic/types.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>
#include <vector>
#include <memory>
#include <string>

///@addtogroup Fermat
///@{

///@addtogroup FramebufferModule
///@{

/// G-buffer view object, to be used within CUDA kernels
///
struct GBufferView
{
	FERMAT_HOST_DEVICE uint32&       tri(const uint32_t pixel)		 { return m_tri[pixel]; }
	FERMAT_HOST_DEVICE const uint32& tri(const uint32_t pixel) const { return m_tri[pixel]; }
	FERMAT_HOST_DEVICE float4&       geo(const uint32_t pixel)		 { return m_geo[pixel]; }
	FERMAT_HOST_DEVICE const float4& geo(const uint32_t pixel) const { return m_geo[pixel]; }
	FERMAT_HOST_DEVICE float4&       uv(const uint32_t pixel)		 { return m_uv[pixel]; }
	FERMAT_HOST_DEVICE const float4& uv(const uint32_t pixel) const  { return m_uv[pixel]; }
	FERMAT_HOST_DEVICE uint4&		 material(const uint32_t pixel)			{ return m_material[pixel]; }
	FERMAT_HOST_DEVICE const uint4&	 material(const uint32_t pixel) const	{ return m_material[pixel]; }
	FERMAT_HOST_DEVICE float&        depth(const uint32_t pixel)					{ return m_depth[pixel]; }
	FERMAT_HOST_DEVICE const float&  depth(const uint32_t pixel) const				{ return m_depth[pixel]; }
	FERMAT_HOST_DEVICE float4&       geo(const uint32_t x, const uint32_t y)		{ return m_geo[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float4& geo(const uint32_t x, const uint32_t y) const	{ return m_geo[y*res_x + x]; }
	FERMAT_HOST_DEVICE float4&       uv(const uint32_t x, const uint32_t y)			{ return m_uv[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float4& uv(const uint32_t x, const uint32_t y) const	{ return m_uv[y*res_x + x]; }
	FERMAT_HOST_DEVICE float&        depth(const uint32_t x, const uint32_t y)			{ return m_depth[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float&  depth(const uint32_t x, const uint32_t y) const	{ return m_depth[y*res_x + x]; }
	FERMAT_HOST_DEVICE uint4&		 material(const uint32_t x, const uint32_t y)		{ return m_material[y*res_x + x]; }
	FERMAT_HOST_DEVICE const uint4&	 material(const uint32_t x, const uint32_t y) const	{ return m_material[y*res_x + x]; }

	FERMAT_HOST_DEVICE float4&       geo(const int2 pixel)			{ return m_geo[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& geo(const int2 pixel) const	{ return m_geo[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float4&       geo(const uint2 pixel)			{ return m_geo[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& geo(const uint2 pixel) const	{ return m_geo[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float4&       uv(const int2 pixel)			{ return m_uv[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& uv(const int2 pixel) const		{ return m_uv[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float4&       uv(const uint2 pixel)			{ return m_uv[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& uv(const uint2 pixel) const	{ return m_uv[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float&		 depth(const uint2 pixel) 		{ return m_depth[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float		 depth(const uint2 pixel) const	{ return m_depth[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE uint4&		 material(const int2 pixel)			{ return m_material[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const uint4&	 material(const int2 pixel) const	{ return m_material[pixel.y * res_x + pixel.x]; }

	FERMAT_HOST_DEVICE
	static float4 pack_geometry(const cugar::Vector3f P, cugar::Vector3f N, bool miss = false)
	{
		const uint32_t n_i = cugar::pack_vector(cugar::uniform_sphere_to_square(N), 15u);
		//const uint32_t n_i = cugar::pack_vector(cugar::sphere_to_oct(N)*0.5f + cugar::Vector2f(0.5f), 15u);

		return make_float4(P.x, P.y, P.z, cugar::binary_cast<float>((uint32_t(miss) << 31) | n_i));
	}

	FERMAT_HOST_DEVICE
	static bool is_miss(const float4 geom)
	{
		return (cugar::binary_cast<uint32>(geom.w) & (1u << 31)) ? true : false;
	}

	FERMAT_HOST_DEVICE
	static cugar::Vector3f unpack_pos(const float4 geom)
	{
		return cugar::Vector3f(geom.x, geom.y, geom.z);
	}

	FERMAT_HOST_DEVICE
	static cugar::Vector3f unpack_normal(const float4 geom)
	{
		const uint32_t n_i = cugar::binary_cast<uint32_t>(geom.w) & (~(1u << 31));

		return cugar::uniform_square_to_sphere(cugar::unpack_vector<float>(n_i, 15u));
		//return cugar::oct_to_sphere(cugar::unpack_vector<float>(n_i, 15u)*2.0f - cugar::Vector2f(1.0f));
	}

	float4*	m_geo;			// gbuffer geometry
	float4* m_uv;			// gbuffer texture coordinates
	uint32* m_tri;			// gbuffer tri ids
	float*  m_depth;		// depth buffer
	uint4*	m_material;		// packed material

	uint32_t res_x;
	uint32_t res_y;
};

/// Framebuffer channel view object, to be used within CUDA kernels
///
struct FBufferChannelView
{
	FERMAT_HOST_DEVICE float4&       operator()(const uint32_t pixel)						{ return c_ptr[pixel]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32_t pixel) const					{ return c_ptr[pixel]; }
	FERMAT_HOST_DEVICE float4&       operator()(const uint32_t x, const uint32_t y)			{ return c_ptr[y*res_x + x]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint32_t x, const uint32_t y) const	{ return c_ptr[y*res_x + x]; }
	FERMAT_HOST_DEVICE float4&       operator()(const uint2 pixel)							{ return c_ptr[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& operator()(const uint2 pixel) const					{ return c_ptr[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE float4&       operator()(const int2 pixel)							{ return c_ptr[pixel.y * res_x + pixel.x]; }
	FERMAT_HOST_DEVICE const float4& operator()(const int2 pixel) const						{ return c_ptr[pixel.y * res_x + pixel.x]; }

	FERMAT_HOST_DEVICE const float4* ptr() const { return c_ptr;  }
	FERMAT_HOST_DEVICE       float4* ptr()       { return c_ptr; }

	float4* c_ptr;				// fbuffer color

	uint32_t res_x;
	uint32_t res_y;
};

/// G-buffer storage class, to be used from the host to allocate g-buffer storage
///
struct GBufferStorage
{
	uint32_t res_x;
	uint32_t res_y;

	DomainBuffer<CUDA_BUFFER, float4>	geo;			// gbuffer geometry
	DomainBuffer<CUDA_BUFFER, float4>	uv;				// gbuffer texture coordinates + group ids
	DomainBuffer<CUDA_BUFFER, uint32>	tri;			// gbuffer triangle ids
	DomainBuffer<CUDA_BUFFER, float>	depth;			// gbuffer depth
	DomainBuffer<CUDA_BUFFER, uint4>	material;		// gbuffer material

	/// set the resolution
	///
	void resize(const uint32_t _res_x, const uint32_t _res_y)
	{
		res_x = _res_x;
		res_y = _res_y;

		geo.alloc(res_x * res_y);
		uv.alloc(res_x * res_y);
		tri.alloc(res_x * res_y);
		depth.alloc(res_x * res_y);
		material.alloc(res_x * res_y);
	}

	/// return the total number of pixels
	///
	size_t size() const { return res_x * res_y; }

	/// clear the G-buffer with invalid bits
	///
	void clear()
	{
		cudaMemset(geo.ptr(), 0xFF, geo.sizeInBytes());
		cudaMemset(uv.ptr(),  0xFF, uv.sizeInBytes());
		cudaMemset(tri.ptr(), 0xFF, tri.sizeInBytes());
		cudaMemset(depth.ptr(), 0xFF, depth.sizeInBytes());
		cudaMemset(material.ptr(), 0xFF, depth.sizeInBytes());
	}

	/// return a view object
	///
	GBufferView view()
	{
		GBufferView out;
		out.res_x = res_x;
		out.res_y = res_y;
		out.m_geo = geo.ptr();
		out.m_uv  = uv.ptr();
		out.m_tri = tri.ptr();
		out.m_depth = depth.ptr();
		out.m_material = material.ptr();
		return out;
	}
};

/// Framebuffer channel storage class, to be used from the host to allocate frame-buffer storage
///
struct FBufferChannelStorage
{
	uint32_t res_x;
	uint32_t res_y;

	DomainBuffer<CUDA_BUFFER, float4>	c;				// color

	// pixel indexing operators
	float4 operator()(const uint32_t pixel) const				{ return c[pixel]; }
	float4 operator()(const uint32_t x, const uint32_t y) const { return c[y*res_x + x]; }

	const float4* ptr() const { return c.ptr(); }
	      float4* ptr()		  { return c.ptr(); }

	/// set the resolution
	///
	void resize(const uint32_t _res_x, const uint32_t _res_y)
	{
		res_x = _res_x;
		res_y = _res_y;

		c.alloc(res_x * res_y);
	}

	/// return the total number of pixels
	///
	size_t size() const { return res_x * res_y; }

	/// clear the channel with zeros
	///
	void clear()
	{
		cudaMemset(c.ptr(), 0, c.sizeInBytes());
	}

	/// copy the channel from another source
	///
	FBufferChannelStorage& operator=(FBufferChannelStorage& other)
	{
		res_x = other.res_x;
		res_y = other.res_y;
		c     = other.c;
		return *this;
	}

	/// return a view object
	///
	FBufferChannelView view()
	{
		FBufferChannelView out;
		out.res_x = res_x;
		out.res_y = res_y;
		out.c_ptr = c.ptr();
		return out;
	}

	/// swap with another framebuffer channel
	///
	void swap(FBufferChannelStorage& other)
	{
		std::swap(res_x, other.res_x);
		std::swap(res_y, other.res_y);

		c.swap(other.c);
	}
};

/// Framebuffer view object, to be used within CUDA kernels
///
struct FBufferView
{
	GBufferView			gbuffer;
	FBufferChannelView* channels;
	uint32				n_channels;

	FERMAT_HOST_DEVICE		 FBufferChannelView&	operator() (const uint32_t channel)			{ return channels[channel]; }
	FERMAT_HOST_DEVICE const FBufferChannelView&	operator() (const uint32_t channel) const	{ return channels[channel]; }

	FERMAT_HOST_DEVICE float4&       operator() (const uint32_t channel, const uint32_t pixel)						{ return channels[channel](pixel); }
	FERMAT_HOST_DEVICE const float4& operator() (const uint32_t channel, const uint32_t pixel) const				{ return channels[channel](pixel); }
	FERMAT_HOST_DEVICE float4&       operator() (const uint32_t channel, const uint32_t x, const uint32_t y)		{ return channels[channel](x, y); }
	FERMAT_HOST_DEVICE const float4& operator() (const uint32_t channel, const uint32_t x, const uint32_t y) const	{ return channels[channel](x, y); }
};

/// Framebuffer storage class, to be used from the host to allocate framebuffer storage
///\n
/// A framebuffer can hold multiple named channels of the same size; currently, only float4 channels
/// are supported - in the future multiple formats might be added.
/// The framebuffer also holds a single G-buffer.
///
struct FBufferStorage
{
	typedef std::shared_ptr<FBufferChannelStorage>	FBufferChannelPtr;

	uint32_t res_x;
	uint32_t res_y;
	uint32_t n_channels;

	GBufferStorage					gbuffer;
	FBufferChannelStorage*			channels;
	std::vector<std::string>		names;

	DomainBuffer<CUDA_BUFFER, FBufferChannelView> channel_views;

	/// constructor
	///
	FBufferStorage() : res_x(0), res_y(0), n_channels(0), channels(NULL) {}

	/// destructor
	///
	~FBufferStorage() { delete[] channels; }

	/// set the number of channels
	///
	void set_channel_count(const uint32 _n_channels)
	{
		n_channels = _n_channels;
		channels = new FBufferChannelStorage[n_channels];
		names.resize(n_channels);
	}

	/// set the name of a channel
	///
	void set_channel(const uint32_t i, const char* name)
	{
		names[i] = std::string(name);
	}

	/// set the resolution of the framebuffer
	///
	void resize(const uint32_t _res_x, const uint32_t _res_y)
	{
		res_x = _res_x;
		res_y = _res_y;

		gbuffer.resize(res_x, res_y);
		for (size_t c = 0; c < n_channels; ++c)
			channels[c].resize(res_x, res_y);

		DomainBuffer<HOST_BUFFER, FBufferChannelView> _channel_views(n_channels);
		for (size_t c = 0; c < n_channels; ++c)
			_channel_views.ptr()[c] = channels[c].view();

		channel_views = _channel_views;
	}

	/// return the number of channels
	///
	uint32_t channel_count() const
	{
		return uint32_t(n_channels);
	}

	/// return the resolution of the framebuffer
	///
	size_t size() const { return res_x * res_y; }

	/// return a view object
	///
	FBufferView view()
	{
		FBufferView out;
		out.gbuffer  = gbuffer.view();
		out.channels = channel_views.ptr();
		out.n_channels = n_channels;
		return out;
	}
};

/// add a sample to a frame-buffer channel, optionally keeping track of variance in the alpha component
///
template <bool ALPHA_AS_VARIANCE>
FERMAT_HOST_DEVICE
void average_in(FBufferChannelView& fb, const uint32_t pixel, const cugar::Vector4f f, const float inv_n)
{
		  cugar::Vector4f mean  = fb(pixel);
	const cugar::Vector4f delta = f - mean;
	mean += delta * inv_n;

	if (ALPHA_AS_VARIANCE)
	{
		// keep track of luminosity variance Welford's algorithm
		const float lum_delta  = cugar::max_comp(delta.xyz());
		const float lum_mean   = cugar::max_comp(mean.xyz());
		const float lum_f      = cugar::max_comp(f.xyz());
		const float lum_delta2 = lum_f - lum_mean;
		mean.w = lum_delta * lum_delta2;
	}
	fb(pixel) = mean;
}

/// add a sample to a frame-buffer channel, optionally keeping track of variance in the alpha component
///
template <bool ALPHA_AS_VARIANCE>
FERMAT_HOST_DEVICE
void average_in(FBufferChannelView& fb, const uint32_t pixel, const cugar::Vector3f f, const float inv_n)
{
	cugar::Vector4f mean = fb(pixel);

	const cugar::Vector3f delta = f - mean.xyz();

	mean.x += delta.x * inv_n;
	mean.y += delta.y * inv_n;
	mean.z += delta.z * inv_n;

	if (ALPHA_AS_VARIANCE)
	{
		// keep track of luminosity variance Welford's algorithm
		const float lum_delta  = cugar::max_comp(delta);
		const float lum_mean   = cugar::max_comp(mean.xyz());
		const float lum_f      = cugar::max_comp(f);
		const float lum_delta2 = lum_f - lum_mean;
		mean.w = lum_delta * lum_delta2;
	}
	fb(pixel) = mean;
}

/// add a sample to a frame-buffer channel, optionally keeping track of variance in the alpha component.
/// Note that in order to compute statistical averages, the frame-buffer should be here pre-multiplied by (n-1)/n.
///
template <bool ALPHA_AS_VARIANCE>
FERMAT_HOST_DEVICE
void add_in(FBufferChannelView& fb, const uint32_t pixel, const cugar::Vector3f f, const float inv_n)
{
	cugar::Vector4f mean = fb(pixel);

	const cugar::Vector3f delta = f - mean.xyz();

	mean.x += f.x * inv_n;
	mean.y += f.y * inv_n;
	mean.z += f.z * inv_n;

	if (ALPHA_AS_VARIANCE)
	{
		// keep track of luminosity variance using Welford's algorithm
		const float lum_delta = cugar::max_comp(delta);
		mean.w += lum_delta * lum_delta * inv_n;
	}
	fb(pixel) = mean;
}

///@} FramebufferModule
///@} Fermat
