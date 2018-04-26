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

#include <buffers.h>
#include <framebuffer.h>
#include <texture.h>
#include <camera.h>
#include <ray.h>
#include <lights.h>
#include <mesh_lights.h>
#include <mesh/MeshStorage.h>
#include <optix_prime/optix_primepp.h>
#include <vector>
#include <renderer_view.h>
#include <renderer_interface.h>

#define SHADOW_BIAS		0.0f
#define SHADOW_TMIN		1.0e-4f

///@addtogroup Fermat
///@{

struct PathTracer;
struct RPT;
struct BPT;
struct MLT;
struct CMLT;
struct PSSMLT;
struct PSFPT;

/// A class encpasulating the entire rendering context
///
struct Renderer
{
	typedef std::shared_ptr< MipMapStorage<HOST_BUFFER> >	HostMipMapStoragePtr;
	typedef std::shared_ptr< MipMapStorage<CUDA_BUFFER> >	DeviceMipMapStoragePtr;

	/// initialize the renderer
	///
	void init(int argc, char** argv);

	/// render a frame
	///
	/// \param instance		the sequence instance / frame number in a progressive render
	void render(const uint32 instance);

	/// clear all framebuffers
	///
	void clear();

	/// rescale the output framebuffer by a constant
	///
	void multiply_frame(const float scale);

	/// rescale the output framebuffer by n/(n-1)
	///
	/// \param instance		the sequence instance / frame number in a progressive render, used for rescaling
	void rescale_frame(const uint32 instance);

	/// update the variance estimates
	///
	/// \param instance		the sequence instance / frame number in a progressive render, used for rescaling
	void update_variances(const uint32 instance);

	/// update the internal data-structures (e.g. BVHs) associated to the geometry
	///
	void update_model();

	/// perform filtering
	///
	/// \param instance		the sequence instance / frame number in a progressive render
	void filter(const uint32 instance);

	/// return the current output resolution
	///
	uint2 res() const { return make_uint2(m_res_x, m_res_y); }

	/// return a view of the renderer
	///
	RendererView view(const uint32 instance);

	unsigned int			m_res_x;							///< X resolution
	unsigned int			m_res_y;							///< Y resolution
	float					m_aspect;							///< aspect ratio
	float					m_exposure;							///< exposure
	float                   m_shading_rate;						///< shading rate
    ShadingMode             m_shading_mode;						///< shading mode

	optix::prime::Context	m_context;							///< internal ray tracing context
	MeshStorage				m_mesh;								///< host-side scene mesh representation
	DeviceMeshStorage		m_mesh_d;							///< device-side scene mesh representation
	MeshLightsStorage		m_mesh_lights;						///< mesh lights
	optix::prime::Model     m_model;							///< internal ray tracing model
    //DeviceUVBvh             m_uv_bvh;

	Camera					m_camera;							///< camera
	DiskLight				m_light;

	RendererType			m_renderer_type;					///< rendering engine type
	RendererInterface*		m_renderer;							///< rendering engine

	std::vector<HostMipMapStoragePtr>		m_textures_h;		///< host-side textures
	std::vector<DeviceMipMapStoragePtr>		m_textures_d;		///< device-side textures
	DomainBuffer<HOST_BUFFER, MipMapView>	m_texture_views_h;	///< host-side texture views
	DomainBuffer<CUDA_BUFFER, MipMapView>	m_texture_views_d;	///< device-side texture views
	DomainBuffer<CUDA_BUFFER, float4>		m_ltc_M;			///< LTC coefficients
	DomainBuffer<CUDA_BUFFER, float4>		m_ltc_Minv;			///< LTC coefficients
	DomainBuffer<CUDA_BUFFER, float>		m_ltc_A;			///< LTC coefficients
	uint32									m_ltc_size;			///< LTC coefficients

	FBufferStorage							m_fb;				///< output framebuffer storage
	FBufferChannelStorage					m_fb_temp[2];		///< temporary framebuffer storage
	DomainBuffer<CUDA_BUFFER, float>		m_var;				///< variance framebuffer storage
	DomainBuffer<CUDA_BUFFER, uint8>		m_rgba;				///< output 8-bit rgba storage
};

///@} Fermat
