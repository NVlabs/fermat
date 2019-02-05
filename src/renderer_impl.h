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

#include <renderer.h>
#include <buffers.h>
#include <framebuffer.h>
#include <texture.h>
#include <camera.h>
#include <ray.h>
#include <lights.h>
#include <mesh_lights.h>
#include <mesh/MeshStorage.h>
#include <vector>
#include <renderer_view.h>
#include <renderer_interface.h>
#include <tiled_sequence.h>
#include <dll.h>

///@addtogroup Fermat
///@{

struct RenderingContext;

/// A class encpasulating the entire rendering context
///
struct RenderingContextImpl
{
	typedef RenderingContext::HostMipMapStoragePtr		HostMipMapStoragePtr;
	typedef RenderingContext::DeviceMipMapStoragePtr	DeviceMipMapStoragePtr;
	typedef DomainBuffer<HOST_BUFFER, DirectionalLight>	HostDirectionalLightVector;
	typedef DomainBuffer<CUDA_BUFFER, DirectionalLight>	DeviceDirectionalLightVector;

	/// constructor
	///
	RenderingContextImpl(RenderingContext* _context) : m_this( _context ) {}

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

	/// clamp the output framebuffer to a given maximum
	///
	/// \param max_value
	void clamp_frame(const float max_value);

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
	RenderingContextView view(const uint32 instance);

	/// return the camera
	///
	Camera& get_camera() { return m_camera; }

	/// return the target resolution
	///
	uint2 get_res() const { return make_uint2(m_res_x, m_res_y); }

	/// return the target aspect ratio
	///
	float get_aspect_ratio() const { return m_aspect; }

	/// return the target exposure
	///
	float get_exposure() const { return m_exposure; }

	/// return the target gamma
	///
	float get_gamma() const { return m_gamma; }

	/// return the frame buffer
	///
	FBufferStorage& get_frame_buffer() { return m_fb; }

	/// return the scene's host-side textures
	///
	std::vector<HostMipMapStoragePtr>& get_host_textures() { return m_textures_h; }

	/// return the scene's device-side textures
	///
	std::vector<DeviceMipMapStoragePtr>& get_device_textures() { return m_textures_d; }

	/// return the scene's host-side textures
	///
	MipMapView* get_host_texture_views() { return m_texture_views_h.ptr(); }

	/// return the scene's device-side textures
	///
	MipMapView* get_device_texture_views() { return m_texture_views_d.ptr(); }

	/// return the scene's host-side mesh
	///
	MeshStorage& get_host_mesh() { return m_mesh; }

	/// return the scene's device-side mesh
	///
	DeviceMeshStorage& get_device_mesh() { return m_mesh_d; }

	/// return the scene's device-side mesh emitters
	///
	MeshLightsStorage& get_mesh_lights() { return m_mesh_lights; }

	/// return the ray tracing context
	///
	RTContext* get_rt_context() const { return m_rt_context; }

	/// return the renderer
	///
	RendererInterface* get_renderer() const { return m_renderer; }

	/// register a new rendering interface type
	///
	uint32 register_renderer(const char* name, RendererFactoryFunction factory);

	/// load a plugin
	///
	uint32 load_plugin(const char* plugin_name);

	/// compute the scene's bbox
	///
	cugar::Bbox3f compute_bbox();

	RenderingContext*		m_this;

	unsigned int			m_res_x;							///< X resolution
	unsigned int			m_res_y;							///< Y resolution
	float					m_aspect;							///< aspect ratio
	float					m_exposure;							///< exposure
	float					m_gamma;							///< gamma
	float                   m_shading_rate;						///< shading rate
    ShadingMode             m_shading_mode;						///< shading mode

	MeshStorage				m_mesh;								///< host-side scene mesh representation
	DeviceMeshStorage		m_mesh_d;							///< device-side scene mesh representation
	MeshLightsStorage		m_mesh_lights;						///< mesh lights
	RTContext*				m_rt_context;						///< internal optix ray tracing context
    //DeviceUVBvh             m_uv_bvh;

	Camera									m_camera;			///< camera
	DirectionalLight						m_light;
	HostDirectionalLightVector				m_dir_lights_h;		///< host-side directional lights
	DeviceDirectionalLightVector			m_dir_lights_d;		///< device-side directional lights

	uint32									m_renderer_type;	///< rendering engine type
	RendererInterface*						m_renderer;			///< rendering engine

	std::vector<HostMipMapStoragePtr>		m_textures_h;		///< host-side textures
	std::vector<DeviceMipMapStoragePtr>		m_textures_d;		///< device-side textures
	DomainBuffer<HOST_BUFFER, MipMapView>	m_texture_views_h;	///< host-side texture views
	DomainBuffer<CUDA_BUFFER, MipMapView>	m_texture_views_d;	///< device-side texture views
	DomainBuffer<CUDA_BUFFER, float4>		m_ltc_M;			///< LTC coefficients
	DomainBuffer<CUDA_BUFFER, float4>		m_ltc_Minv;			///< LTC coefficients
	DomainBuffer<CUDA_BUFFER, float>		m_ltc_A;			///< LTC coefficients
	uint32									m_ltc_size;			///< LTC coefficients
	DomainBuffer<CUDA_BUFFER, float>		m_glossy_reflectance;	///< glossy reflectance/albedo profile

	FBufferStorage							m_fb;				///< output framebuffer storage
	FBufferChannelStorage					m_fb_temp[4];		///< temporary framebuffer storage
	DomainBuffer<CUDA_BUFFER, float>		m_var;				///< variance framebuffer storage
	DomainBuffer<CUDA_BUFFER, uint8>		m_rgba;				///< output 8-bit rgba storage

	TiledSequence							m_sequence;			///< a tiled sequence

	std::vector<std::string>				m_renderer_names;		///< plugin renderer names
	std::vector<RendererFactoryFunction>	m_renderer_factories;	///< plugin renderer factories
	std::vector<DLL>						m_plugins;				///< plugin DLLs
};

///@} Fermat
