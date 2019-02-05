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

///@addtogroup Fermat
///@{

struct RTContext;
struct RenderingContextImpl;

/// A class encpasulating the entire rendering context
///
struct FERMAT_API RenderingContext
{
	typedef std::shared_ptr< MipMapStorage<HOST_BUFFER> >	HostMipMapStoragePtr;
	typedef std::shared_ptr< MipMapStorage<CUDA_BUFFER> >	DeviceMipMapStoragePtr;

	/// constructor
	///
	RenderingContext();

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
	uint2 res() const;

	/// return a view of the renderer
	///
	RenderingContextView view(const uint32 instance);

	/// return the camera
	///
	Camera& get_camera();

	/// return the directional light count
	///
	uint32 get_directional_light_count() const;

	/// return the host-side directional lights
	///
	const DirectionalLight* get_host_directional_lights() const;

	/// return the device-side directional lights
	///
	const DirectionalLight* get_device_directional_lights() const;

	/// set the number of directional lights
	///
	void set_directional_light_count(const uint32 count);

	/// set a directional light
	///
	void set_directional_light(const uint32 i, const DirectionalLight& light);

	/// return the target resolution
	///
	uint2 get_res() const;

	/// return the target aspect ratio
	///
	float get_aspect_ratio() const;

	/// set the target exposure
	///
	void set_aspect_ratio(const float v);

	/// return the target exposure
	///
	float get_exposure() const;

	/// set the target exposure
	///
	void set_exposure(const float v);

	/// return the target gamma
	///
	float get_gamma() const;

	/// set the target gamma
	///
	void set_gamma(const float v);

	/// return the shading mode
	///
	ShadingMode& get_shading_mode();

	/// return the frame buffer
	///
	FBufferStorage& get_frame_buffer();

	/// return the frame buffer
	///
	uint8* get_device_rgba_buffer();

	/// return the number of textures
	///
	uint32 get_texture_count() const;

	/// return the scene's host-side textures
	///
	HostMipMapStoragePtr* get_host_textures();

	/// return the scene's device-side textures
	///
	DeviceMipMapStoragePtr* get_device_textures();

	/// return the scene's host-side textures
	///
	MipMapView* get_host_texture_views();

	/// return the scene's device-side textures
	///
	MipMapView* get_device_texture_views();

	/// return the scene's host-side mesh
	///
	MeshStorage& get_host_mesh();

	/// return the scene's device-side mesh
	///
	DeviceMeshStorage& get_device_mesh();

	/// return the scene's device-side mesh emitters
	///
	MeshLightsStorage& get_mesh_lights();

	/// return the ray tracing context
	///
	RTContext* get_rt_context() const;

	/// return the renderer
	///
	RendererInterface* get_renderer() const;

	/// return the sampling sequence
	///
	TiledSequence& get_sequence();

	/// register a new rendering interface type
	///
	uint32 register_renderer(const char* name, RendererFactoryFunction factory);

	/// compute the scene's bbox
	///
	cugar::Bbox3f compute_bbox();

private:
	RenderingContextImpl* m_impl;
};

///@} Fermat
