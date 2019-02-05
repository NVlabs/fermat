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

#include <vertex_storage.h>
#include <cugar/linalg/vector.h>
#include <camera.h>
#include <ray_queues.h>
#include <bpt_queues.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

//! [BPTContextBaseBlock]
///\par
/// Basic context for bidirectional path tracing
///\par
/// This context class is responsible for storing:
///\n
/// - a set of light vertices, generated throughout the light subpath sampling phase
/// - a set of ray queues used throughout the wavefront scheduling process of the entire bidirectional path tracing pipeline
/// - an options member, deriving from \ref BPTOptionsBase
///
template <typename TBPTOptions>
struct BPTContextBase
{
	BPTContextBase() :
		in_bounce(0) {}

	BPTContextBase(
		const RenderingContextView&	_renderer,
		const VertexStorageView&	_light_vertices,
		const BPTQueuesView&		_queues,
		const TBPTOptions			_options = TBPTOptions()) :
		in_bounce(0),
		light_vertices(_light_vertices),
		in_queue(_queues.in_queue),
		shadow_queue(_queues.shadow_queue),
		scatter_queue(_queues.scatter_queue),
		options(_options)
	{
		set_camera(_renderer.camera, _renderer.res_x, _renderer.res_y, _renderer.aspect);
	}

	// precompute some camera-related quantities
	void set_camera(const Camera& camera, const uint32 res_x, const uint32 res_y, const float aspect_ratio)
	{
		camera_frame(camera, aspect_ratio, camera_U, camera_V, camera_W);

		camera_W_len = cugar::length(camera_W);

		//camera_square_focal_length = camera.square_pixel_focal_length(res_x, res_y);
		camera_square_focal_length = camera.square_screen_focal_length();
	}

	uint32				in_bounce;
	RayQueue			in_queue;
	RayQueue			shadow_queue;
	RayQueue			scatter_queue;

	VertexStorageView	light_vertices;

	cugar::Vector3f		camera_U;
	cugar::Vector3f		camera_V;
	cugar::Vector3f		camera_W;
	float				camera_W_len;
	float				camera_square_focal_length;

	TBPTOptions			options;
};
//! [BPTContextBaseBlock]

///@} BPTLib
///@} Fermat
