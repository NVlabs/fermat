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

#include <framebuffer.h>
#include <texture.h>
#include <camera.h>
#include <ray.h>
#include <lights.h>
#include <mesh_lights.h>
#include <mesh/MeshStorage.h>

///@addtogroup Fermat
///@{

#define SHADOW_BIAS		0.0f
#define SHADOW_TMIN		1.0e-4f

enum RendererType
{
	kPT				= 0,
	kBPT			= 1,
	kCMLT			= 2,
	kMLT			= 3,
	kPSSMLT			= 4,
	kRPT			= 5,
	kPSFPT			= 6,
};

enum ShadingMode {
    kShaded			= 0,
    kUV				= 1,
    kUVStretch		= 2,
    kCharts			= 3,
	kAlbedo			= 4,
	kDiffuseAlbedo  = 5,
	kSpecularAlbedo = 6,
	kDiffuseColor   = 7,
	kSpecularColor  = 8,
	kDirectLighting = 9,
	kFiltered		= 10,
	kVariance		= 11,
	kAux0			= 12,
};


struct RendererView
{
	RendererView(
	    Camera				_camera,
	    DiskLight			_light,
        MeshView			_mesh,
		MeshLight			_mesh_light,
		MeshLight			_mesh_vpls,
		const MipMapView*	_textures,
		const uint32		_ltc_size,
		const float4*		_ltc_M,
		const float4*		_ltc_Minv,
		const float*		_ltc_A,
		const uint32_t		_x,
	    const uint32_t		_y,
		const float			_aspect,
		const float			_exposure,
		const float			_shading_rate,
        const ShadingMode	_shading_mode,
		const FBufferView	_fb,
		const uint32		_instance)
	: camera(_camera), light(_light), mesh(_mesh), mesh_light(_mesh_light), mesh_vpls(_mesh_vpls), textures(_textures), ltc_M(_ltc_M), ltc_Minv(_ltc_Minv), ltc_A(_ltc_A), ltc_size(_ltc_size), res_x(_x), res_y(_y), aspect(_aspect), exposure(_exposure), shading_rate(_shading_rate), shading_mode(_shading_mode), fb(_fb), instance(_instance) {}

	Camera				camera;
	DiskLight			light;
    MeshView			mesh;
	MeshLight			mesh_light;
	MeshLight			mesh_vpls;
	const MipMapView*	textures;
	const float4*		ltc_M;
	const float4*		ltc_Minv;
	const float*		ltc_A;
	uint32				ltc_size;
	uint32_t			res_x;
	uint32_t			res_y;
	float				aspect;
	float				exposure;
	float				shading_rate;
    ShadingMode			shading_mode;
	FBufferView			fb;
	uint32				instance;
};

struct FBufferDesc
{
	static const uint32_t DIFFUSE_C		= 0;
	static const uint32_t DIFFUSE_A		= 1;
	static const uint32_t SPECULAR_C	= 2;
	static const uint32_t SPECULAR_A	= 3;
	static const uint32_t DIRECT_C		= 4;
	static const uint32_t COMPOSITED_C	= 5;
	static const uint32_t FILTERED_C	= 6;
	static const uint32_t LUMINANCE		= 7;
	static const uint32_t VARIANCE		= 7;
	static const uint32_t NUM_CHANNELS	= 8;
};

///@} Fermat
