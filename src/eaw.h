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

#include "framebuffer.h"

///@addtogroup Fermat
///@{

///@addtogroup FilteringModule
///@{

enum EAWOp
{
	kEAWOpNone				= 0x0u,

	kEAWOpModulateInput		= 0x1u,
	kEAWOpDemodulateInput	= 0x2u,

	kEAWOpModulateOutput	= 0x4u,
	kEAWOpDemodulateOutput	= 0x8u,

	kEAWOpAddMode			= 0x10u,
	kEAWOpReplaceMode		= 0x20u
};

/// Edge A-trous Wavelet filtering parameters
///
struct EAWParams
{
	float phi_normal;
	float phi_position;	// must take into account the scene size
	float phi_color;	// must take into account the maximum intensity
};

/// perform a step of Edge A-trous Wavelet filtering
///
void EAW(FBufferChannelView dst, const FBufferChannelView img, const GBufferView gb, const float* var, const EAWParams params, const uint32 step_size);

/// perform a step of Edge A-trous Wavelet filtering, multiplying the result by a weight image and adding it to the output, i.e. solving:
///
///   dst += w_img * eaw(img)
///
void EAW(FBufferChannelView dst, const EAWOp op, const FBufferChannelView w_img, const float w_min, const FBufferChannelView img, const GBufferView gb, const float* var, const EAWParams params, const uint32 step_size);

/// perform several iterations of Edge A-trous Wavelet filtering
///
void EAW(const uint32 n_iterations, uint32& in_buffer, FBufferChannelView pingpong[2], const GBufferView gb, const float* var, const EAWParams params);

/// perform several iterations of Edge A-trous Wavelet filtering
///
void EAW(const uint32 n_iterations, FBufferChannelView dst, const FBufferChannelView img, const GBufferView gb, const float* var, const EAWParams params, FBufferChannelView pingpong[2]);

/// perform several iterations of Edge A-trous Wavelet filtering
///
void EAW(const uint32 n_iterations, FBufferChannelView dst, const FBufferChannelView w_img, const FBufferChannelView img, const GBufferView gb, const float* var, const EAWParams params, FBufferChannelView pingpong[2]);

//-------------------------------------------------------------------------------
//   helper functions
//-------------------------------------------------------------------------------

FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
cugar::Vector3f demodulate(const cugar::Vector3f f, const cugar::Vector3f c)
{
	return f / cugar::max(c, 1.0e-4f);
}

FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
cugar::Vector3f modulate(const cugar::Vector3f f, const cugar::Vector3f c)
{
	return f * cugar::max(c, 1.0e-4f);
}
FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
cugar::Vector4f modulate(const cugar::Vector4f f, const cugar::Vector4f c)
{
	return f * cugar::max(c, 1.0e-4f);
}

///@} FilteringModule
///@} Fermat
