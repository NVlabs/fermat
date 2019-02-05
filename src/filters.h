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

#include "framebuffer.h"

///@addtogroup Fermat
///@{

///@addtogroup FilteringModule
///@{

enum FilterOp
{
	kFilterOpNone				= 0x0u,

	kFilterOpModulateInput		= 0x1u,
	kFilterOpDemodulateInput	= 0x2u,

	kFilterOpModulateOutput		= 0x4u,
	kFilterOpDemodulateOutput	= 0x8u,

	kFilterOpAddMode			= 0x10u,
	kFilterOpReplaceMode		= 0x20u
};

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
