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

#include <types.h>

struct Renderer;
struct FBufferStorage;

///@addtogroup Fermat
///@{

/// The abstract renderer / solver interface
///
struct RendererInterface
{
	/// this method is responsible for returning the number of auxiliary framebuffer channels needed by the renderer
	///
	virtual uint32 auxiliary_channel_count() { return 0; }

	/// this method is responsible for registering the auxiliary framebuffer channels needed by the renderer, starting at the specified offset
	///
	virtual void register_auxiliary_channels(FBufferStorage& fbuffer, const uint32 channel_offset) {}

	/// this method is responsible for any command options parsing / initializations the renderer might need to perform
	///
	virtual void init(int argc, char** argv, Renderer& renderer) {}

	///\anchor RendererInterfaceRenderMethod
	/// this method is responsible for rendering a given frame in a progressive rendering
	///
	/// \param	instance		the frame instance
	///
	virtual void render(const uint32 instance, Renderer& renderer) {}

	/// this method is responsible for handling keyboard events
	///
	virtual void keyboard(unsigned char character, int x, int y, bool& invalidate) {}

	/// this method is responsible for destroying the object itself
	///
	virtual void destroy() {}

	/// this method is responsible for handling mouse events
	///
	virtual void mouse(Renderer& renderer, int button, int state, int x, int y) {}

	/// this method is responsible for any additional UI/OpenGL drawing on screen
	///
	virtual void draw(Renderer& renderer) {}
};

///@} Fermat
