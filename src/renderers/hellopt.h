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

#include <renderer_interface.h>
#include <buffers.h>
#include <tiled_sequence.h>

struct RenderingContext;


///@addtogroup Fermat
///@{

///@defgroup HelloPTModule
/// This module defines a "Hello Path Tracing" renderer
///@{

/// our "Hello Path Tracing" options
///
struct HelloPTOptions
{
	uint32 max_path_length;

	/// default constructor
	///
	HelloPTOptions() : max_path_length(6) {}

	/// do some simple option parsing
	///
	void parse(const int argc, char** argv)
	{
		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-path-length") == 0)
				max_path_length = atoi(argv[++i]);
		}
	}
};

/// A "Hello Path Tracing" renderer
///
struct HelloPT : RendererInterface
{
	void init(int argc, char** argv, RenderingContext& renderer);

	void render(const uint32 instance, RenderingContext& renderer);

	void destroy() { delete this; }

	static RendererInterface* factory() { return new HelloPT(); }

	HelloPTOptions						m_options;
	DomainBuffer<CUDA_BUFFER, uint8>	m_memory_pool;
	TiledSequence						m_sequence;
};

///@} HelloPTModule
///@} Fermat
