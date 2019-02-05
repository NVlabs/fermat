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

#include <types.h>
#include <buffers.h>
#include <ray.h>
#include <lights.h>
#include <vertex_storage.h>
#include <bpt_options.h>
#include <bpt_queues.h>
#include <tiled_sequence.h>
#include <cugar/sampling/lfsr.h>
#include <renderer_interface.h>

struct RenderingContext;
struct RenderingContextView;
struct BPTContext;

///@addtogroup Fermat
///@{

///@defgroup BPTModule
/// This module provides an implementation of a bidirectional path tracer built on top of the \ref BPTLibPage library.
///@{

/// Options for the BPT renderer
///
struct BPTOptions : BPTOptionsBase
{
	bool single_connection;
	bool rr;

	BPTOptions() :
		BPTOptionsBase(),
		single_connection(true),
		rr(true) {}

	void parse(const int argc, char** argv)
	{
		BPTOptionsBase::parse(argc, argv);

		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-single-connection") == 0 ||
				strcmp(argv[i], "-sc") == 0)
				single_connection = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-rr") == 0 ||
					 strcmp(argv[i], "-RR") == 0)
				rr = atoi(argv[++i]) > 0;
		}
	}
};

/// A bidirectional path tracing renderer built on top of the \ref BPTLibPage library.
///
struct BPT : RendererInterface
{
	BPT();

	void init(int argc, char** argv, RenderingContext& renderer);

	void render(const uint32 instance, RenderingContext& renderer);

	void destroy() { delete this; }

	static RendererInterface* factory() { return new BPT(); }

	void regenerate_primary_light_vertices(const uint32 instance, RenderingContext& renderer);

	BPTOptions					m_options;

	TiledSequence				m_sequence;

	BPTQueuesStorage			m_queues;

	VertexStorage				m_light_vertices;
	uint32						m_n_light_subpaths;
	uint32						m_n_eye_subpaths;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;

	float						m_time;
};

///@} BPT
///@} Fermat
