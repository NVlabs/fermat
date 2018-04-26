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
#include <buffers.h>
#include <ray.h>
#include <tiled_sequence.h>
#include <cugar/sampling/lfsr.h>
#include <renderer_interface.h>

struct Renderer;


///@addtogroup Fermat
///@{

///@addtogroup PTModule
///@{

/// Path tracer renderer options
///
struct PTOptions
{
	uint32	max_path_length;
	bool	direct_lighting_nee;
	bool	direct_lighting_bsdf;
	bool	indirect_lighting_nee;
	bool	indirect_lighting_bsdf;
	bool	visible_lights;
	bool	rr;

	PTOptions() :
		max_path_length(6),
		direct_lighting_nee(true),
		direct_lighting_bsdf(true),
		indirect_lighting_nee(true),
		indirect_lighting_bsdf(true),
		visible_lights(true),
		rr(true) {}

	void parse(const int argc, char** argv)
	{
		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-pl") == 0 ||
				strcmp(argv[i], "-path-length") == 0 ||
				strcmp(argv[i], "-max-path-length") == 0)
				max_path_length = atoi(argv[++i]);
			else if (strcmp(argv[i], "-bounces") == 0)
				max_path_length = atoi(argv[++i]) + 1;
			else if (strcmp(argv[i], "-nee") == 0)
				direct_lighting_nee = indirect_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-nee") == 0)
				direct_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-bsdf") == 0)
				direct_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-nee") == 0)
				indirect_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-bsdf") == 0)
				indirect_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-bsdf-lighting") == 0)
				direct_lighting_bsdf = indirect_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-visible-lights") == 0)
				visible_lights = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-rr") == 0)
				rr = atoi(argv[++i]) > 0;
		}
	}
};

/// Path tracer
///
struct PathTracer : RendererInterface
{
	PathTracer();

	void init(int argc, char** argv, Renderer& renderer);

	void render(const uint32 instance, Renderer& renderer);

	void setup_samples(const uint32 instance);

	void destroy() { delete this; }

	DomainBuffer<CUDA_BUFFER, uint8>	m_memory_pool;

	PTOptions					m_options;
	TiledSequence				m_sequence;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;

	float						m_time;
};

///@} PTModule
///@} Fermat
