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
#include <lights.h>
#include <vertex_storage.h>
#include <bpt_options.h>
#include <bpt_queues.h>
#include <tiled_sequence.h>
#include <cugar/sampling/lfsr.h>
#include <renderer_interface.h>

struct Renderer;
struct RendererView;
struct PSSMLTContext;

///@addtogroup Fermat
///@{

///@addtogroup PSSMLTModule
///@{

/// PSSMLT renderer options
///
struct PSSMLTOptions : BPTOptionsBase
{
	uint32	n_chains;
	uint32	spp;
	bool	rr;
	bool    light_perturbations;
	bool	eye_perturbations;
	float	independent_samples;

	PSSMLTOptions() :
		BPTOptionsBase(),
		n_chains(64 * 1024),
		spp(4),
		rr(true),
		light_perturbations(true),
		eye_perturbations(true),
		independent_samples(0.0f)
	{
		BPTOptionsBase::light_tracing = 0.0f;
	}

	void parse(const int argc, char** argv)
	{
		BPTOptionsBase::parse(argc, argv);

		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-chains") == 0)
				n_chains = atoi(argv[++i]) * 1024;
			else if (strcmp(argv[i], "-spp") == 0)
				spp = atoi(argv[++i]);
			else if (strcmp(argv[i], "-rr") == 0)
				rr = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-light-perturbations") == 0)
				light_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-eye-perturbations") == 0)
				eye_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-perturbations") == 0)
				light_perturbations = eye_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-independent-samples") == 0 ||
					 strcmp(argv[i], "-independent") == 0 ||
					 strcmp(argv[i], "-is") == 0)
				independent_samples = (float)atof(argv[++i]);
		}
	}
};

/// Primary Sample Space MLT renderer
///
struct PSSMLT : RendererInterface
{
	PSSMLT();

	void init(int argc, char** argv, Renderer& renderer);

	void render(const uint32 instance, Renderer& renderer);

	void sample_seeds(const uint32 n_chains);

	void recover_primary_coordinates(PSSMLTContext& context, RendererView& renderer_view);

	void destroy() { delete this; }

	PSSMLTOptions						m_options;

	TiledSequence						m_sequence;

	BPTQueuesStorage					m_queues;

	VertexStorage						m_light_vertices;

	DomainBuffer<CUDA_BUFFER, float>			m_mut_u;
	DomainBuffer<CUDA_BUFFER, float>			m_light_u;
	DomainBuffer<CUDA_BUFFER, float>			m_eye_u;
	DomainBuffer<CUDA_BUFFER, float4>			m_path_value;
	DomainBuffer<CUDA_BUFFER, float>			m_path_pdf;
	DomainBuffer<CUDA_BUFFER, float4>			m_connections_value;
	DomainBuffer<CUDA_BUFFER, float>			m_connections_cdf;
	DomainBuffer<CUDA_BUFFER, uint32>			m_seeds;
	DomainBuffer<CUDA_BUFFER, float>			m_st_norms;
	DomainBuffer<CUDA_BUFFER, uint32>			m_rejections;
	float										m_image_brightness;
	uint32										m_n_lights;
	uint32										m_n_init_light_paths;
	uint32										m_n_init_paths;
	uint32										m_n_connections;
	float										m_time;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;
};

///@} PSSMLTModule
///@} Fermat
