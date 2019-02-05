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
#include <cugar/color/rgbe.h>
#include <cugar/basic/memory_arena.h>
#include <renderer_interface.h>

struct RenderingContext;
struct RenderingContextView;

struct MLTContext;

/// MLT renderer options
///
struct MLTOptions : BPTOptionsBase
{
	uint32	n_chains;
	uint32	spp;
	uint32  reseeding_freq;
	uint32	mis					 : 1;
	float	st_perturbations;
	float	screen_perturbations;
	float	exp_perturbations;
	float	H_perturbations;
	float	perturbation_radius;
	uint32	flags;

	MLTOptions() :
		BPTOptionsBase(),
		n_chains(256 * 1024),
		spp(1),
		reseeding_freq(16),
		st_perturbations(1.0f),
		screen_perturbations(1.0f),
		exp_perturbations(0.45f),
		H_perturbations(0.45f),
		perturbation_radius(0.1f),
		mis(false),
		flags(0)
	{
		// temporarily kill light tracing
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
			else if (strcmp(argv[i], "-st-mutations") == 0 ||
					 strcmp(argv[i], "-st-perturbations") == 0)
				st_perturbations = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-screen-perturbations") == 0 ||
					 strcmp(argv[i], "-lens-perturbations") == 0 ||
					 strcmp(argv[i], "-lp") == 0)
				screen_perturbations = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-H-perturbations") == 0 ||
					 strcmp(argv[i], "-h-perturbations") == 0 ||
					 strcmp(argv[i], "-hp") == 0 ||
					 strcmp(argv[i], "-H") == 0 ||
					 strcmp(argv[i], "-h") == 0)
				H_perturbations = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-exp-perturbations") == 0 ||
					 strcmp(argv[i], "-e-perturbations") == 0 ||
					 strcmp(argv[i], "-exp") == 0 ||
					 strcmp(argv[i], "-ep") == 0)
				exp_perturbations = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-rf") == 0 ||
					 strcmp(argv[i], "-reseed") == 0 ||
					 strcmp(argv[i], "-reseed-freq") == 0)
				reseeding_freq = atoi(argv[++i]);
			else if (strcmp(argv[i], "-mis") == 0)
				mis = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-pr") == 0 ||
					 strcmp(argv[i], "-perturbation-radius") == 0)
				perturbation_radius = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-flags") == 0)
				flags = strtol(argv[++i], NULL, 16u);
		}

		const float prob_perturbations = exp_perturbations + H_perturbations;
		if (prob_perturbations > 1.0f)
		{
			exp_perturbations /= prob_perturbations;
			H_perturbations   /= prob_perturbations;
		}
	}
};

/// Plain old Veach's MLT (or a parallel version thereof...)
///
struct MLT : RendererInterface
{
	MLT();

	void init(int argc, char** argv, RenderingContext& renderer);

	void render(const uint32 instance, RenderingContext& renderer);

	void sample_seeds(MLTContext& context);

	void recover_seed_paths(MLTContext& context, RenderingContextView& renderer_view);

	void destroy() { delete this; }

	static RendererInterface* factory();

	void alloc(MLTContext& context, cugar::memory_arena& arena, const uint32 n_pixels, const uint32 n_chains);

	MLTOptions							m_options;
	TiledSequence						m_sequence;
	BPTQueuesStorage					m_queues;
	VertexStorage						m_light_vertices;

	float										m_image_brightness;
	uint32										m_n_lights;
	uint32										m_n_init_paths;
	uint32										m_n_connections;

	DomainBuffer<CUDA_BUFFER, uint8>			m_memory_pool;
	DomainBuffer<CUDA_BUFFER, float>			m_connections_cdf;
	DomainBuffer<HOST_BUFFER, float>			m_path_pdf_backup;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;
};
