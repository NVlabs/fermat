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

struct CMLTContext;

///@addtogroup Fermat
///@{

///@defgroup CMLTModule
/// This module provides an implementation of a Charted Metropolis light transport sampler, as described in:
///   <a href="https://arxiv.org/abs/1612.05395">Charted Metropolis Light Transport</a>, Jacopo Pantaleoni, ACM Transactions on Graphics, Volume 36 Issue 4, July 2017.
///@{

/// CMLT renderer options
///
struct CMLTOptions : BPTOptionsBase
{
	uint32	n_chains;
	uint32	spp;
	uint32	swap_frequency;
	uint32	mmlt_frequency;
	uint32	startup_skips;
	uint32  reseeding;
	bool	single_connection;
	bool	rr;
	bool    light_perturbations;
	bool	eye_perturbations;
	float   perturbation_radius;

	CMLTOptions() :
		BPTOptionsBase(),
		n_chains(256 * 1024),
		spp(1),
		swap_frequency(32),
		mmlt_frequency(0),
		startup_skips(0),
		single_connection(false),
		rr(false),
		light_perturbations(true),
		eye_perturbations(true),
		reseeding(8),
		perturbation_radius(0.01f)
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
			else if (strcmp(argv[i], "-c-swaps") == 0 || strcmp(argv[i], "-chart-swaps") == 0)
				swap_frequency = atoi(argv[++i]);
			else if (strcmp(argv[i], "-m-swaps") == 0 || strcmp(argv[i], "-mmlt-swaps") == 0)
				mmlt_frequency = atoi(argv[++i]);
			else if (strcmp(argv[i], "-skips") == 0 ||
					 strcmp(argv[i], "-startup-skips") == 0)
				startup_skips = atoi(argv[++i]);
			else if (strcmp(argv[i], "-sc") == 0 || 
					 strcmp(argv[i], "-single-conn") == 0 ||
					 strcmp(argv[i], "-single-connection") == 0)
				single_connection = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-rr") == 0 || strcmp(argv[i], "-RR") == 0)
				rr = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-light-perturbations") == 0)
				light_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-eye-perturbations") == 0)
				eye_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-perturbations") == 0)
				light_perturbations = eye_perturbations = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-rf") == 0 ||
					 strcmp(argv[i], "-reseed-freq") == 0 ||
					 strcmp(argv[i], "-reseeding") == 0)
				reseeding = atoi(argv[++i]);
			else if (strcmp(argv[i], "-perturbation-radius") == 0 ||
				     strcmp(argv[i], "-p-radius") == 0 ||
				     strcmp(argv[i], "-pr") == 0)
				perturbation_radius = (float)atof(argv[++i]);
		}
	}
};

/// Charted MLT renderer
///
struct CMLT : RendererInterface
{
	CMLT();

	void init(int argc, char** argv, RenderingContext& renderer);

	void render(const uint32 instance, RenderingContext& renderer);

	void sample_seeds(const uint32 n_chains);

	void build_st_norms_cdf();

	void recover_primary_coordinates(CMLTContext& context, RenderingContextView& renderer_view);

	void destroy() { delete this; }

	static RendererInterface* factory() { return new CMLT(); }

	CMLTOptions							m_options;

	TiledSequence						m_sequence;

	BPTQueuesStorage					m_queues;

	VertexStorage						m_light_vertices;

	DomainBuffer<CUDA_BUFFER, float>			m_mut_u;
	DomainBuffer<CUDA_BUFFER, float>			m_light_u;
	DomainBuffer<CUDA_BUFFER, float>			m_eye_u;
	DomainBuffer<CUDA_BUFFER, float4>			m_path_value;
	DomainBuffer<CUDA_BUFFER, float>			m_path_pdf;
	DomainBuffer<CUDA_BUFFER, float4>			m_connections_value;
	DomainBuffer<CUDA_BUFFER, uint4>			m_connections_index;
	DomainBuffer<CUDA_BUFFER, float>			m_connections_cdf;
	DomainBuffer<CUDA_BUFFER, uint32>			m_connections_counter;
	DomainBuffer<CUDA_BUFFER, uint32>			m_seeds;
	DomainBuffer<CUDA_BUFFER, uint32>			m_st_counters;
	DomainBuffer<CUDA_BUFFER, float>			m_st_norms;
	DomainBuffer<CUDA_BUFFER, float>			m_st_norms_cdf;
	DomainBuffer<CUDA_BUFFER, uint32>			m_rejections;
	DomainBuffer<CUDA_BUFFER, VertexGeometryId>	m_vertices;
	float										m_image_brightness;
	uint32										m_n_init_light_paths;
	uint32										m_n_init_paths;
	uint32										m_n_chains;
	uint32										m_n_connections;
	float										m_time;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;
};

///@} CMLTModule
///@} Fermat
