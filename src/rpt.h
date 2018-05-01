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

///@defgroup RPTModule Reuse-based Path Tracer Module
/// this module implements a reuse-based path tracer inspired by the paper:
/// "Accelerating path tracing by re-using paths", by Bekaert et al
/// EGRW '02 Proceedings of the 13th Eurographics workshop on Rendering
/// Pages 125-134
///@{

/// RPT VPL view object
/// The reuse-based path tracer encodes ray hits as "virtual point lights" to be used by neighboring pixels;
/// this class implements a view on their storage.
///
struct RPTVPLView
{
	float4*	pos;		// packed VPL geometry
	uint4*	gbuffer;	// packed VPL material descriptors
	uint4*	ebuffer;	// packed VPL edf descriptors
	float4*	weight;		// sampled BSDF value and probability
	float4*	weight2;	// sampled BSDF value and probability
	uint32*	in_dir;		// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction and radiance
	uint32*	in_dir2;	// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction and radiance
	float4* in_alpha;	// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance
	float4* in_alpha2;	// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance
};

/// RPT VPL storage:
/// The reuse-based path tracer encodes ray hits as "virtual point lights" to be used by neighboring pixels;
/// this class implements their storage.
///
struct RPTVPLStorage
{
	DomainBuffer<CUDA_BUFFER, float4>	m_pos;		// packed VPL geometry
	DomainBuffer<CUDA_BUFFER, uint4>	m_gbuffer;	// packed VPL material descriptors
	DomainBuffer<CUDA_BUFFER, uint4>	m_ebuffer;	// packed VPL material descriptors
	DomainBuffer<CUDA_BUFFER, float4>	m_weight;	// sampled BSDF value and probability
	DomainBuffer<CUDA_BUFFER, float4>	m_weight2;	// sampled BSDF value and probability
	DomainBuffer<CUDA_BUFFER, uint32>	m_in_dir;	// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction
	DomainBuffer<CUDA_BUFFER, uint32>	m_in_dir2;	// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction
	DomainBuffer<CUDA_BUFFER, float4>	m_in_alpha;	// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance
	DomainBuffer<CUDA_BUFFER, float4>	m_in_alpha2;// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance

	void alloc(const uint32 pixels)
	{
		m_pos.alloc(pixels);
		m_gbuffer.alloc(pixels);
		m_ebuffer.alloc(pixels);
		m_weight.alloc(pixels);
		m_weight2.alloc(pixels);
		m_in_dir.alloc(pixels);
		m_in_dir2.alloc(pixels);
		m_in_alpha.alloc(pixels);
		m_in_alpha2.alloc(pixels);
	}

	RPTVPLView view()
	{
		RPTVPLView r;
		r.pos		= m_pos.ptr();
		r.gbuffer	= m_gbuffer.ptr();
		r.ebuffer	= m_ebuffer.ptr();
		r.weight	= m_weight.ptr();
		r.weight2	= m_weight2.ptr();
		r.in_dir	= m_in_dir.ptr();
		r.in_dir2	= m_in_dir2.ptr();
		r.in_alpha	= m_in_alpha.ptr();
		r.in_alpha2	= m_in_alpha2.ptr();
		return r;
	}
};

/// Reuse-based Path tracer renderer options
///
struct RPTOptions
{
	uint32  filter_width;
	uint32	max_path_length;
	bool	direct_lighting;
	bool	direct_lighting_nee;
	bool	direct_lighting_bsdf;
	bool	indirect_lighting_nee;
	bool	indirect_lighting_bsdf;
	bool	visible_lights;
	bool	diffuse_scattering;
	bool	glossy_scattering;
	bool	indirect_glossy;
	bool	rr;
	bool	tiled_reuse;

	RPTOptions() :
		max_path_length(6),
		direct_lighting_nee(true),
		direct_lighting_bsdf(true),
		indirect_lighting_nee(true),
		indirect_lighting_bsdf(true),
		visible_lights(true),
		direct_lighting(true),
		diffuse_scattering(true),
		glossy_scattering(true),
		indirect_glossy(false),
		rr(true),
		filter_width(30),
		tiled_reuse(false)
	{}

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
			else if (strcmp(argv[i], "-bsdf") == 0)
				direct_lighting_bsdf = indirect_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-nee") == 0)
				direct_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-bsdf") == 0)
				direct_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-nee") == 0)
				indirect_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-bsdf") == 0)
				indirect_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-visible-lights") == 0)
				visible_lights = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-lighting") == 0)
				direct_lighting = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-glossy") == 0)
				indirect_glossy = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-diffuse") == 0)
				diffuse_scattering = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-glossy") == 0)
				glossy_scattering = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-rr") == 0)
				rr = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-filter-width") == 0)
				filter_width = (uint32)atof(argv[++i]);
			else if (strcmp(argv[i], "-tiled") == 0 ||
					 strcmp(argv[i], "-tiled-reuse") == 0)
				tiled_reuse = atoi(argv[++i]) > 0;
		}
	}
};

/// Reuse-based Path tracer:
/// This class implements a reuse-based path tracer, inspired by the paper:
/// "Accelerating path tracing by re-using paths", by Bekaert et al
/// EGRW '02 Proceedings of the 13th Eurographics workshop on Rendering
/// Pages 125-134
///
struct RPT : RendererInterface
{
	RPT();

	void init(int argc, char** argv, Renderer& renderer);

	void render(const uint32 instance, Renderer& renderer);

	void setup_samples(const uint32 instance);

	void destroy() { delete this; }

	RPTOptions							m_options;

	DomainBuffer<CUDA_BUFFER, Ray>		m_rays;
	DomainBuffer<CUDA_BUFFER, Hit>		m_hits;
	DomainBuffer<CUDA_BUFFER, float4>	m_weights;
	DomainBuffer<CUDA_BUFFER, float4>	m_weights2;
	DomainBuffer<CUDA_BUFFER, float>	m_probs;
	DomainBuffer<CUDA_BUFFER, uint32>	m_pixels;
	DomainBuffer<CUDA_BUFFER, uint32>	m_counters;

	RPTVPLStorage						m_vpls;

	TiledSequence				m_sequence;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;

	float						m_time;
};

///@} RPTModule
///@} Fermat
