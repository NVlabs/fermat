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

struct RPTVPLView
{
	float4*	pos;		// packed VPL geometry
	uint4*	gbuffer;	// packed VPL material descriptors
	float4*	weight;		// sampled BSDF value and probability
	uint32*	in_dir;		// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction and radiance
	float4* in_alpha;	// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance
};

struct RPTVPLStorage
{
	DomainBuffer<CUDA_BUFFER, float4>	m_pos;		// packed VPL geometry
	DomainBuffer<CUDA_BUFFER, uint4>	m_gbuffer;	// packed VPL material descriptors
	DomainBuffer<CUDA_BUFFER, float4>	m_weight;	// sampled BSDF value and probability
	DomainBuffer<CUDA_BUFFER, uint32>	m_in_dir;	// packed incoming (reversed, i.e. the outgoing one from the PT's perspective) direction
	DomainBuffer<CUDA_BUFFER, float4>	m_in_alpha;	// unpacked incoming (reversed, i.e. the outgoing one from the PT's perspective) radiance

	void alloc(const uint32 pixels)
	{
		m_pos.alloc(pixels);
		m_gbuffer.alloc(pixels);
		m_weight.alloc(pixels);
		m_in_dir.alloc(pixels);
		m_in_alpha.alloc(pixels);
	}

	RPTVPLView view()
	{
		RPTVPLView r;
		r.pos		= m_pos.ptr();
		r.gbuffer	= m_gbuffer.ptr();
		r.weight	= m_weight.ptr();
		r.in_dir	= m_in_dir.ptr();
		r.in_alpha	= m_in_alpha.ptr();
		return r;
	}
};

struct RPT : RendererInterface
{
	RPT();

	void init(int argc, char** argv, Renderer& renderer);

	void render(const uint32 instance, Renderer& renderer);

	void setup_samples(const uint32 instance);

	void destroy() { delete this; }

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
