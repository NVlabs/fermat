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

#include <bpt.h>
#include <renderer.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_utils.h>
#include <bpt_context.h>
#include <bpt_kernels.h>
#include <bpt_control.h>
#include <bpt_samplers.h>

#define SHIFT_RES	256u

#define DEBUG_PIXEL (723 + 90*1600)

///@addtogroup Fermat
///@{

///@addtogroup BPTModule
///@{

/// The CUDA context class for the BPT renderer
///
struct BPTContext : BPTContextBase<BPTOptions>
{
	BPTContext(
		BPT&		_bpt,
		const RenderingContextView& _renderer) :
		BPTContextBase(
			_renderer,
			_bpt.m_light_vertices.view(),
			_bpt.m_queues.view(_renderer.res_x * _renderer.res_y, _bpt.m_n_light_subpaths),
			_bpt.m_options),
		sequence(_bpt.m_sequence.view()) {}

	TiledSequenceView	sequence;
};

///@} BPTModule
///@} Fermat

namespace { // anonymous namespace

	///@addtogroup Fermat
	///@{

	///@addtogroup BPTModule
	///@{

	//! [BPTConfig]
	///
	/// The \ref TBPTConfig used by BPT
	///
	struct BPTConfig : public BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		BPTConfig(BPTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kAll,
				_context.options.single_connection ? VertexOrdering::kRandomOrdering : VertexOrdering::kPathOrdering,
				VertexSampling::kAll,
				_context.options.rr)
		{}

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		void visit_eye_vertex(
			const uint32			path_id,
			const uint32			depth,
			const VertexGeometryId	v_id,
			const EyeVertex&		v,
			BPTContext&				context,
			RenderingContextView&	renderer) const
		{
			// write out gbuffer information
			if (depth == 1)
			{
				renderer.fb.gbuffer.geo(path_id) = GBufferView::pack_geometry(v.geom.position, v.geom.normal_s);
				renderer.fb.gbuffer.uv(path_id)  = make_float4(v_id.uv.x, v_id.uv.y, v.geom.texture_coords.x, v.geom.texture_coords.y);
				renderer.fb.gbuffer.tri(path_id) = v_id.prim_id;
			}
		}
	};
	//! [BPTConfig]

	//! [BPTConnectionsSink]
	///
	/// The \ref TSampleSink used by BPT
	///
	/// \tparam	USE_ATOMICS	specify whether to use atomics or not
	template <bool USE_ATOMICS>
	struct ConnectionsSink : SampleSinkBase
	{
		FERMAT_HOST_DEVICE
		ConnectionsSink() {}

		// accumulate a bidirectional sample
		//
		FERMAT_HOST_DEVICE
		void sink(
			const uint32			channel,
			const cugar::Vector4f	value,
			const uint32			light_path_id,
			const uint32			eye_path_id,
			const uint32			s,
			const uint32			t,
			BPTContext&				context,
			RenderingContextView&	renderer)
		{
			const float frame_weight = 1.0f / float(renderer.instance + 1);

			if (USE_ATOMICS)
			{
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).x, value.x * frame_weight);
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).y, value.y * frame_weight);
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).z, value.z * frame_weight);

				if (channel != FBufferDesc::COMPOSITED_C)
				{
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).x, value.x * frame_weight);
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).y, value.y * frame_weight);
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).z, value.z * frame_weight);
				}
			}
			else
			{
				renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id) += value * frame_weight;

				if (channel != FBufferDesc::COMPOSITED_C)
					renderer.fb(channel, eye_path_id) += value * frame_weight;
			}
		}

		// record an eye scattering event
		//
		FERMAT_HOST_DEVICE
		void sink_eye_scattering_event(
			const Bsdf::ComponentType	component,
			const cugar::Vector4f		value,
			const uint32				eye_path_id,
			const uint32				t,
			BPTContext&					context,
			RenderingContextView&		renderer)
		{
			if (t == 2) // accumulate the albedo of visible surfaces
			{
				const float frame_weight = 1.0f / float(renderer.instance + 1);

				if (component == Bsdf::kDiffuseReflection)
					renderer.fb(FBufferDesc::DIFFUSE_A, eye_path_id) += value * frame_weight;
				else if (component == Bsdf::kGlossyReflection)
					renderer.fb(FBufferDesc::SPECULAR_A, eye_path_id) += value * frame_weight;
			}
		}
	};
	//! [BPTConnectionsSink]

	///@} BPTModule
	///@} Fermat

} // anonymous namespace


//! [BPT::render]
void BPT::render(const uint32 instance, RenderingContext& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.multiply_frame(float(instance) / float(instance + 1));

	cugar::Timer timer;
	timer.start();

	// get a view of the renderer
	RenderingContextView renderer_view = renderer.view(instance);
	
	// initialize the sampling sequence for this frame
	m_sequence.set_instance(instance);
	
	// setup our BPT context
	BPTContext context(*this,renderer_view);

	// setup our BPT configuration
	BPTConfig config(context);
	
	// sample a set of bidirectional paths corresponding to our current primary coordinates
	TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

	PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.res().x, renderer.res().y);

	ConnectionsSink<false> sink;
	
	// debug only: regenerate the VPLs
	//regenerate_primary_light_vertices(instance, renderer);

	bpt::sample_paths(
		m_n_eye_subpaths,
		m_n_light_subpaths,
		eye_primary_coords,
		light_primary_coords,
		sink,
		context,
		config,
		renderer,
		renderer_view);

	// solve pure light tracing occlusions
	{
		ConnectionsSink<true> atomic_sink;

		bpt::light_tracing(
			m_n_light_subpaths,
			atomic_sink,
			context,
			config,
			renderer,
			renderer_view);
	}

	timer.stop();
	const float time = timer.seconds();

	// clear the global timer at instance zero
	m_time = (instance == 0) ? time : time + m_time;

	fprintf(stderr, "\r  %.1fs (%.1fms)        ",
		m_time,
		time * 1000.0f);
}
//! [BPT::render]
