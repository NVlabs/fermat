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

#include <bpt.h>
#include <renderer.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_matrix.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/sampling/lfsr.h>
#include <cugar/sampling/random.h>
#include <cugar/sampling/multijitter.h>
#include <cugar/bsdf/lambert.h>
#include <cugar/bsdf/ggx.h>
#include <cugar/color/rgbe.h>
#include <cugar/basic/cuda/warp_atomics.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_utils.h>
#include <bpt_context.h>
#include <bpt_kernels.h>
#include <bpt_control.h>
#include <bpt_samplers.h>
#include <ray_queues.h>
#include <vector>

#define SHIFT_RES	256u

#define DEBUG_PIXEL (723 + 90*1600)

///@addtogroup Fermat
///@{

///@addtogroup BPTModule
///@{

/// The CUDA context class for the BPT renderer
///
struct BPTContext : BPTContextBase
{
	BPTContext(
		BPT&		_bpt,
		const RendererView& _renderer) :
		BPTContextBase(
			_renderer,
			_bpt.m_light_vertices.view(),
			_bpt.m_queues.view(_renderer.res_x * _renderer.res_y, _bpt.m_n_light_subpaths)),
		options(_bpt.m_options),
		sequence(_bpt.m_sequence.view()) {}

	BPTOptions	options;
	TiledSequenceView	sequence;
};

///@} BPTModule
///@} Fermat

namespace { // anonymous namespace

	///@addtogroup Fermat
	///@{

	///@addtogroup BPTModule
	///@{

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
			BPTContext&		context,
			RendererView&			renderer) const
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

	///
	/// The \ref SampleSinkAnchor "Sample Sink" used by BPT
	///
	template <bool USE_ATOMICS>
	struct ConnectionsSink : SampleSinkBase
	{
		FERMAT_HOST_DEVICE
		ConnectionsSink() {}

		FERMAT_HOST_DEVICE
		void sink(
			const uint32			channel,
			const cugar::Vector4f	value,
			const uint32			light_path_id,
			const uint32			eye_path_id,
			const uint32			s,
			const uint32			t,
			BPTContext&				context,
			RendererView&			renderer)
		{
			if (USE_ATOMICS)
			{
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).x, value.x / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).y, value.y / float(renderer.instance + 1));
				cugar::atomic_add(&renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id).z, value.z / float(renderer.instance + 1));

				if (channel != FBufferDesc::COMPOSITED_C)
				{
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).x, value.x / float(renderer.instance + 1));
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).y, value.y / float(renderer.instance + 1));
					cugar::atomic_add(&renderer.fb(channel, eye_path_id).z, value.z / float(renderer.instance + 1));
				}
			}
			else
			{
				renderer.fb(FBufferDesc::COMPOSITED_C, eye_path_id) += value / float(renderer.instance + 1);

				if (channel != FBufferDesc::COMPOSITED_C)
					renderer.fb(channel, eye_path_id) += value / float(renderer.instance + 1);
			}
		}

		// record an eye scattering event
		FERMAT_HOST_DEVICE
		void sink_eye_scattering_event(
			const Bsdf::ComponentType	component,
			const cugar::Vector4f		value,
			const uint32				eye_path_id,
			const uint32				t,
			BPTContext&					context,
			RendererView&				renderer)
		{
			if (t == 2) // accumulate the albedo of visible surfaces
			{
				if (component == Bsdf::kDiffuseReflection)
					renderer.fb(FBufferDesc::DIFFUSE_A, eye_path_id) += value / float(renderer.instance + 1);
				else if (component == Bsdf::kGlossyReflection)
					renderer.fb(FBufferDesc::SPECULAR_A, eye_path_id) += value / float(renderer.instance + 1);
			}
		}
	};

	///@} BPTModule
	///@} Fermat

} // anonymous namespace

BPT::BPT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void BPT::init(int argc, char** argv, Renderer& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	const uint32 n_light_paths = n_pixels;

	fprintf(stderr, "  creatign mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.m_mesh_lights.init( n_light_paths, renderer.m_mesh.view(), renderer.m_mesh_d.view(), renderer.m_texture_views_h.ptr(), renderer.m_texture_views_d.ptr() );

	fprintf(stderr, "  creatign mesh lights... done\n");

	// parse the options
	m_options.parse(argc, argv);

	// if we perform a single connection, RR must be enabled
	if (m_options.single_connection)
		m_options.rr = true;

	fprintf(stderr, "  BPT settings:\n");
	fprintf(stderr, "    single-conn    : %u\n", m_options.single_connection);
	fprintf(stderr, "    RR             : %u\n", m_options.rr);
	fprintf(stderr, "    path-length    : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee     : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf    : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee   : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf  : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    light-tracing  : %f\n", m_options.light_tracing);

	const uint32 queue_size = std::max(std::max(n_pixels, n_light_paths) * 3, n_light_paths * m_options.max_path_length);

	fprintf(stderr, "  allocating ray queue storage... started (%u entries)\n", queue_size);

	// compensate for the amount of light vs eye subpaths, changing the screen sampling densities
	m_options.light_tracing *= float(n_light_paths) / float(n_pixels);

	// alloc ray queues
	m_queues.alloc(n_pixels, n_light_paths, m_options.max_path_length);
	
	fprintf(stderr, "  allocating ray queue storage...done\n");

	// build the set of shifts
	const uint32 n_dimensions = (m_options.max_path_length + 1) * 2 * 6;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	m_sequence.setup(n_dimensions, SHIFT_RES);

	fprintf(stderr, "  allocating light vertex storage... started (%u paths, %u vertices)\n", n_light_paths, n_light_paths * m_options.max_path_length);
	m_light_vertices.alloc(n_light_paths, n_light_paths * m_options.max_path_length);
	fprintf(stderr, "  allocating light vertex storage... done\n");

	if (m_options.use_vpls)
		cudaMemcpy(m_light_vertices.vertex.ptr(), cugar::raw_pointer(renderer.m_mesh_lights.vpls), sizeof(VPL)*n_light_paths, cudaMemcpyDeviceToDevice);

	m_n_light_subpaths = n_light_paths;
	m_n_eye_subpaths   = n_pixels;
}

void BPT::render(const uint32 instance, Renderer& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.multiply_frame(float(instance) / float(instance + 1));

	cugar::Timer timer;
	timer.start();

	// get a view of the renderer
	RendererView renderer_view = renderer.view(instance);
	
	// initialize the sampling sequence for this frame
	m_sequence.set_instance(instance);
	
	// setup our BPT context
	BPTContext context(*this,renderer_view);

	// setup our BPT configuration
	BPTConfig config(context);
	
	// sample a set of bidirectional paths corresponding to our current primary coordinates
	TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

	PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.m_res_x, renderer.m_res_y);

	ConnectionsSink<false> sink;
	
	// debug only: regenerate the VPLs
	/*renderer.m_mesh_lights.init(
		n_light_paths,
		renderer.m_mesh.view(),
		renderer.m_mesh_d.view(),
		renderer.m_texture_views_h.ptr(),
		renderer.m_texture_views_d.ptr(),
		instance);

	if (USE_VPLS)
		cudaMemcpy(m_light_vertices.vertex.ptr(), cugar::raw_pointer(renderer.m_mesh_lights.vpls), sizeof(VPL)*n_light_paths, cudaMemcpyDeviceToDevice);*/

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
	if (instance == 0)
		m_time = time;
	else
		m_time += time;

	fprintf(stderr, "\r  %.1fs (%.1fms)        ",
		m_time,
		time * 1000.0f);
}
