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

#include <pssmlt.h>
#include <renderer.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/sampling/distributions.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_context.h>
#include <bpt_control.h>
#include <bpt_samplers.h>
#include <random_sequence.h>
#include <path_inversion.h>
#include <ray_queues.h>
#include <vector>

#define SHIFT_RES	256u

#define DEBUG_PIXEL (714 + 66*1600)

#define CONNECTIONS_COUNTER 3

namespace {

	FERMAT_HOST_DEVICE
	uint32 chain_coordinate_index(const uint32 n_chains, const uint32 idx, const uint32 dim)
	{
		return dim*n_chains + idx;
	}

} // anonymous namespace

///@addtogroup Fermat
///@{

///@addtogroup PSSMLTModule
///@{

/// The CUDA context class for the PSSMLT renderer
///
struct PSSMLTContext : BPTContextBase<PSSMLTOptions>
{
	PSSMLTContext(
		PSSMLT&							_pssmlt,
		const RenderingContextView&		_renderer) :
		BPTContextBase<PSSMLTOptions>(
			_renderer,
			_pssmlt.m_light_vertices.view(),
			_pssmlt.m_queues.view(_pssmlt.m_n_init_paths, _pssmlt.m_n_init_light_paths),
			_pssmlt.m_options),
		sequence			(_pssmlt.m_sequence.view()),
		connections_value	(_pssmlt.m_connections_value.ptr()),
		st_norms			(_pssmlt.m_st_norms.ptr()),
		seeds				(_pssmlt.m_seeds.ptr()),
		n_chains			(_pssmlt.m_options.n_chains),
		mut_u				(_pssmlt.m_mut_u.ptr()),
		light_u				(_pssmlt.m_light_u.ptr()),
		eye_u				(_pssmlt.m_eye_u.ptr()),
		path_value			(_pssmlt.m_path_value.ptr()),
		path_pdf			(_pssmlt.m_path_pdf.ptr()),
		new_path_value		(_pssmlt.m_path_value.ptr() + _pssmlt.m_options.n_chains),
		rejections			(_pssmlt.m_rejections.ptr()),
		mutation_type		(PerturbedPrimaryCoords::CauchyPerturbation)
	{}

	TiledSequenceView	sequence;
	float4*				connections_value;
	float*				st_norms;
	uint32*				seeds;
	float*				mut_u;
	float*				light_u;
	float*				eye_u;
	float4*				new_path_value;
	float4*				path_value;
	float*				path_pdf;
	uint32*				rejections;
	uint32				n_chains;
	float				pdf_norm;

	PerturbedPrimaryCoords::Type	mutation_type;

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float u_ar(const uint32 chain_id) const { return mut_u ? mut_u[chain_coordinate_index(n_chains, chain_id, (options.max_path_length + 1) * 3 * 2)] : 0.5f; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float& u_L(const uint32 chain_id, const uint32 dim) { return light_u[chain_coordinate_index(n_chains, chain_id, dim)]; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float& u_E(const uint32 chain_id, const uint32 dim) { return eye_u[chain_coordinate_index(n_chains, chain_id, dim)]; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	PerturbedPrimaryCoords light_primary_coords() const
	{
		return PerturbedPrimaryCoords(
			n_chains,
			light_u, 0u,
			mut_u, 0u,
			options.light_perturbations ? mutation_type : PerturbedPrimaryCoords::Null);
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	PerturbedPrimaryCoords eye_primary_coords() const
	{
		return PerturbedPrimaryCoords(
			n_chains,
			eye_u, 0u,
			mut_u, (options.max_path_length + 1),
			options.eye_perturbations ? mutation_type : PerturbedPrimaryCoords::Null);
	}
};

///@} PSSMLTModule
///@} Fermat

namespace { // anonymous namespace

	///@addtogroup Fermat
	///@{

	///@addtogroup PSSMLTModule
	///@{

	FERMAT_DEVICE FERMAT_FORCEINLINE
	void accept_reject_accumulate(const uint32 chain_id, const cugar::Vector4f w, PSSMLTContext& context, RenderingContextView& renderer)
	{
		// perform the MH acceptance-rejection test
		const float new_pdf = cugar::max_comp(w.xyz());
		const float old_pdf = context.path_pdf[chain_id];

		PerturbedPrimaryCoords eye_primary_coords = context.eye_primary_coords();

		const cugar::Vector2f old_uv(
			eye_primary_coords.u(chain_id, 3 + 0),
			eye_primary_coords.u(chain_id, 3 + 1));

		const cugar::Vector2f new_uv(
			eye_primary_coords.perturbed_u(chain_id, 3 + 0),
			eye_primary_coords.perturbed_u(chain_id, 3 + 1));

		const uint32 old_pixel_x = cugar::quantize(old_uv.x, renderer.res_x);
		const uint32 old_pixel_y = cugar::quantize(old_uv.y, renderer.res_y);
		const uint32 old_pixel = old_pixel_x + old_pixel_y*renderer.res_x;

		const uint32 new_pixel_x = cugar::quantize(new_uv.x, renderer.res_x);
		const uint32 new_pixel_y = cugar::quantize(new_uv.y, renderer.res_y);
		const uint32 new_pixel = new_pixel_x + new_pixel_y*renderer.res_x;

		const float ar = old_pdf ? fminf(1.0f, new_pdf / old_pdf) : 1.0f;

		if (old_pdf > 0)
		{
			const float4 old_value = context.path_value[chain_id];

			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).x, context.pdf_norm * (1.0f - ar) * (old_value.x / old_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).y, context.pdf_norm * (1.0f - ar) * (old_value.y / old_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, old_pixel).z, context.pdf_norm * (1.0f - ar) * (old_value.z / old_pdf) / float(renderer.instance + 1));
		}
		if (new_pdf > 0)
		{
			const float4 new_value = w;

			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).x, context.pdf_norm * ar * (new_value.x / new_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).y, context.pdf_norm * ar * (new_value.y / new_pdf) / float(renderer.instance + 1));
			atomicAdd(&renderer.fb(FBufferDesc::COMPOSITED_C, new_pixel).z, context.pdf_norm * ar * (new_value.z / new_pdf) / float(renderer.instance + 1));
		}

		if (context.u_ar(chain_id) * old_pdf < new_pdf)
		{
			context.path_value[chain_id] = w;
			context.path_pdf[chain_id] = new_pdf;

			PerturbedPrimaryCoords light_primary_coords = context.light_primary_coords();

			// copy the successful mutation coordinates
			for (uint32 i = 0; i < (context.options.max_path_length + 1) * 3; ++i)
				light_primary_coords.u(chain_id, i) = light_primary_coords.perturbed_u(chain_id, i);

			// copy the successful mutation coordinates
			for (uint32 i = 0; i < (context.options.max_path_length + 1) * 3; ++i)
				eye_primary_coords.u(chain_id, i) = eye_primary_coords.perturbed_u(chain_id, i);

			// reset the rejections counter
			context.rejections[chain_id] = 0;
		}
		else
		{
			// increase the rejections counter
			context.rejections[chain_id]++;

			//if ((context.rejections[chain_id] % 5) == 0)
			//	printf("chain[%u] stuck for %u iterations\n", chain_id, context.rejections[chain_id]);
		}
	}

	///
	/// The BPT configuration used by the BPT presampling/seeding pass
	///
	struct PSSMLTPresamplingBPTConfig : BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		PSSMLTPresamplingBPTConfig(PSSMLTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kAll,
				VertexOrdering::kPathOrdering,
				VertexSampling::kAll,
				_context.options.rr) {}
	};

	///
	/// The BPT configuration used by the MLT pass
	///
	struct PSSMLTChainBPTConfig : BPTConfigBase
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		PSSMLTChainBPTConfig(PSSMLTContext& _context) :
			BPTConfigBase(
				_context.options,
				VertexSampling::kAll,
				VertexOrdering::kPathOrdering,
				VertexSampling::kAll,
				_context.options.rr) {}
	};

	///
	/// The \ref SampleSinkAnchor "Sample Sink" used for the BPT presampling/seeding pass
	///
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
			PSSMLTContext&			context,
			RenderingContextView&			renderer)
		{
			// TODO: keep per-channel accumulators...
			cugar::atomic_add(&context.connections_value[eye_path_id].x, value.x);
			cugar::atomic_add(&context.connections_value[eye_path_id].y, value.y);
			cugar::atomic_add(&context.connections_value[eye_path_id].z, value.z);

			cugar::atomic_add(context.st_norms + s + t * (context.options.max_path_length + 2), cugar::max_comp(value.xyz()));
		}
	};

	///
	/// The \ref SampleSinkAnchor "Sample Sink" used by the actual MLT chain sampler
	///
	struct ChainSamplerSink : SampleSinkBase
	{
		FERMAT_HOST_DEVICE
		ChainSamplerSink() {}

		FERMAT_HOST_DEVICE
		void sink(
			const uint32			channel,
			const cugar::Vector4f	value,
			const uint32			light_path_id,
			const uint32			eye_path_id,
			const uint32			s,
			const uint32			t,
			PSSMLTContext&			context,
			RenderingContextView&			renderer)
		{
			// accumulate the sample value to the per-chain sum
			context.new_path_value[eye_path_id] += value;
		}
	};

	//------------------------------------------------------------------------------

	__global__
	void accept_reject_mlt_kernel(PSSMLTContext context, RenderingContextView renderer)
	{
		const uint32 chain_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (chain_id < context.n_chains) // *context.shadow_queue.size
			accept_reject_accumulate(chain_id, context.new_path_value[chain_id], context, renderer);
	}

	void accept_reject_mlt(PSSMLTContext context, RenderingContextView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(context.n_chains, blockSize));
		accept_reject_mlt_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	//------------------------------------------------------------------------------

	__global__
	void sample_seeds_kernel(const uint32 n_connections, const float* connections_cdf, const uint32 n_seeds, uint32* seeds)
	{
		const uint32 seed_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (seed_id < n_seeds)
		{
			// pick a stratified sample
			const float r = (seed_id + cugar::randfloat( seed_id, 0 )) / float(n_seeds);

			seeds[seed_id] = cugar::upper_bound_index( r * connections_cdf[n_connections-1], connections_cdf, n_connections);
		}
	}

	void sample_seeds(const uint32 n_connections, const float* connections_cdf, const uint32 n_seeds, uint32* seeds)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		sample_seeds_kernel<<< gridSize, blockSize >>>(n_connections, connections_cdf, n_seeds, seeds);
	}

	//------------------------------------------------------------------------------

	__global__
	void recover_primary_coordinates_kernel(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, PSSMLTContext context, RenderingContextView renderer)
	{
		const uint32 seed_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (seed_id < n_seeds)
		{
			const uint32 seed = seeds[seed_id];

			const uint32 light_idx	= seed;
			const uint32 eye_idx	= seed;
			const uint32 s			= context.options.max_path_length + 1;
			const uint32 t			= context.options.max_path_length + 1;

			TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

			for (uint32 i = 0; i < s; ++i)
			{
				if (i == 0)
				{
					// the first vertex might be somewhat special
					if (context.options.use_vpls)
						context.u_L(seed_id, i * 3 + 0) = (float(light_idx) + light_primary_coords.sample(light_idx, i, 0)) / float(n_lights);
					else
						context.u_L(seed_id, i * 3 + 0) = light_primary_coords.sample(light_idx, i, 0);

					context.u_L(seed_id, i * 3 + 1) = light_primary_coords.sample(light_idx, i, 1);
					context.u_L(seed_id, i * 3 + 2) = light_primary_coords.sample(light_idx, i, 2);
				}
				else
				{
					// fetch the regular coordinates
					context.u_L(seed_id, i * 3 + 0) = light_primary_coords.sample(light_idx, i, 0);
					context.u_L(seed_id, i * 3 + 1) = light_primary_coords.sample(light_idx, i, 1);
					context.u_L(seed_id, i * 3 + 2) = light_primary_coords.sample(light_idx, i, 2);
				}
			}

			PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.res_x, renderer.res_y);

			for (uint32 i = 0; i < t; ++i)
			{
				if (i == 0)
				{
					// we set the lens sample to (0,0,0)
					context.u_E(seed_id, i * 3 + 0) = 0;
					context.u_E(seed_id, i * 3 + 1) = 0;
					context.u_E(seed_id, i * 3 + 2) = 0;
				}
				else
				{
					// fetch the regular coordinates
					context.u_E(seed_id, i * 3 + 0) = eye_primary_coords.sample(eye_idx, i, 0);
					context.u_E(seed_id, i * 3 + 1) = eye_primary_coords.sample(eye_idx, i, 1);
					context.u_E(seed_id, i * 3 + 2) = eye_primary_coords.sample(eye_idx, i, 2);
				}
			}
		}
	}
	void recover_primary_coordinates(const uint32 n_seeds, const uint32* seeds, const uint32 n_lights, PSSMLTContext context, RenderingContextView renderer)
	{
		dim3 blockSize(128);
		dim3 gridSize(cugar::divide_ri(n_seeds, blockSize.x));
		recover_primary_coordinates_kernel <<< gridSize, blockSize >>>(n_seeds, seeds, n_lights, context, renderer);
	}

	///@} PSSMLTModule
	///@} Fermat

	//------------------------------------------------------------------------------
} // anonymous namespace

PSSMLT::PSSMLT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void PSSMLT::init(int argc, char** argv, RenderingContext& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	const uint32 n_init_light_paths = n_pixels;

	m_n_init_light_paths = n_init_light_paths;
	m_n_init_paths		 = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	// TODO: re-enable when light tracing is implemented
	m_options.light_tracing = 0.0f;

	// compute how long our chains are
	const uint32 chain_length = (m_options.spp * n_pixels) / m_options.n_chains;

	fprintf(stderr, "  PSSMLT settings:\n");
	fprintf(stderr, "    spp            : %u\n", m_options.spp);
	fprintf(stderr, "    chains         : %u\n", m_options.n_chains);
	fprintf(stderr, "    chain-length   : %u\n", chain_length);
	fprintf(stderr, "    RR             : %u\n", m_options.rr);
	fprintf(stderr, "    independent    : %f\n", m_options.independent_samples);
	fprintf(stderr, "    path-length    : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee     : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf    : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee   : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf  : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    light-tracing  : %u\n", m_options.light_tracing ? 1 : 0);

	const uint32 n_chains = m_options.n_chains;

	fprintf(stderr, "  creating mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.get_mesh_lights().init( n_init_light_paths, renderer );

	fprintf(stderr, "  creating mesh lights... done\n");

	// alloc ray queues
	m_queues.alloc(n_pixels, n_init_light_paths, m_options.max_path_length);

	// build the shifted sample sequence
	const uint32 n_dimensions = (m_options.max_path_length + 1) * 2 * 6;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	m_sequence.setup(n_dimensions, SHIFT_RES);

	fprintf(stderr, "  allocating light vertex storage... started");
	m_light_vertices.alloc(n_init_light_paths, n_init_light_paths * (m_options.max_path_length + 1));
	fprintf(stderr, "  allocating light vertex storage storage... done\n");

	if (m_options.use_vpls)
		cudaMemcpy(m_light_vertices.vertex.ptr(), renderer.get_mesh_lights().get_vpls(), sizeof(VPL)*n_init_light_paths, cudaMemcpyDeviceToDevice);

	fprintf(stderr, "  allocating bpt connections... started\n");
	m_connections_value.alloc(n_pixels * m_options.max_path_length);
	m_connections_cdf.alloc(n_pixels);
	m_seeds.alloc(n_chains);
	fprintf(stderr, "  allocating bpt connections... done\n");

	fprintf(stderr, "  allocating chain storage... started\n");
	m_mut_u.alloc(n_chains*((m_options.max_path_length + 1) * 3 * 2 + 2));
	m_light_u.alloc(n_chains*(m_options.max_path_length + 1) * 3);
	m_eye_u.alloc(n_chains*(m_options.max_path_length + 1) * 3);
	m_path_value.alloc(n_chains * 2);
	m_path_pdf.alloc(n_chains);
	m_rejections.alloc(n_chains);
	m_st_norms.alloc((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	fprintf(stderr, "  allocating chain storage... done\n");
}

struct max_comp_functor
{
	typedef float4	argument_type;
	typedef float   result_type;

	FERMAT_HOST_DEVICE
	float operator() (const argument_type c) { return cugar::max3(c.x, c.y, c.z); }
};

// sample seed paths from the stored bidirectional connections
//
void PSSMLT::sample_seeds(const uint32 n_chains)
{
	cugar::device_vector<uint8> temp_storage;

	const uint32 n_connections = m_n_init_paths;

	// compute the connections CDF
	cugar::inclusive_scan<cugar::device_tag>(
		n_connections,
		thrust::make_transform_iterator(m_connections_value.ptr(), max_comp_functor()),
		m_connections_cdf.ptr(),
		thrust::plus<float>(),
		temp_storage);

	// resample n_chains of them
	::sample_seeds(n_connections, m_connections_cdf.ptr(), n_chains, m_seeds.ptr());

	// the total image brightness is the sum of all contributions, divided by the number of paths we shot
	m_image_brightness = m_connections_cdf[n_connections - 1] / float(m_n_init_paths);
}

// recover primary sample space coordinates of the sampled light and eye subpaths
//
void PSSMLT::recover_primary_coordinates(PSSMLTContext& context, RenderingContextView& renderer_view)
{
	::recover_primary_coordinates( context.n_chains, m_seeds.ptr(), m_n_lights, context, renderer_view );
}

void PSSMLT::render(const uint32 instance, RenderingContext& renderer)
{
	// clear the global timer at instance zero
	if (instance == 0)
		m_time = 0.0f;

	// pre-multiply the previous frame for blending
	renderer.multiply_frame(float(instance) / float(instance + 1));

	cugar::Timer timer;

	timer.start();

	// get a view of the renderer
	RenderingContextView renderer_view = renderer.view(instance);

	const uint2  res			= renderer.res();
	const uint32 n_pixels		= res.x * res.y;
	const uint32 n_chains		= m_options.n_chains;
	const uint32 chain_length	= (m_options.spp * n_pixels) / n_chains;

	// initialize the sampling sequence for this frame
	m_sequence.set_instance(instance);

	// setup our BPT context
	PSSMLTContext context(*this, renderer_view);

	// perform the BPT presampling pass
	{
		// reset the connections
		cudaMemset(context.connections_value, 0x00, sizeof(float4)*m_n_init_paths);

		// zero out the stats
		cudaMemset(m_st_norms.ptr(), 0x00, sizeof(float)*(context.options.max_path_length + 2)*(context.options.max_path_length + 2));

		TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

		PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, renderer.res().x, renderer.res().y);

		PSSMLTPresamplingBPTConfig config(context);

		ConnectionsSink connections_sink;

		bpt::sample_paths(
			m_n_init_paths,
			m_n_init_light_paths,
			eye_primary_coords,
			light_primary_coords,
			connections_sink,
			context,
			config,
			renderer,
			renderer_view);
	}

	timer.stop();
	const float presampling_time = timer.seconds();

	timer.start();

	// resample the seeds for our chains to remove startup bias
	sample_seeds(n_chains);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("sample seeds"));

	// print out the chain stats
	if (instance == 0)
	{
		fprintf(stderr, "  image brightness: %f\n", m_image_brightness);
		fprintf(stderr, "  st chains\n");
		float  st_norms[1024];
		cudaMemcpy(st_norms, m_st_norms.ptr(), sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);
		for (uint32 s = 0; s < m_options.max_path_length; ++s)
		{
			for (uint32 t = 0; t < m_options.max_path_length + 2; ++t)
			{
				if (s + t <= m_options.max_path_length + 1)
					fprintf(stderr, "    [%u,%u] : %f\n", s, t, st_norms[s + t*(m_options.max_path_length + 2)] / float(m_n_init_paths));
			}
		}
	}

	// setup the normalization constant
	context.pdf_norm = m_image_brightness * float(res.x * res.y) / float(chain_length * n_chains);

	// recover the primary coordinates of the selected seed paths
	recover_primary_coordinates(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("recover primary coordinates"));

	timer.stop();
	const float seed_resampling_time = timer.seconds();

	timer.start();

	// initialize the initial path pdfs to zero
	cudaMemset(context.path_pdf, 0x00, sizeof(float)*n_chains);

	// initialize the rejection counters to zero
	cudaMemset(context.rejections, 0x00, sizeof(uint32)*n_chains);

	// initialize the random generator
	DeviceRandomSequence sequence;

	// start the chains
	for (uint32 l = 0; l < chain_length; ++l)
	{
		// disable any kind of mutations for the first step, and enable them later on
		context.mutation_type = (l == 0) ? PerturbedPrimaryCoords::Null :
								m_random.next() < m_options.independent_samples ?
											PerturbedPrimaryCoords::IndependentSample :
											PerturbedPrimaryCoords::CauchyPerturbation;

		if (l > 0)
		{
			// generate all the random numbers needed for mutations
			sequence.next(n_chains*((context.options.max_path_length + 1) * 2 * 3 + 1), context.mut_u);
		}

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

		// reset the new path values
		cudaMemset(context.new_path_value, 0x00, sizeof(float4)*n_chains);

		// sample a set of bidirectional paths corresponding to our current primary coordinates
		PSSMLTChainBPTConfig config(context);

		PerturbedPrimaryCoords light_primary_coords = context.light_primary_coords();
		PerturbedPrimaryCoords eye_primary_coords   = context.eye_primary_coords();

		ChainSamplerSink chain_sink;

		bpt::sample_paths(
			context.n_chains,
			context.n_chains,
			eye_primary_coords,
			light_primary_coords,
			chain_sink,
			context,
			config,
			renderer,
			renderer_view,
			false);			// lazy shadows

		// and perform the usual acceptance-rejection step
		accept_reject_mlt(context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("accept-reject"));
	}

	timer.stop();
	const float mlt_time = timer.seconds();

	m_time += presampling_time + seed_resampling_time + mlt_time;

	fprintf(stderr, "\r  %.1fs (%.1fms = init: %.1fms, seed: %.1fms, mut: %.1fms)        ",
		m_time,
		(presampling_time + seed_resampling_time + mlt_time) * 1000.0f,
		presampling_time * 1000.0f,
		seed_resampling_time * 1000.0f,
		mlt_time * 1000.0f);
}