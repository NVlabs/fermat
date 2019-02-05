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

#include <mlt.h>
#include <mlt_core.h>
#include <mlt_kernels.h>
#include <cugar/basic/memory_arena.h>
#include <vector>


void MLT::alloc(MLTContext& context, cugar::memory_arena& arena, const uint32 n_pixels, const uint32 n_chains)
{
	context.connections_value	= arena.alloc<float4>(n_pixels * (m_options.max_path_length + 1));
	context.connections_index	= arena.alloc<uint4>(n_pixels * (m_options.max_path_length + 1));
	context.connections_counter = arena.alloc<uint32>(1);
	context.seeds				= arena.alloc<uint32>(n_chains);
	context.st					= arena.alloc<char4>(n_chains);
	context.st_norms			= arena.alloc<float>((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	context.st_norms_cdf		= arena.alloc<float>((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	context.st_counters			= arena.alloc<uint32>((m_options.max_path_length + 2)*(m_options.max_path_length + 2));
	context.path_value			= arena.alloc<float4>(n_chains);
	context.path_pdf			= arena.alloc<float>(n_chains);
	context.rejections			= arena.alloc<uint32>(n_chains);
	context.bpt_light_vertices  = arena.alloc<VertexGeometryId>(n_pixels * (m_options.max_path_length + 1));
	context.bpt_eye_vertices    = arena.alloc<VertexGeometryId>(n_pixels * (m_options.max_path_length + 1));
	context.mut_vertices		= arena.alloc<VertexGeometryId>(n_chains * (m_options.max_path_length + 1));
	context.vertices			= arena.alloc<VertexGeometryId>(n_chains * (m_options.max_path_length + 1));
	context.acceptance_rate_sum = arena.alloc<float>(1);
	context.checksum			= arena.alloc<float>(1);

	context.Q_old				= arena.alloc<cugar::Vector4f>(n_chains);
	context.Q_new				= arena.alloc<cugar::Vector4f>(n_chains);
	context.mut_f_vertices		= arena.alloc<float4>(n_chains * (m_options.max_path_length + 1));
	context.f_vertices			= arena.alloc<float4>(n_chains * (m_options.max_path_length + 1));
}

MLT::MLT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void MLT::init(int argc, char** argv, RenderingContext& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	const uint32 n_light_paths = n_pixels;
	
	fprintf(stderr, "  creatign mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.get_mesh_lights().init( n_light_paths, renderer, 0u );

	fprintf(stderr, "  creatign mesh lights... done\n");
	
	// parse options
	m_options.parse(argc, argv);

	const uint32 chain_length = (m_options.spp * n_pixels) / m_options.n_chains;

	fprintf(stderr, "  settings:\n");
	fprintf(stderr, "    spp             : %u\n", m_options.spp);
	fprintf(stderr, "    reseeding freq  : %u\n", m_options.reseeding_freq);
	fprintf(stderr, "    chains          : %u\n", m_options.n_chains);
	fprintf(stderr, "    chain-length    : %u (%u per frame)\n", chain_length * m_options.reseeding_freq, chain_length);
	fprintf(stderr, "    MIS             : %u\n", m_options.mis ? 1 : 0);
	fprintf(stderr, "    perturbations   :\n");
	fprintf(stderr, "      st            : %f\n", m_options.st_perturbations);
	fprintf(stderr, "      screen        : %f\n", m_options.screen_perturbations);
	fprintf(stderr, "      exp           : %f\n", m_options.exp_perturbations);
	fprintf(stderr, "      H             : %f\n", m_options.H_perturbations);
	fprintf(stderr, "    flags           : %x\n", m_options.flags);

	const uint32 n_chains = m_options.n_chains;

	// pre-alloc ray queues
	m_queues.alloc(n_pixels, n_light_paths, m_options.max_path_length);

	// build the set of shifts
	const uint32 n_dimensions = (m_options.max_path_length + 1) * 2 * 6;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	const uint32 n_tile_size = 256;
	m_sequence.setup(n_dimensions, n_tile_size);

	fprintf(stderr, "  allocating light vertex storage... started (%u paths, %u vertices)\n", n_light_paths, n_light_paths * m_options.max_path_length);
	m_light_vertices.alloc(n_light_paths, n_light_paths * m_options.max_path_length);
	fprintf(stderr, "  allocating light vertex storage... done\n");

	fprintf(stderr, "  allocating internal storage... started\n");
	// keep track of how much storage we'll need
	cugar::memory_arena arena;
	{
		MLTContext context;

		alloc( context, arena, n_pixels, n_chains );

		m_memory_pool.alloc(arena.size);
	}
	fprintf(stderr, "  allocating internal storage... done (%.1f MB)\n", float(arena.size) / float(1024*1024));

	// alloc a separate buffer for the connections cdf
	m_connections_cdf.alloc( n_pixels );
	m_path_pdf_backup.alloc( n_chains );

	m_n_lights = n_light_paths;
	m_n_init_paths = res.x * res.y;
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
void MLT::sample_seeds(MLTContext& context)
{
	cugar::device_vector<uint8> temp_storage;

	// compute the connections CDF
	cugar::inclusive_scan<cugar::device_tag>(
		m_n_connections,
		thrust::make_transform_iterator(context.connections_value, max_comp_functor()),
		m_connections_cdf.ptr(),
		thrust::plus<float>(),
		temp_storage);

	// zero out the stats
	cudaMemset(context.st_counters, 0x00, sizeof(uint32)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2));

	// resample n_chains of them
	::sample_seeds(m_n_connections, m_connections_cdf.ptr(), context.connections_index, m_options.n_chains, context.seeds, context.st_counters, m_options.max_path_length);

	// and sort them
	cugar::radix_sort<cugar::device_tag>(m_options.n_chains, context.seeds, temp_storage);

	m_image_brightness = m_connections_cdf[m_n_connections - 1] / float(m_n_init_paths);
}

// recover primary sample space coordinates of the sampled light and eye subpaths
//
void MLT::recover_seed_paths(MLTContext& context, RenderingContextView& renderer_view)
{
	::recover_seed_paths( m_options.n_chains, context.seeds, m_n_lights, context, renderer_view );
}

void MLT::render(const uint32 instance, RenderingContext& renderer)
{
	//fprintf(stderr, "render started (%u)\n", instance);
	renderer.multiply_frame(float(instance) / float(instance + 1));

	float presampling_time		= 0;
	float seed_resampling_time	= 0;
	float acceptance_rate_avg	= 0;

	cugar::Timer timer;

	//fprintf(stderr, "  trace eye subpaths... started\n");
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	const uint32 n_light_paths = renderer.get_mesh_lights().get_vpl_count();
	assert(n_light_paths == n_pixels);

	const uint32 chain_length = (m_options.spp * n_pixels) / m_options.n_chains;

	m_sequence.set_instance(instance);

	RenderingContextView renderer_view = renderer.view(instance);

	// initialize the memory arena
	cugar::memory_arena arena( m_memory_pool.ptr() );

	// setup a context
	MLTContext context( *this, renderer_view );
	context.n_light_paths		= n_light_paths;
	context.n_eye_paths			= n_pixels;
	context.n_chains			= m_options.n_chains;
	context.chain_length		= chain_length;
	alloc( context, arena, n_pixels, m_options.n_chains );

	uint32 mutation_pass = (instance % m_options.reseeding_freq);

	// check whether it's time to perform a bidirectional path tracing "seeding" pass,
	// to sample the starting points of our parallel chains (see the description
	// of the global seeding strategy in:
	//   <a href="https://arxiv.org/abs/1612.05395">Charted Metropolis Light Transport</a>,
	//   Jacopo Pantaleoni, ACM Transactions on Graphics, Volume 36 Issue 4, July 2017.
	if (mutation_pass == 0 || m_n_connections == 0)
	{
		timer.start();

		// reset the connections counter
		cudaMemset(context.connections_counter, 0x00, sizeof(uint32));

		// zero out the stats
		cudaMemset(context.st_norms, 0x00, sizeof(float)*(context.options.max_path_length + 2)*(context.options.max_path_length + 2));
	
		// perform a simple bidirectional path tracing pass
		{
			TiledLightSubpathPrimaryCoords light_primary_coords(context.sequence);

			PerPixelEyeSubpathPrimaryCoords eye_primary_coords(context.sequence, res.x, res.y);

			MLTPresamplingBPTConfig config(context);

			ConnectionsSink connections_sink;

			bpt::sample_paths(
				m_n_init_paths,
				m_n_init_paths,
				eye_primary_coords,
				light_primary_coords,
				connections_sink,
				context,
				config,
				renderer,
				renderer_view);
		}
		timer.stop();
		presampling_time = timer.seconds();

		timer.start();

		cudaMemcpy(&m_n_connections, context.connections_counter, sizeof(uint32), cudaMemcpyDeviceToHost);

		// exit if we didn't find any valid path
		if (m_n_connections == 0)
			return;

		// sample a set of 'n_chains seeds out of all the valid connections (i.e. entire paths) we found
		sample_seeds( context );

		// print out the chain stats
		if (instance == 0)
		{
			fprintf(stderr, "  image brightness: %f\n", m_image_brightness);
			fprintf(stderr, "  st chains\n");
			uint32 st_counters[1024];
			float  st_norms[1024];
			cudaMemcpy(st_counters,	context.st_counters,	sizeof(uint32) * (m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);
			cudaMemcpy(st_norms,	context.st_norms,		sizeof(float)  * (m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);
			for (uint32 s = 0; s < m_options.max_path_length; ++s)
			{
				for (uint32 t = 0; t < m_options.max_path_length + 2; ++t)
				{
					if (s + t <= m_options.max_path_length + 1)
						fprintf(stderr, "    [%u,%u] : %7u, %f\n", s, t, st_counters[s + t*(m_options.max_path_length + 2)], st_norms[s + t*(m_options.max_path_length + 2)] / float(m_n_init_paths));
				}
			}
		}

		// build a CDF on st_norms
		{
			float st_norms[1024];
			float st_norms_cdf[1024] = { 0 };

			cudaMemcpy(st_norms, context.st_norms, sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyDeviceToHost);

			for (uint32 k = 1; k <= m_options.max_path_length; ++k)
			{
				// consider all the (k+2) paths of length k
				for (uint32 s = 0; s < k + 2; ++s)
				{
					const uint32 t = k + 1 - s;

					const float norm = st_norms[s + t*(m_options.max_path_length + 2)] / float(m_n_init_paths);

					st_norms_cdf[k * (m_options.max_path_length + 2) + s] = norm;
				}

				// compute the cumulative sum
				for (uint32 i = 1; i < k + 2; ++i)
					st_norms_cdf[k * (m_options.max_path_length + 2) + i] +=
					st_norms_cdf[k * (m_options.max_path_length + 2) + i - 1];

				// and normalize it
				const float inv_sum = 1.0f / st_norms_cdf[k * (m_options.max_path_length + 2) + k + 1];
				for (uint32 i = 0; i < k + 2; ++i)
					st_norms_cdf[k * (m_options.max_path_length + 2) + i] *= inv_sum;

				//for (uint32 i = 0; i < k + 2; ++i)
				//	fprintf(stderr, "cdf[%u][%u] = %f\n", k, i, st_norms_cdf[k * (m_options.max_path_length + 2) + i]);
				//fgetc(stdin);
			}
			cudaMemcpy(context.st_norms_cdf, st_norms_cdf, sizeof(float)*(m_options.max_path_length + 2)*(m_options.max_path_length + 2), cudaMemcpyHostToDevice);
		}

		// reconstruct the actual paths corresponding to the sampled seed indices
		recover_seed_paths(context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("recover seed paths"));

		// initialize the initial path pdfs to zero
		cudaMemset(context.path_pdf, 0x00, sizeof(float)*m_options.n_chains);

		// initialize the rejection counters to zero
		cudaMemset(context.rejections, 0x00, sizeof(uint32)*m_options.n_chains);

		// initialize the initial path bsdfs to zero
		cudaMemset(context.f_vertices, 0x00, sizeof(float4)*m_options.n_chains * (m_options.max_path_length + 1));

		timer.stop();
		seed_resampling_time = timer.seconds();
	}

	// setup the normalization constant
	context.pdf_norm = m_image_brightness * float(res.x * res.y) / float(chain_length * m_options.n_chains);

	timer.start();

	const uint32 begin_chain_step = mutation_pass * chain_length;
	const uint32 end_chain_step   = begin_chain_step + chain_length;

	// loop through all the steps in the chains
	for (context.chain_step = begin_chain_step;
		 context.chain_step < end_chain_step;
		 context.chain_step++)
	{
		// disable mutations if this is the very first pass after a reseeding step
		context.enable_mutations = (context.chain_step > 0);

		// reset the acceptance rate stats
		cudaMemset(context.acceptance_rate_sum, 0x00, sizeof(float));

		// reset the checksum
		cudaMemset(context.checksum, 0x00, sizeof(float));

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

		// generate primary light vertices from the mesh light samples, including the sampling of a direction
		perturb_primary_light_vertices_mlt(context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("perturb primary light vertices mlt"));

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);

		// for each bounce: trace rays, process hits (store light vertices, generate sampling directions)
		for (context.in_bounce = 0;
			 context.in_bounce < m_options.max_path_length - 1;
			 context.in_bounce++)
		{
			uint32 in_queue_size;

			// read out the number of output rays produced by the previous pass
			cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// check whether there's still any work left
			if (in_queue_size == 0)
				break;

			// reset the output queue counters
			cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
			cugar::cuda::check_error("memset");

			// trace the rays generated at the previous bounce
			//
			{
				renderer.get_rt_context()->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
			}

			// process the light vertices at this bounce
			//
			perturb_secondary_light_vertices_mlt(in_queue_size, context, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("perturb light vertices"));

			// swap the input and output queues
			std::swap(context.in_queue, context.scatter_queue);
		}

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

		perturb_primary_rays_mlt(context,renderer_view);

		for (context.in_bounce = 0;
			 context.in_bounce < m_options.max_path_length;
			 context.in_bounce++)
		{
			uint32 in_queue_size;

			// read out the number of output rays produced by the previous pass
			cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// check whether there's still any work left
			if (in_queue_size == 0)
				break;

			// reset the output queue counters
			cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
			cugar::cuda::check_error("memset");

			// trace the rays generated at the previous bounce
			//
			{
				renderer.get_rt_context()->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
			}

			// perform lighting at this bounce
			//
			perturb_secondary_eye_vertices_mlt(in_queue_size, context, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("perturb eye vertices"));

			// swap the input and output queues
			std::swap(context.in_queue, context.scatter_queue);
		}

		// trace & accumulate occlusion queries
		{
			uint32 shadow_queue_size;
			cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// trace the rays
			//
			if (shadow_queue_size)
			{
				renderer.get_rt_context()->trace(shadow_queue_size, (Ray*)context.shadow_queue.rays, context.shadow_queue.hits);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("trace occlusion"));
			}

			// shade the results
			//
			if (shadow_queue_size)
			{
				solve_occlusion_mlt(shadow_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion mlt"));
			}
		}

		// keep track of the acceptance rate
		{
			float acceptance_rate_sum;
			cudaMemcpy(&acceptance_rate_sum, context.acceptance_rate_sum, sizeof(float), cudaMemcpyDeviceToHost);			

			acceptance_rate_avg += acceptance_rate_sum / m_options.n_chains;
			//fprintf(stderr, "ar[%u:%u]: %f\n", instance, context.chain_step, acceptance_rate_sum / m_options.n_chains);
		}
		// keep track of the checksum
		if (0)
		{
			float checksum;
			cudaMemcpy(&checksum, context.checksum, sizeof(float), cudaMemcpyDeviceToHost);			

			fprintf(stderr, "cs[%u:%u]: %f\n", instance, context.chain_step, checksum / m_options.n_chains);
		}
	}
			
	timer.stop();
	const float mlt_time = timer.seconds();

	fprintf(stderr, "\r  %.1fms (presampling: %.1fms, seeding: %.1fms, mlt: %.1fms, AR: %f)            ",
		(presampling_time + seed_resampling_time + mlt_time) * 1000.0f,
		presampling_time * 1000.0f,
		seed_resampling_time * 1000.0f,
		mlt_time * 1000.0f,
		acceptance_rate_avg / chain_length);
}

// renderer factory method
//
RendererInterface* MLT::factory() { return new MLT(); }
