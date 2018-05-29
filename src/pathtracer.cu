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

#include <pathtracer.h>
#include <renderer.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/cuda/warp_atomics.h>
#include <cugar/basic/memory_arena.h>
#include <bsdf.h>
#include <edf.h>
#include <mis_utils.h>
#include <bpt_utils.h>
#include <eaw.h>
#include <vector>

#define SHIFT_RES	256u

#define SHADE_HITS_BLOCKSIZE	64
#define SHADE_HITS_CTA_BLOCKS	16

#define DEBUG_PIXEL	(187 + 826 * 1600)

#define MIS_HEURISTIC	POWER_HEURISTIC

namespace {

	union PixelInfo
	{
		FERMAT_HOST_DEVICE PixelInfo() {}
		FERMAT_HOST_DEVICE PixelInfo(const uint32 _packed) : packed(_packed) {}
		FERMAT_HOST_DEVICE PixelInfo(const uint32 _pixel, const uint32 _comp) : pixel(_pixel), comp(_comp) {}

		uint32	packed;
		struct
		{
			uint32 pixel : 28;
			uint32 comp  : 4;
		};
	};

	struct PTRayQueue
	{
		Ray*		  rays;
		Hit*		  hits;
		float4*		  weights;		// path weight
		float4*		  weights_d;	// diffuse path weight
		float4*		  weights_g;	// glossy path weight
		uint32*		  pixels;
		uint32*		  size;

		FERMAT_DEVICE
		void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight)
		{
			const uint32 slot = cugar::cuda::warp_increment(size);

			rays[slot]		= ray;
			weights[slot]	= weight;
			pixels[slot]	= pixel.packed;
		}

		FERMAT_DEVICE
		void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float4 weight_d, const float4 weight_g)
		{
			const uint32 slot = cugar::cuda::warp_increment(size);

			rays[slot]		= ray;
			weights[slot]	= weight;

			weights_d[slot]	= weight_d;
			weights_g[slot]	= weight_g;

			pixels[slot]	= pixel.packed;
		}
	};

	struct PathTracingContext
	{
		PTOptions			options;

		uint32				in_bounce;

		TiledSequenceView	sequence;

		PTRayQueue			in_queue;
		PTRayQueue			shadow_queue;
		PTRayQueue			scatter_queue;
	};

	//------------------------------------------------------------------------------
	__global__ void generate_primary_rays_kernel(PathTracingContext context, RendererView renderer, cugar::Vector3f U, cugar::Vector3f V, cugar::Vector3f W)
	{
		int pixel_x = threadIdx.x + blockIdx.x*blockDim.x;
		int pixel_y = threadIdx.y + blockIdx.y*blockDim.y;
		if (pixel_x >= renderer.res_x || pixel_y >= renderer.res_y)
			return;

		int idx = pixel_x + pixel_y*renderer.res_x;

		// use an optimized sampling pattern to rotate a Halton sequence
		const cugar::Vector2f uv(
			context.sequence.sample_2d(pixel_x, pixel_y, 0),
			context.sequence.sample_2d(pixel_x, pixel_y, 1));

		const float2 d = make_float2(
			(pixel_x + uv.x) / float(renderer.res_x),
			(pixel_y + uv.y) / float(renderer.res_y)) * 2.f - 1.f;

		// write the pixel index
		context.in_queue.pixels[idx] = idx;

		float3 ray_origin	 = renderer.camera.eye;
		float3 ray_direction = d.x*U + d.y*V + W;

		reinterpret_cast<float4*>(context.in_queue.rays)[2 * idx + 0] = make_float4(ray_origin.x, ray_origin.y, ray_origin.z, 0.0f); // origin, tmin
		reinterpret_cast<float4*>(context.in_queue.rays)[2 * idx + 1] = make_float4(ray_direction.x, ray_direction.y, ray_direction.z, 1e34f); // dir, tmax

		// write the filter weight
		context.in_queue.weights[idx] = cugar::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

		if (idx == 0)
			*context.in_queue.size = renderer.res_x * renderer.res_y;
	}

	//------------------------------------------------------------------------------

	void generate_primary_rays(PathTracingContext context, const RendererView renderer)
	{
		cugar::Vector3f U, V, W;
		camera_frame(renderer.camera, renderer.aspect, U, V, W);

		dim3 blockSize(32, 16);
		dim3 gridSize(cugar::divide_ri(renderer.res_x, blockSize.x), cugar::divide_ri(renderer.res_y, blockSize.y));
		generate_primary_rays_kernel << < gridSize, blockSize >> > (context, renderer, U, V, W);
	}
	//------------------------------------------------------------------------------



	template <uint32 NUM_WARPS>
	__global__
	__launch_bounds__(SHADE_HITS_BLOCKSIZE, SHADE_HITS_CTA_BLOCKS)
	void shade_hits_kernel(const uint32 in_queue_size, PathTracingContext context, RendererView renderer, const float frame_weight, const bool do_nee, const bool do_accumulate_emissive, const bool do_scatter)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.in_queue.size
		{
			const PixelInfo		  pixel_info	= context.in_queue.pixels[thread_id];
			const Ray			  ray			= context.in_queue.rays[thread_id];
			const Hit			  hit			= context.in_queue.hits[thread_id];
			const cugar::Vector4f w				= context.in_queue.weights[thread_id];
			const float           p_prev		= w.w;

			const uint32 pixel_x = pixel_info.pixel % renderer.res_x;
			const uint32 pixel_y = pixel_info.pixel / renderer.res_x;

			// initialize our shifted sampling sequence
			float samples[6];
			for (uint32 i = 0; i < 6; ++i)
				samples[i] = context.sequence.sample_2d(pixel_x, pixel_y, (context.in_bounce + 1) * 6 + i);

			if (hit.t > 0.0f && hit.triId >= 0)
			{
				EyeVertex ev;
				ev.setup(ray, hit, w.xyz(), cugar::Vector4f(0.0f), context.in_bounce, renderer);

				// write out gbuffer information
				if (context.in_bounce == 0)
				{
					renderer.fb.gbuffer.geo(pixel_info.pixel) = GBufferView::pack_geometry(ev.geom.position, ev.geom.normal_s);
					renderer.fb.gbuffer.uv(pixel_info.pixel)  = make_float4(hit.u, hit.v, ev.geom.texture_coords.x, ev.geom.texture_coords.y);
					renderer.fb.gbuffer.tri(pixel_info.pixel) = hit.triId;
				}

				cugar::Vector3f in = -cugar::normalize(cugar::Vector3f(ray.dir));

				// perform next-event estimation to compute direct lighting
				if (do_nee)
				{
					// fetch the sampling dimensions
					const float z[3] = { samples[0], samples[1], samples[2] }; // use dimensions 0,1,2

					VertexGeometryId light_vertex;
					VertexGeometry   light_vertex_geom;
					float			 light_pdf;
					Edf				 light_edf;

					// sample the light source surface
					renderer.mesh_vpls.sample(z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);
					//renderer.mesh_light.sample(z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);

					// join the light sample with the current vertex
					cugar::Vector3f out = (light_vertex_geom.position - ev.geom.position);
						
					const float d2 = fmaxf(1.0e-8f, cugar::square_length(out));

					// normalize the outgoing direction
					out *= rsqrtf(d2);

					cugar::Vector3f f_s_comp[Bsdf::kNumComponents];
					float			p_s_comp[Bsdf::kNumComponents];

					ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s_comp, p_s_comp, cugar::kProjectedSolidAngle);

					cugar::Vector3f f_s(0.0f);
					float			p_s(0.0f);

					for (uint32 i = 0; i < Bsdf::kNumComponents; ++i)
					{
						f_s += f_s_comp[i];
						p_s += p_s_comp[i];
					}

					// evaluate the light's EDF and the surface BSDF
					const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;

					// evaluate the geometric term
					const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

					// TODO: perform MIS with the possibility of directly hitting the light source
					const float p1 = light_pdf;
					const float p2 = p_s * G;
					const float mis_w =
						(context.in_bounce == 0 && context.options.direct_lighting_bsdf) ||
						(context.in_bounce >  0 && context.options.indirect_lighting_bsdf) ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

					// calculate the cumulative sample weight, equal to f_L * f_s * G / p
					const cugar::Vector3f out_w		= w.xyz() * f_L * f_s * G * mis_w;
					const cugar::Vector3f out_w_d	= (context.in_bounce == 0 ? f_s_comp[Bsdf::kDiffuseReflectionIndex] + f_s_comp[Bsdf::kDiffuseTransmissionIndex] : f_s) * w.xyz() * f_L *  G * mis_w;
					const cugar::Vector3f out_w_g	= (context.in_bounce == 0 ? f_s_comp[Bsdf::kGlossyReflectionIndex] + f_s_comp[Bsdf::kGlossyTransmissionIndex]  : f_s) * w.xyz() * f_L *  G * mis_w;

					if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
					{
						// enqueue the output ray
						Ray out_ray;
						out_ray.origin	= ev.geom.position - ray.dir * 1.0e-4f; // shift back in space along the viewing direction
						out_ray.dir		= (light_vertex_geom.position - out_ray.origin); //out;
						out_ray.tmin	= 0.0f;
						out_ray.tmax	= 0.9999f; //d * 0.9999f;

						const PixelInfo out_pixel = pixel_info;

						context.shadow_queue.warp_append(
							out_pixel,
							out_ray,
							cugar::Vector4f(out_w, 0.0f),
							cugar::Vector4f(out_w_d, 0.0f),
							cugar::Vector4f(out_w_g, 0.0f) );
					}
				}

				// accumulate the emissive component along the incoming direction
				if (do_accumulate_emissive)
				{
					VertexGeometry	light_vertex_geom = ev.geom;
					float			light_pdf;
					Edf				light_edf;

					renderer.mesh_vpls.map(hit.triId, cugar::Vector2f(hit.u, hit.v), light_vertex_geom, &light_pdf, &light_edf);
					//renderer.mesh_light.map(hit.triId, cugar::Vector2f(hit.u, hit.v), light_vertex_geom, &light_pdf, &light_edf);

					// evaluate the edf's output along the incoming direction
					const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, ev.in);

					const float d2 = fmaxf(1.0e-10f, hit.t * hit.t);

					// compute the MIS weight with next event estimation at the previous vertex
					const float G_partial = fabsf(cugar::dot(ev.in, light_vertex_geom.normal_s)) / d2; // NOTE: G_partial doesn't include the dot product between 'in and the normal at the previous vertex

					const float p1 = G_partial * p_prev;											// NOTE: p_prev is the solid angle probability of sampling the BSDF at the previous vertex, i.e. p_proj * dot(in,normal)
					const float p2 = light_pdf;
					const float mis_w =
						(context.in_bounce == 1 && context.options.direct_lighting_nee) ||
						(context.in_bounce >  1 && context.options.indirect_lighting_nee) ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

					// and accumulate the weighted contribution
					const cugar::Vector3f out_w	= w.xyz() * f_L * mis_w;

					// and accumulate the weighted contribution
					if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
					{
						// accumulate to the image
						add_in<false>(renderer.fb(FBufferDesc::COMPOSITED_C), pixel_info.pixel, out_w, frame_weight);

						// accumulate the per-component value to the proper output channel
						if (context.in_bounce == 0)
							add_in<false>(renderer.fb(FBufferDesc::DIRECT_C), pixel_info.pixel, out_w, frame_weight);
						else
						{
							if (pixel_info.comp & Bsdf::kDiffuseMask)	add_in<true>(renderer.fb(FBufferDesc::DIFFUSE_C),  pixel_info.pixel, out_w, frame_weight);
							if (pixel_info.comp & Bsdf::kGlossyMask)	add_in<true>(renderer.fb(FBufferDesc::SPECULAR_C), pixel_info.pixel, out_w, frame_weight);
						}
					}
				}

				// trace a bounce ray
				if (do_scatter)
				{
					// fetch the sampling dimensions
					const float z[3] = { samples[3], samples[4], samples[5] }; // use dimensions 3,4,5

					// sample a scattering event
					cugar::Vector3f		out(0.0f);
					cugar::Vector3f		g(0.0f);
					float				p(0.0f);
					float				p_proj(0.0f);
					Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

					scatter(ev, z, out_comp, out, p, p_proj, g, true, false);

					cugar::Vector3f out_w = g * w.xyz();

					if (context.in_bounce == 0)
					{
						renderer.fb(FBufferDesc::DIFFUSE_A,  pixel_info.pixel) += cugar::Vector4f(ev.material.diffuse)  * frame_weight;
						renderer.fb(FBufferDesc::SPECULAR_A, pixel_info.pixel) += (cugar::Vector4f(ev.material.specular) + cugar::Vector4f(1.0f))*0.5f * frame_weight;
					}

					if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
					{
						// enqueue the output ray
						Ray out_ray;
						out_ray.origin	= ev.geom.position;
						out_ray.dir		= out;
						out_ray.tmin	= 1.0e-4f;
						out_ray.tmax	= 1.0e8f;

						const float out_p = p;

						const PixelInfo out_pixel = context.in_bounce ?
							pixel_info :												// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, out_comp);						// otherwise (i.e. this is the first bounce) choose the output channel for the rest of the path

						context.scatter_queue.warp_append(
							out_pixel,
							out_ray,
							cugar::Vector4f(out_w, out_p) );
					}
				}
			}
			else
			{
				// hit the environment - perform sky lighting
			}
		}
	}

	void shade_hits(const uint32 in_queue_size, PathTracingContext context, RendererView renderer)
	{
		const uint32 blockSize(SHADE_HITS_BLOCKSIZE);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));

		// decide whether to perform next-event estimation
		const bool do_nee =
			((context.in_bounce + 2 <= context.options.max_path_length) &&
			((context.in_bounce == 0 && context.options.direct_lighting_nee) ||
			 (context.in_bounce >  0 && context.options.indirect_lighting_nee)));

		// decide whether to evaluate and accumulate emissive surfaces
		const bool do_accumulate_emissive =
			((context.in_bounce == 0 && context.options.visible_lights) ||
			 (context.in_bounce == 1 && context.options.direct_lighting_bsdf) ||
			 (context.in_bounce >  1 && context.options.indirect_lighting_bsdf));

		// compute the number of path vertices we want to generate from the eye
		const uint32 max_path_vertices = context.options.max_path_length +
			((context.options.max_path_length == 2 && context.options.direct_lighting_bsdf) ||
			 (context.options.max_path_length >  2 && context.options.indirect_lighting_bsdf) ? 1 : 0);

		// decide whether to perform scattering
		const bool do_scatter = (context.in_bounce + 2 < max_path_vertices);

		shade_hits_kernel<blockSize / 32> << < gridSize, blockSize >> > (in_queue_size, context, renderer, 1.0f / float(renderer.instance + 1), do_nee, do_accumulate_emissive, do_scatter);
	}

	__global__
	void solve_occlusion_kernel(const uint32 in_queue_size, PathTracingContext context, RendererView renderer, const float frame_weight)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.shadow_queue.size
		{
			const PixelInfo		  pixel_info	= context.shadow_queue.pixels[thread_id];
			const Hit			  hit			= context.shadow_queue.hits[thread_id];
			const cugar::Vector4f w				= context.shadow_queue.weights[thread_id];
			const cugar::Vector4f w_d			= context.shadow_queue.weights_d[thread_id];
			const cugar::Vector4f w_g			= context.shadow_queue.weights_g[thread_id];

			// TODO: break this up in separate diffuse and specular components
			if (hit.t < 0.0f)
			{
				add_in<false>( renderer.fb(FBufferDesc::COMPOSITED_C), pixel_info.pixel, w.xyz(), frame_weight );

				if (context.in_bounce == 0)
				{
					// accumulate the per-component values to the respective output channels
					add_in<true>( renderer.fb(FBufferDesc::DIFFUSE_C),  pixel_info.pixel, w_d.xyz(), frame_weight );
					add_in<true>( renderer.fb(FBufferDesc::SPECULAR_C), pixel_info.pixel, w_g.xyz(), frame_weight );
				}
				else
				{
					// accumulate the per-component value to the proper output channel
					if (pixel_info.comp & Bsdf::kDiffuseMask)	add_in<true>( renderer.fb(FBufferDesc::DIFFUSE_C),  pixel_info.pixel, w_d.xyz(), frame_weight );
					if (pixel_info.comp & Bsdf::kGlossyMask)	add_in<true>( renderer.fb(FBufferDesc::SPECULAR_C), pixel_info.pixel, w_g.xyz(), frame_weight );;
				}
			}
		}
	}

	void solve_occlusion(const uint32 in_queue_size, PathTracingContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		solve_occlusion_kernel << < gridSize, blockSize >> > (in_queue_size, context, renderer, 1.0f / float(renderer.instance + 1) );
	}

	void alloc_queues(
		PTOptions				options,
		const uint32			n_pixels,
		PTRayQueue&				input_queue,
		PTRayQueue&				scatter_queue,
		PTRayQueue&				shadow_queue,
		cugar::memory_arena&	arena)
	{
		input_queue.rays		= arena.alloc<Ray>(n_pixels);
		input_queue.hits		= arena.alloc<Hit>(n_pixels);
		input_queue.weights		= arena.alloc<float4>(n_pixels);
		input_queue.weights_d	= NULL;
		input_queue.weights_g	= NULL;
		input_queue.pixels		= arena.alloc<uint32>(n_pixels);
		input_queue.size		= arena.alloc<uint32>(1);

		scatter_queue.rays		= arena.alloc<Ray>(n_pixels);
		scatter_queue.hits		= arena.alloc<Hit>(n_pixels);
		scatter_queue.weights	= arena.alloc<float4>(n_pixels);
		scatter_queue.weights_d	= NULL;
		scatter_queue.weights_g	= NULL;
		scatter_queue.pixels	= arena.alloc<uint32>(n_pixels);
		scatter_queue.size		= arena.alloc<uint32>(1);

		shadow_queue.rays		= arena.alloc<Ray>(n_pixels);
		shadow_queue.hits		= arena.alloc<Hit>(n_pixels);
		shadow_queue.weights	= arena.alloc<float4>(n_pixels);
		shadow_queue.weights_d	= arena.alloc<float4>(n_pixels);
		shadow_queue.weights_g	= arena.alloc<float4>(n_pixels);
		shadow_queue.pixels		= arena.alloc<uint32>(n_pixels);
		shadow_queue.size		= arena.alloc<uint32>(1);
	}

} // anonymous namespace

PathTracer::PathTracer() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void PathTracer::init(int argc, char** argv, Renderer& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	// pre-alloc queue storage
	{
		// determine how much storage we will need
		cugar::memory_arena arena;

		PTRayQueue	input_queue;
		PTRayQueue	scatter_queue;
		PTRayQueue	shadow_queue;

		alloc_queues(
			m_options,
			n_pixels,
			input_queue,
			scatter_queue,
			shadow_queue,
			arena );

		fprintf(stderr, "  allocating queue storage: %.1f MB\n", float(arena.size) / (1024*1024));
		m_memory_pool.alloc(arena.size);
	}

	// build the set of shifts
	const uint32 n_dimensions = 6 * (m_options.max_path_length + 1);
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);
	m_sequence.setup(n_dimensions, SHIFT_RES);

	const uint32 n_light_paths = n_pixels;

	fprintf(stderr, "  creatign mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.m_mesh_lights.init( n_light_paths, renderer.m_mesh.view(), renderer.m_mesh_d.view(), renderer.m_texture_views_h.ptr(), renderer.m_texture_views_d.ptr() );

	fprintf(stderr, "  creatign mesh lights... done\n");
}

void PathTracer::render(const uint32 instance, Renderer& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.rescale_frame( instance );

	//fprintf(stderr, "render started (%u)\n", instance);
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	cugar::memory_arena arena( m_memory_pool.ptr() );

	PTRayQueue queue1;
	PTRayQueue queue2;
	PTRayQueue shadow_queue;

	alloc_queues(
		m_options,
		n_pixels,
		queue1,
		queue2,
		shadow_queue,
		arena );

	cugar::Timer timer;
	timer.start();

	float path_rt_time = 0.0f;
	float shadow_rt_time = 0.0f;
	float path_shade_time = 0.0f;
	float shadow_shade_time = 0.0f;

	// fetch a view of the renderer
	RendererView renderer_view = renderer.view(instance);

	// setup the samples for this frame
	m_sequence.set_instance(instance);

	PathTracingContext context;
	context.options			= m_options;
	context.in_bounce		= 0;
	context.in_queue		= queue1;
	context.scatter_queue	= queue2;
	context.shadow_queue	= shadow_queue;
	context.sequence		= m_sequence.view();

	generate_primary_rays(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	for (context.in_bounce = 0;
		context.in_bounce < context.options.max_path_length;
		context.in_bounce++)
	{
		uint32 in_queue_size;

		// fetch the amount of tasks in the queue
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		// trace the rays generated at the previous bounce
		//
		{
			FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> trace_timer(&path_rt_time));

			optix::prime::Query query = renderer.m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
			query->setRays(in_queue_size, Ray::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.rays);
			query->setHits(in_queue_size, Hit::format, RTP_BUFFER_TYPE_CUDA_LINEAR, context.in_queue.hits);
			query->execute(0);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
		}

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// perform lighting at this bounce
		//
		{
			FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> shade_timer(&path_shade_time));

			shade_hits(in_queue_size, context, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("shade hits"));
		}

		// trace & accumulate occlusion queries
		{
			uint32 shadow_queue_size;
			cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// trace the rays
			//
			if (shadow_queue_size)
			{
				FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> trace_timer(&shadow_rt_time));

				optix::prime::Query query = renderer.m_model->createQuery(RTP_QUERY_TYPE_CLOSEST);
				query->setRays(shadow_queue_size, Ray::format, RTP_BUFFER_TYPE_CUDA_LINEAR, shadow_queue.rays);
				query->setHits(shadow_queue_size, Hit::format, RTP_BUFFER_TYPE_CUDA_LINEAR, shadow_queue.hits);
				query->execute(0);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("trace occlusion"));
			}

			// shade the results
			//
			if (shadow_queue_size)
			{
				FERMAT_CUDA_TIME(cugar::cuda::ScopedTimer<float> shade_timer(&shadow_shade_time));

				solve_occlusion(shadow_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));
			}
		}

		std::swap(context.in_queue, context.scatter_queue);
	}

	timer.stop();
	const float time = timer.seconds();
	// clear the global timer at instance zero
	if (instance == 0)
		m_time = time;
	else
		m_time += time;

	fprintf(stderr, "\r  %.1fs (%.1fms = rt[%.1fms + %.1fms] + shade[%.1fms + %.1fms])        ",
		m_time,
		time * 1000.0f,
		path_rt_time * 1000.0f,
		shadow_rt_time * 1000.0f,
		path_shade_time * 1000.0f,
		shadow_shade_time * 1000.0f);

	renderer.update_variances( instance );
}
