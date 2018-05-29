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

#include <rpt.h>
#include <renderer.h>
#include <optix_prime/optix_primepp.h>
#include <mesh/MeshStorage.h>
#include <cugar/basic/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/cuda/warp_atomics.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_utils.h>
#include <mis_utils.h>
#include <tiled_sampling.h>
#include <vector>

#define SHIFT_RES	64u

#define MAX_DEPTH				5
#define DIRECT_LIGHTING_NEE		1
#define DIRECT_LIGHTING_BSDF	1

#define MINIMUM_G				1.0e-6f

#define MIS_HEURISTIC	POWER_HEURISTIC
#define REUSE_HEURISTIC	BALANCE_HEURISTIC

#define TILE_X					4u
#define TILE_Y					4u
#define MACRO_TILE_X			TILE_Y
#define MACRO_TILE_Y			TILE_X
#define MACRO_TILE_SIZE			(TILE_X * TILE_Y)

#define REUSE_SHADOW_SAMPLES	4u

namespace {

	union PixelInfo
	{
		FERMAT_HOST_DEVICE PixelInfo() {}
		FERMAT_HOST_DEVICE PixelInfo(const uint32 _packed) : packed(_packed) {}
		FERMAT_HOST_DEVICE PixelInfo(const uint32 _pixel, const uint32 _channel) : pixel(_pixel), channel(_channel) {}

		uint32	packed;
		struct
		{
			uint32 pixel : 28;
			uint32 channel : 4;
		};
	};

	struct RPTRayQueue
	{
		Ray*		rays;
		Hit*		hits;
		float4*		weights;
		float4*		weights2;
		float*		probs;
		uint32*		pixels;
		uint32*		size;
		uint32		max_size;

		FERMAT_DEVICE
		void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float4 weight2, const float p)
		{
			const uint32 slot = cugar::cuda::warp_increment(size);
			FERMAT_ASSERT(slot < max_size);

			rays[slot]		= ray;
			weights[slot]	= weight;
			weights2[slot]	= weight2;
			probs[slot]		= p;
			pixels[slot]	= pixel.packed;
		}
	};

	struct RPTContext
	{
		RPTOptions			options;

		uint32				in_bounce;

		TiledSequenceView	sequence;

		RPTRayQueue			in_queue;
		RPTRayQueue			shadow_queue;
		RPTRayQueue			scatter_queue;

		RPTVPLView			vpls;

		cugar::LFSRGeneratorMatrix  generator;
	};

	//------------------------------------------------------------------------------
	__global__ void generate_primary_rays_kernel(RPTContext context, RendererView renderer, float3 U, float3 V, float3 W)
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
		context.in_queue.weights[idx]  = cugar::Vector4f(1.0f, 1.0f, 1.0f, 0.0f);
		context.in_queue.weights2[idx] = cugar::Vector4f(1.0f, 1.0f, 1.0f, 0.0f);

		if (idx == 0)
			*context.in_queue.size = renderer.res_x * renderer.res_y;
	}

	//------------------------------------------------------------------------------

	void generate_primary_rays(RPTContext context, const RendererView renderer)
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
	void shade_hits_kernel(const uint32 in_queue_size, RPTContext context, RendererView renderer, const bool do_nee, const bool do_accumulate_emissive, const bool do_scatter)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		const float frame_weight = 1.0f / float(renderer.instance + 1);

		if (thread_id < in_queue_size) // *context.in_queue.size
		{
			const PixelInfo		  pixel_info	= context.in_queue.pixels[thread_id];
			const Ray			  ray			= context.in_queue.rays[thread_id];
			const Hit			  hit			= context.in_queue.hits[thread_id];
			const float           p_prev		= context.in_queue.probs[thread_id];
			const cugar::Vector4f w				= context.in_queue.weights[thread_id];
			const cugar::Vector4f w2			= context.in_queue.weights2[thread_id];

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

				if (context.in_bounce == 1)
				{
					// store the VPL geometry
					context.vpls.pos[pixel_info.pixel]		= cugar::Vector4f(ev.geom.position, cugar::binary_cast<float>(pack_direction(ev.geom.normal_s)));
					context.vpls.gbuffer[pixel_info.pixel]	= pack_bsdf(ev.material);
					context.vpls.ebuffer[pixel_info.pixel]	= pack_edf(ev.material);
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
					//renderer.mesh_vpls.sample(z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);
					renderer.mesh_light.sample(z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);

					// join the light sample with the current vertex
					cugar::Vector3f out = (light_vertex_geom.position - ev.geom.position);
						
					const float d2 = fmaxf(MINIMUM_G, cugar::square_length(out));
					const float d = sqrtf(d2);

					// normalize the outgoing direction
					out /= d;

					// check which components we have to sample
					uint32 component_mask = uint32(Bsdf::kAllComponents);
					{
						// disable diffuse scattering if not allowed
						if (context.options.diffuse_scattering == false)
							component_mask &= ~uint32(Bsdf::kDiffuseMask);

						// disable glossy scattering if:
						// 1. we have sampled a diffuse reflection and indirect_glossy == false, OR
						// 2. indirect glossy scattering is disabled
						if (context.options.glossy_scattering == false)
							component_mask &= ~uint32(Bsdf::kGlossyMask);
					}

					// evaluate the light's EDF and the surface BSDF
					const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;
					const cugar::Vector3f f_s = ev.bsdf.f(ev.geom, ev.in, out, Bsdf::ComponentType(component_mask));
					// TODO: for the first bounce (i.e. direct-lighting), break this up in separate diffuse and specular components

					// evaluate the geometric term
					const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

					// TODO: perform MIS with the possibility of directly hitting the light source
					const float p1 = light_pdf;
					//const float p2 = bsdf.p(geom, in, out, cugar::kProjectedSolidAngle) * G;
					const float p2 = (ev.bsdf.p(ev.geom, ev.in, out, cugar::kSolidAngle) * fabsf(cugar::dot(out, light_vertex_geom.normal_s))) / d2;
					const float mis_w =
						(context.in_bounce == 0 && context.options.direct_lighting_bsdf) ||
						(context.in_bounce >  0 && context.options.indirect_lighting_bsdf) ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

					// calculate the cumulative sample weight, equal to f_L * f_s * G / p
					const cugar::Vector4f out_w  = cugar::Vector4f(w.xyz() * f_L * f_s * G * mis_w, w.w);
					const cugar::Vector4f out_w2 = context.in_bounce == 1 ?
						cugar::Vector4f(f_L * G * mis_w, w2.w) :
						cugar::Vector4f(w2.xyz() * f_L * f_s * G * mis_w, p_prev * ev.prev_G_prime);

					if (cugar::max_comp(out_w.xyz()) > 0.0f && cugar::is_finite(out_w.xyz()))
					{
						// enqueue the output ray
						Ray out_ray;
						out_ray.origin	= ev.geom.position - ray.dir * 1.0e-4f; // shift back in space along the viewing direction
						out_ray.dir		= (light_vertex_geom.position - out_ray.origin); //out;
						out_ray.tmin	= 0.0f;
						out_ray.tmax	= 0.9999f;

						const PixelInfo out_pixel = context.in_bounce ?
							pixel_info :										// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, FBufferDesc::COMPOSITED_C);	// otherwise (i.e. this is the first bounce) choose the direct-lighting output channel

						if (context.in_bounce == 1)
							context.vpls.in_dir2[pixel_info.pixel] = pack_direction(out);

						context.shadow_queue.warp_append(out_pixel, out_ray, out_w, out_w2, 1.0f);
					}
				}

				// accumulate the emissive component along the incoming direction
				if (do_accumulate_emissive)
				{
					VertexGeometry	light_vertex_geom;
					float			light_pdf;
					Edf				light_edf;

					//renderer.mesh_vpls.map(hit.triId, cugar::Vector2f(hit.u, hit.v), &light_vertex_geom, &light_pdf, &light_edf);
					renderer.mesh_light.map(hit.triId, cugar::Vector2f(hit.u, hit.v), &light_vertex_geom, &light_pdf, &light_edf);

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
					const cugar::Vector4f out_w  = cugar::Vector4f(w.xyz() * f_L * mis_w, w.w);
					const cugar::Vector4f out_w2 = cugar::Vector4f(w2.xyz() * f_L * mis_w, w2.w);

					if (cugar::is_finite(out_w.xyz()))
					{
						if (context.in_bounce > 1)
						{
							// accumulate this contribution to the VPL's incoming radiance
							context.vpls.in_alpha[pixel_info.pixel] += out_w2;
						}
						else
							renderer.fb(context.in_bounce == 0 ? FBufferDesc::COMPOSITED_C : pixel_info.channel, pixel_info.pixel) += out_w * frame_weight;
					}
				}

				// accumulate the average albedo of visible surfaces
				if (context.in_bounce == 0)
				{
					const cugar::Vector3f specular = (cugar::Vector3f(ev.material.specular.x, ev.material.specular.y, ev.material.specular.z) + cugar::Vector3f(1.0f))*0.5f;

					renderer.fb(FBufferDesc::DIFFUSE_A,  pixel_info.pixel) += cugar::Vector4f(ev.material.diffuse) * frame_weight;
					renderer.fb(FBufferDesc::SPECULAR_A, pixel_info.pixel) += cugar::Vector4f(specular, ev.material.roughness) * frame_weight;
				}

				// trace a bounce ray
				if (do_scatter)
				{
					// fetch the sampling dimensions
					const float z[3] = { samples[3], samples[4], samples[5] }; // use dimensions 3,4,5

					// sample a scattering event
					cugar::Vector3f		out(0.0f);
					cugar::Vector3f		out_g(0.0f);
					float				p(0.0f);
					float				p_proj(0.0f);
					Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

					// check which components we have to sample
					uint32 component_mask = uint32(Bsdf::kAllComponents);
					{
						// disable diffuse scattering if not allowed
						if (context.options.diffuse_scattering == false)
							component_mask &= ~uint32(Bsdf::kDiffuseMask);

						// disable glossy scattering if:
						// 1. we have sampled a diffuse reflection and indirect_glossy == false, OR
						// 2. indirect glossy scattering is disabled
						if (context.options.glossy_scattering == false)
							component_mask &= ~uint32(Bsdf::kGlossyMask);
					}
						
					scatter(ev, z, out_comp, out, p, p_proj, out_g, true, false, false, Bsdf::ComponentType(component_mask));

					const cugar::Vector3f out_w  = out_g * ev.alpha;
					const cugar::Vector3f out_w2 = ev.depth == 1 ? cugar::Vector3f(1.0f) : out_g * w2.xyz();

					if (context.in_bounce == 0)
					{
						// store the weight and probability for this scattering event
						context.vpls.weight2[pixel_info.pixel] = cugar::Vector4f(out_w, p_proj);
					}
					if (context.in_bounce == 1)
					{
						// store a VPL's direction and initialize the incoming radiance to zero
						context.vpls.in_dir[pixel_info.pixel]	= pack_direction(out);
						context.vpls.in_alpha[pixel_info.pixel] = cugar::Vector4f(0.0f);

						// store the cumulative weight for this VPL, together with its sampling probability
						context.vpls.weight[pixel_info.pixel] = cugar::Vector4f(out_w, p_prev * ev.prev_G_prime * p_proj);
					}

					// TODO: chose the channel based on our new BSDF factorization heuristic

					if (cugar::max_comp(out_w) > 0.0f)
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
							PixelInfo(pixel_info.pixel, channel_selector(out_comp));	// otherwise (i.e. this is the first bounce) choose the output channel for the rest of the path

						context.scatter_queue.warp_append(
							out_pixel,
							out_ray,
							cugar::Vector4f(out_w, w.w),
							cugar::Vector4f(out_w2, out_p),
							out_p );
					}
				}
			}
			else
			{
				// hit the environment - perform sky lighting
			}
		}
	}

	void shade_hits(const uint32 in_queue_size, RPTContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));

		// decide whether to perform next-event estimation
		const bool do_nee =
			((context.in_bounce + 2 <= context.options.max_path_length) &&
			((context.in_bounce == 0 && context.options.direct_lighting_nee && context.options.direct_lighting) ||
			 (context.in_bounce >  0 && context.options.indirect_lighting_nee)));

		// decide whether to evaluate and accumulate emissive surfaces
		const bool do_accumulate_emissive =
			((context.in_bounce == 0 && context.options.visible_lights) ||
			 (context.in_bounce == 1 && context.options.direct_lighting_bsdf && context.options.direct_lighting) ||
			 (context.in_bounce >  1 && context.options.indirect_lighting_bsdf));

		// compute the number of path vertices we want to generate from the eye
		const uint32 max_path_vertices = context.options.max_path_length +
			((context.options.max_path_length == 2 && context.options.direct_lighting_bsdf) ||
			 (context.options.max_path_length >  2 && context.options.indirect_lighting_bsdf) ? 1 : 0);

		// decide whether to perform scattering
		const bool do_scatter = (context.in_bounce + 2 < max_path_vertices);

		shade_hits_kernel<blockSize / 32> << < gridSize, blockSize >> > (in_queue_size, context, renderer, do_nee, do_accumulate_emissive, do_scatter);
	}

	// Bekaert-style tiled reuse
	// first pass: for each pixel j in each tile, compute w_j = p(j,j) / \sum_k p(k,j) = p(j,j) * w_j*, w_j* = 1 / \sum_k p(k,j)
	//      => i.e. each pixel stores its own w_j
	// second pass: for each pixel j in each tile, compute sum_k w_k(y_k) * L(x_j,y_k) / p(k,k) = sum_k w_k* L(x_j,y_k)
	//      => i.e. each pixel loops through all other pixels' w_k's
	// => use 16 rooks in 16x16 tile grids

	template <uint32 TILE_SIZE>
	struct tiled_work_group
	{
		CUGAR_DEVICE
		tiled_work_group() {}

		CUGAR_DEVICE
		uint32 rank() const { return threadIdx.x % TILE_SIZE; }

		template <typename T>
		CUGAR_DEVICE
		T shuffle(T input, const uint32 idx) const { return cub::ShuffleIndex(input, idx, __activemask()); }

		CUGAR_DEVICE
		void sync() { __syncwarp(); }
	};

	FERMAT_DEVICE
	uint2 compute_pixel_coordinates(const uint32 thread_rank, const uint32 pixel_class, const uint2 macro_tile, RendererView& renderer)
	{
		// each thread in the group/subwarp gets a different tile in the macro-tile
		const uint32 tile_x = thread_rank % TILE_X;
		const uint32 tile_y = thread_rank / TILE_X;

		// compute a global tile id
		const uint32 tile_id =
			(macro_tile.x + macro_tile.y * MACRO_TILE_SIZE) * MACRO_TILE_SIZE + tile_x + tile_y*TILE_X +
			renderer.instance * (renderer.res_x * renderer.res_y) / (TILE_X*TILE_Y); // shuffle by the frame number

		// and one specific pixel in that tile, determined by the "pixel_class"
		const uint32 subtile_x = cugar::permute(pixel_class, MACRO_TILE_SIZE, tile_id) % TILE_X;
		const uint32 subtile_y = cugar::permute(pixel_class, MACRO_TILE_SIZE, tile_id) / TILE_X;

		// compute a random per-frame offset
		cugar::LCG_random rand(renderer.instance);
		const uint2 offset = make_uint2( rand.next() % MACRO_TILE_SIZE, rand.next() % MACRO_TILE_SIZE );

		// finally, we can compute the global pixel coordinates
		const uint32 pixel_x = macro_tile.x * MACRO_TILE_SIZE + tile_x * TILE_X + subtile_x - offset.x;
		const uint32 pixel_y = macro_tile.y * MACRO_TILE_SIZE + tile_y * TILE_Y + subtile_y - offset.y;
		return make_uint2( pixel_x, pixel_y );
	}

	FERMAT_DEVICE
	void macrotile_group(tiled_work_group<MACRO_TILE_SIZE> group, const uint32 pixel_class, const uint2 macro_tile, RPTContext& context, RendererView& renderer, float2* smem_w)
	{
		// compute the pixel coordinates for this thread
		const uint2 pixel_coords = compute_pixel_coordinates(group.rank(), pixel_class, macro_tile, renderer);

		smem_w[group.rank()] = cugar::Vector2f(0.0f); // initialize smem_w

		// First Phase:
		//
		// for each pixel j in each tile, compute w_j = p(j,j) / \sum_k p(k,j) = p(j,j) * w_j*, w_j* = 1 / \sum_k p(k,j)
		//      => i.e. each pixel stores its own w_j
		cugar::Vector2f mis_w  = 0.0f;
		cugar::Vector2f p_j    = 0.0f;
		cugar::Vector2f p_sum  = 0.0f;

		// check if this pixel actually exists
		if (pixel_coords.x >= renderer.res_x ||
			pixel_coords.y >= renderer.res_y)
			return;	// this thread can go home...

		// 2.
		// fetch the "vpl" data corresponding to this pixel
		const uint32 pixel = pixel_coords.x + pixel_coords.y * renderer.res_x;

		// check if there is anything to do at all for this pixel
		if (renderer.fb.gbuffer.tri(pixel) == uint32(-1))
			return;	// this thread can go home...

		// TODO: build a group of coalesced threads here for the purpose of syncing later on!

		// fetch the VPL energy and weight
		const cugar::Vector4f light_alpha  = context.vpls.in_alpha[pixel];
		const cugar::Vector4f light_alpha2 = context.vpls.in_alpha2[pixel];
		const cugar::Vector4f light_weight = context.vpls.weight[pixel];

		// check whether it is worth processing
		if (cugar::max_comp(light_alpha2) > 0.0f || (cugar::max_comp(light_alpha) > 0.0f && light_weight.w > 0.0f))
		{
			// reconstruct the VPL geometry
			VertexGeometry light_geom;
			const cugar::Vector4f packed_geo = context.vpls.pos[pixel];
			light_geom.position	= packed_geo.xyz();
			light_geom.normal_s	= unpack_direction(cugar::binary_cast<uint32>(packed_geo.w));
			light_geom.normal_g	= light_geom.normal_s;
			light_geom.tangent	= cugar::orthogonal(light_geom.normal_s);
			light_geom.binormal	= cugar::cross(light_geom.normal_s, light_geom.tangent);

			// fetch the rest of the VPL
			const cugar::Vector3f	light_in	= unpack_direction(context.vpls.in_dir[pixel]);
			const Bsdf				light_bsdf	= unpack_bsdf(renderer, context.vpls.gbuffer[pixel]);

			// 3.
			// loop through all the other pixels in this set, i.e. all of the thread ranks k in this group,
			// and compute their probability p(pixel_k,pixel) of generating the given VPL
			for (uint32 k = 0; k < MACRO_TILE_SIZE; ++k)
			{
				// make sure threads don't go out of sync, as they have to read the same locations of memory
				group.sync(); // TODO: make sure to use the coalesced threads after the first returns!

				// compute the pixel coordinates for this thread
				const uint2 pixel_coords_k = compute_pixel_coordinates(k, pixel_class, macro_tile, renderer);

				const uint32 pixel_k = pixel_coords_k.x + pixel_coords_k.y * renderer.res_x;

				// check if there is anything to do at all for this pixel
				if (pixel_coords_k.x < renderer.res_x &&
					pixel_coords_k.y < renderer.res_y &&
					renderer.fb.gbuffer.tri(pixel_k) != uint32(-1))
				{
					//
					// 3.a compute the probability of generating the given light
					//

					// reconstruct the local geometry
					const VertexGeometryId v_id(
						renderer.fb.gbuffer.tri(pixel_k),
						renderer.fb.gbuffer.uv(pixel_k).x,
						renderer.fb.gbuffer.uv(pixel_k).y);

					VertexGeometry geom;
					setup_differential_geometry(renderer.mesh, v_id, &geom);

					// reconstruct the local bsdf
					const int material_id = renderer.mesh.material_indices[v_id.prim_id];

					FERMAT_ASSERT(material_id < renderer.mesh.num_materials);
					MeshMaterial material = renderer.mesh.materials[material_id];

					// perform all texture lookups
					material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map,		renderer.textures, cugar::Vector4f(1.0f));
					material.specular		*= texture_lookup(geom.texture_coords, material.specular_map,		renderer.textures, cugar::Vector4f(1.0f));
					material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map,		renderer.textures, cugar::Vector4f(1.0f));
					material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map,	renderer.textures, cugar::Vector4f(1.0f));

					FERMAT_ASSERT(material.roughness > 0.0f);
					const Bsdf bsdf(kRadianceTransport, renderer, material);

					// compute the probability with which this light could have been sampled by the current pixel
					const cugar::Vector3f in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - geom.position);

					// compute the connecting edge
					cugar::Vector3f out = light_geom.position - geom.position;

					// check whether we are on the cusp of a singularity...
					if (cugar::square_length(out) == 0.0f)
						continue;

					const float d2 = cugar::max(1.0e-6f, cugar::square_length(out));
					const float d = sqrtf(d2);

					// normalize the outgoing direction
					out /= d;

					// compute the G' term
					const float G_prime = fabsf(cugar::dot(out, light_geom.normal_s)) / d2;

					// and put it all together to get the desired probabilities
					const float p_base =
								bsdf.p(geom, in, out, cugar::kSolidAngle, true/*, Bsdf::kDiffuseMask*/) *
								G_prime;

					const cugar::Vector2f p_k(
						p_base,
						p_base * light_bsdf.p(light_geom, -out, light_in, cugar::kProjectedSolidAngle, true/*, Bsdf::kDiffuseMask*/)
					);

					if (k == group.rank())
						p_j  = p_k;

					p_sum  += p_k * p_k;
					//p_sum += p_k;
				} // if pixel is valid
			} // loop on k

			// w = 1 / sum_k p_k
			if (p_sum.x > 0.0f)
			{
				mis_w = p_j / p_sum;
				//mis_w = 1.0f / p_sum;

				//if (mis_w > 700.0f && fabsf( mis_w - 1624.967285 ) < 0.001f)
				//	printf("mis_w: %f, sum_p: %f, p_j: %f, vpl(%u)\n", mis_w.x, p_sum.x, p_j.x, group.rank());

				smem_w[group.rank()] = mis_w;
			}
		}

		// NOTE:
		// at this point, each thread in the group has computed its own mis_w,
		//    => i.e. thread k owns the value w_k

		group.sync(); // TODO: use the coalesced threads!

		// Second Phase:
		//
		// for each pixel j in each tile, compute sum_k w_k(y_k) * L(x_j,y_k) / p(k,k) = sum_k w_k* L(x_j,y_k)
		//    => i.e. each pixel loops through all other pixels' w_k's

		// reconstruct the local geometry
		const VertexGeometryId v_id(
			renderer.fb.gbuffer.tri(pixel),
			renderer.fb.gbuffer.uv(pixel).x,
			renderer.fb.gbuffer.uv(pixel).y);

		VertexGeometry geom;
		setup_differential_geometry(renderer.mesh, v_id, &geom);

		// reconstruct the local bsdf
		const int material_id = renderer.mesh.material_indices[v_id.prim_id];

		FERMAT_ASSERT(material_id < renderer.mesh.num_materials);
		MeshMaterial material = renderer.mesh.materials[material_id];

		// perform all texture lookups
		material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map,		renderer.textures, cugar::Vector4f(1.0f));
		material.specular		*= texture_lookup(geom.texture_coords, material.specular_map,		renderer.textures, cugar::Vector4f(1.0f));
		material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map,		renderer.textures, cugar::Vector4f(1.0f));
		material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map,	renderer.textures, cugar::Vector4f(1.0f));

		FERMAT_ASSERT(material.roughness > 0.0f);
		const Bsdf bsdf(kRadianceTransport, renderer, material);

		// compute the probability with which this light could have been sampled by the current pixel
		const cugar::Vector3f in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - geom.position);
		
		// keep track of the accumulated results
		cugar::Vector3f f_sum(0.0f);
		float			w_sum(0.0f);

		float cdf[MACRO_TILE_SIZE];

		for (uint32 k = 0; k < MACRO_TILE_SIZE; ++k)
		{
			// make sure threads don't go out of sync, as they have to read the same locations of memory
			group.sync(); // TODO: make sure to use the coalesced threads after the first returns!

			cdf[k] = k ? cdf[k-1] : 0.0f;

			const float2 w_k = smem_w[k];
			//const float w_kk = group.shuffle(mis_w,k);
			//if (w_k != w_kk)
			//	printf("fuck! %f != %f\n", w_k, w_kk);

			// compute the pixel coordinates for this thread
			const uint2 pixel_coords_k = compute_pixel_coordinates(k, pixel_class, macro_tile, renderer);

			const uint32 pixel_k = pixel_coords_k.x + pixel_coords_k.y * renderer.res_x;

			// check wether this pixel is valid
			if (pixel_coords_k.x >= renderer.res_x ||
				pixel_coords_k.y >= renderer.res_y)
				continue;

			const uint32 light_pixel = pixel_k;

			// fetch the VPL energy and weight
			const cugar::Vector4f light_alpha  = context.vpls.in_alpha[light_pixel];
			const cugar::Vector4f light_alpha2 = context.vpls.in_alpha2[light_pixel];
			const cugar::Vector4f light_weight = context.vpls.weight[light_pixel];

			// check whether it is worth processing
			if (cugar::max_comp(light_alpha2) == 0.0f && (cugar::max_comp(light_alpha) == 0.0f || light_weight.w == 0.0f))
				continue;

			// reconstruct the VPL geometry
			VertexGeometry light_geom;
			const cugar::Vector4f packed_geo = context.vpls.pos[light_pixel];
			light_geom.position	= packed_geo.xyz();
			light_geom.normal_s	= unpack_direction(cugar::binary_cast<uint32>(packed_geo.w));
			light_geom.normal_g	= light_geom.normal_s;
			light_geom.tangent	= cugar::orthogonal(light_geom.normal_s);
			light_geom.binormal	= cugar::cross(light_geom.normal_s, light_geom.tangent);

			// fetch the rest of the VPL
			const cugar::Vector3f	light_in	= unpack_direction(context.vpls.in_dir[light_pixel]);
			const cugar::Vector3f	light_in2	= unpack_direction(context.vpls.in_dir2[light_pixel]);
			const Bsdf				light_bsdf	= unpack_bsdf(renderer, context.vpls.gbuffer[light_pixel]);
			const Edf				light_edf	= unpack_edf(context.vpls.ebuffer[light_pixel]);

			// compute the connecting edge
			cugar::Vector3f out = light_geom.position - geom.position;

			// check whether we are on the cusp of a singularity...
			if (cugar::square_length(out) == 0.0f)
				continue;

			const float d2 = cugar::max(1.0e-6f, cugar::square_length(out));
			const float d  = sqrtf(d2);

			// normalize the outgoing direction
			out /= d;

			// compute the G' term
			const float G_prime = fabsf(cugar::dot(out, light_geom.normal_s)) / d2;

			// compute the total path throughput
			const float G = G_prime * fabsf(cugar::dot(out, geom.normal_s));

			const cugar::Vector3f f_s  = bsdf.f(geom, in, out/*, Bsdf::kDiffuseMask*/);
			const cugar::Vector3f f_L  = light_bsdf.f(light_geom, -out, light_in/*, Bsdf::kDiffuseMask*/);
			const cugar::Vector3f f_L2 = light_bsdf.f(light_geom, -out, light_in2/*, Bsdf::kDiffuseMask*/);
			//const cugar::Vector3f e_L = light_edf.f(light_geom, light_geom.position, -out);
			//const cugar::Vector3f f	  = f_s * (e_L * (G * w_k.x) + light_alpha.xyz() * f_L * (G * w_k.y)); // TODO: if we want to enable this, we need to multiply the e_L contribution by its (approximate) MIS weight against NEE
			//const cugar::Vector3f f	  = light_alpha.xyz() * f_s * f_L * (G * w_k.y);
			const cugar::Vector3f f	  = f_s * (
				light_alpha.xyz() * f_L * (G * w_k.y) +
				light_alpha2.xyz() * f_L2 * (G * w_k.x)			// NEE contribution at bounce 1
			);

			if (cugar::is_finite(f))
			{
				const float w_filter = 1.0f;

				// and accumulate with the proper filtering weight
				f_sum += f * w_filter;
				w_sum += w_filter;

				cdf[k] += cugar::max_comp(f);
			}
		} // for loop on k

	  #if 1
		const float cdf_sum = cdf[MACRO_TILE_SIZE-1];
		if (cdf_sum)
		{
			const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

			const float base_sample = context.sequence.sample_2d(pixel_coords.x, pixel_coords.y, 6 + 2);

			cugar::LFSRRandomStream sampler( &context.generator, 1u, 1531u );

			// take N samples according to the cdf
			for (uint32 i = 0; i < REUSE_SHADOW_SAMPLES; ++i)
			{
				const float r = cugar::mod( sampler.next() + base_sample, one );

				const uint32 cdf_idx = cugar::upper_bound_index( cugar::min( r, one ) * cdf_sum, cdf, MACRO_TILE_SIZE );

				const float cdf_begin = cdf_idx ? cdf[cdf_idx-1] : 0.0f;
				const float cdf_end   = cdf[cdf_idx];

				const float pdf = (cdf_end - cdf_begin) / cdf_sum;

				const uint32 k = cdf_idx;

				// compute the pixel coordinates for this entry
				const uint2 pixel_coords_k = compute_pixel_coordinates(k, pixel_class, macro_tile, renderer);

				const uint32 pixel_k = pixel_coords_k.x + pixel_coords_k.y * renderer.res_x;

				// reconstruct the VPL position
				const cugar::Vector4f packed_geo = context.vpls.pos[pixel_k];

				cugar::Vector3f out = packed_geo.xyz() - geom.position;

				// setup a shadow ray
				Ray out_ray;
				out_ray.origin	= geom.position;
				out_ray.dir		= out;
				out_ray.tmin	= 1.0e-3f;
				out_ray.tmax	= 0.999f;

				const PixelInfo out_pixel =
					PixelInfo(pixel, FBufferDesc::COMPOSITED_C);

				context.shadow_queue.warp_append(out_pixel, out_ray, cugar::Vector4f(f_sum / pdf, 0.0f) / REUSE_SHADOW_SAMPLES, cugar::Vector4f(0.0f), 1.0f);
			}
		}
	  #else
		// accumulate the final result to the image
		renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f_sum, 0.0f) / float(renderer.instance + 1);
	  #endif
	}

	__global__
	void tiled_reuse_vpls_kernel(RPTContext context, RendererView renderer)
	{
		__shared__ float2 smem_w[8][MACRO_TILE_SIZE];

		const uint32 block_threads			= blockDim.x * blockDim.y;
		const uint32 blocks_per_macro_tile	= (MACRO_TILE_SIZE * MACRO_TILE_SIZE) / block_threads;

		const uint2 macro_tile = make_uint2(
			blockIdx.x / blocks_per_macro_tile,
			blockIdx.y );

		// for each macro-tile, we launch blocks_per_macro_tile blocks of blockDim.y (sub)warps, each as large as a macro-tile:
		// now we need to identify which MACRO_TILE_SIZE-wide group this is.
		const uint32 group_id = blockDim.y * (blockIdx.x % blocks_per_macro_tile) + threadIdx.y; // group number within a macro-tile

		tiled_work_group<MACRO_TILE_SIZE> group;

		macrotile_group( group, group_id, macro_tile, context, renderer, &smem_w[threadIdx.y][0] );
	}
	void tiled_reuse_vpls(RPTContext context, RendererView renderer)
	{
		//
		// We subdivide the image in 4x4 macro-tiles, each made of 4x4 pixels
		//
		const uint32 macro_tiles_x = cugar::divide_ri(renderer.res_x, MACRO_TILE_SIZE) + 1;
		const uint32 macro_tiles_y = cugar::divide_ri(renderer.res_y, MACRO_TILE_SIZE) + 1;
		//const uint32 num_macro_tiles = macro_tiles_x * macro_tiles_x;

		//
		// For each macro-tile, we launch 16 16-wide warps, i.e. 256 threads
		//
		const dim3 blockSize(MACRO_TILE_SIZE, 8);	// 8 16-wide sub-warps
		const uint32 block_threads			= blockSize.x * blockSize.y;
		const uint32 blocks_per_macro_tile	= (MACRO_TILE_SIZE*MACRO_TILE_SIZE) / block_threads;
		const dim3 gridSize(
			macro_tiles_x * blocks_per_macro_tile,			// each macro-tile in a row launches blocks_per_macro_tile blocks, i.e. MACRO_TILE_SIZE threads
			macro_tiles_y );

		tiled_reuse_vpls_kernel <<< gridSize, blockSize >>> (context, renderer);
	}

	__global__
	void reuse_vpls_kernel(RPTContext context, RendererView renderer)
	{
		const uint32 pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
		const uint32 pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

		const uint32 pixel = pixel_x + pixel_y * renderer.res_x;

		// check which components we have to sample
		uint32 component_mask = uint32(Bsdf::kAllComponents);
		{
			// disable diffuse scattering if not allowed
			if (context.options.diffuse_scattering == false)
				component_mask &= ~uint32(Bsdf::kDiffuseMask);

			// disable glossy scattering if:
			// 1. we have sampled a diffuse reflection and indirect_glossy == false, OR
			// 2. indirect glossy scattering is disabled
			if (context.options.glossy_scattering == false)
				component_mask &= ~uint32(Bsdf::kGlossyMask);
		}

		if (pixel_x < renderer.res_x &&
			pixel_y < renderer.res_y)
		{
			if (renderer.fb.gbuffer.tri(pixel) != uint32(-1))
			{
			#if 0
				const cugar::Vector4f packed_geo = context.vpls.pos[pixel];

				// reconstruct the VPL geometry
				VertexGeometry light_geom;
				light_geom.position	= packed_geo.xyz();
				light_geom.normal_s	= unpack_direction(cugar::binary_cast<uint32>(packed_geo.w));
				light_geom.normal_g	= light_geom.normal_s;
				light_geom.tangent	= cugar::orthogonal(light_geom.normal_s);
				light_geom.binormal	= cugar::cross(light_geom.normal_s, light_geom.tangent);

				const cugar::Vector4f	light_weight	= context.vpls.weight[pixel];
				const cugar::Vector4f	light_alpha		= context.vpls.in_alpha[pixel];
				const cugar::Vector3f	light_in		= unpack_direction(context.vpls.in_dir[pixel]);
				const Bsdf				light_bsdf		= unpack_bsdf(renderer, context.vpls.gbuffer[pixel]);

				renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += light_weight * cugar::Vector4f(light_alpha.xyz(), 0.0f) / float(renderer.instance + 1);
			#else
				// reconstruct the local geometry
				const VertexGeometryId v_id(
					renderer.fb.gbuffer.tri(pixel),
					renderer.fb.gbuffer.uv(pixel).x,
					renderer.fb.gbuffer.uv(pixel).y);

				VertexGeometry geom;
				setup_differential_geometry(renderer.mesh, v_id, &geom);

				// reconstruct the local bsdf
				const int material_id = renderer.mesh.material_indices[v_id.prim_id];

				FERMAT_ASSERT(material_id < renderer.mesh.num_materials);
				MeshMaterial material = renderer.mesh.materials[material_id];

				// perform all texture lookups
				material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map,		renderer.textures, cugar::Vector4f(1.0f));
				material.specular		*= texture_lookup(geom.texture_coords, material.specular_map,		renderer.textures, cugar::Vector4f(1.0f));
				material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map,		renderer.textures, cugar::Vector4f(1.0f));
				material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map,	renderer.textures, cugar::Vector4f(1.0f));

				FERMAT_ASSERT(material.roughness > 0.0f);
				const Bsdf bsdf(kRadianceTransport, renderer, material);

				const uint32 FILTER_WIDTH = context.options.filter_width;

				// accumulate the contributions from all the VPLs in a (FILTER_WIDTH*2+1) x (FILTER_WIDTH*2+1) area
				const uint32 lx = pixel_x > FILTER_WIDTH ? pixel_x - FILTER_WIDTH : 0;
				const uint32 ly = pixel_y > FILTER_WIDTH ? pixel_y - FILTER_WIDTH : 0;

				const uint32 rx = pixel_x + FILTER_WIDTH < renderer.res_x ? pixel_x + FILTER_WIDTH : renderer.res_x - 1;
				const uint32 ry = pixel_y + FILTER_WIDTH < renderer.res_y ? pixel_y + FILTER_WIDTH : renderer.res_y - 1;

				cugar::Vector3f f_sum = 0.0f;
				float			w_sum = 0.0f;

				const uint32 NUM_SAMPLES = 8;

				//const float CENTRAL_WEIGHT = 0.5f;
				//const float CENTRAL_WEIGHT = 1.0f / ((rx - lx + 1)*(ry - ly + 1));
				const float CENTRAL_WEIGHT = 1.0f / (NUM_SAMPLES+1);

			  #if 1
				cugar::LFSRRandomStream sampler( &context.generator, 1u, cugar::hash(pixel) );

				for (uint32 s = 0; s < NUM_SAMPLES; ++s)
				{
					const float u = sampler.next();
					const float v = sampler.next();

					const uint32 xx = lx + cugar::quantize( u, (rx - lx) );
					const uint32 yy = ly + cugar::quantize( v, (ry - ly) );

					const uint32 light_pixel = xx + yy * renderer.res_x;

					// fetch the VPL energy and weight
					const cugar::Vector4f light_alpha  = context.vpls.in_alpha[light_pixel];
					const cugar::Vector4f light_alpha2 = context.vpls.in_alpha2[light_pixel];
					const cugar::Vector4f light_weight = context.vpls.weight[light_pixel];

					if (xx == pixel_x &&
						yy == pixel_y)
						continue;

					// check whether it is worth processing
					if (cugar::max_comp(light_alpha) == 0.0f || light_weight.w == 0.0f)
						continue;

					// reconstruct the VPL geometry
					VertexGeometry light_geom;
					const cugar::Vector4f packed_geo = context.vpls.pos[light_pixel];
					light_geom.position	= packed_geo.xyz();
					light_geom.normal_s	= unpack_direction(cugar::binary_cast<uint32>(packed_geo.w));
					light_geom.normal_g	= light_geom.normal_s;
					light_geom.tangent	= cugar::orthogonal(light_geom.normal_s);
					light_geom.binormal	= cugar::cross(light_geom.normal_s, light_geom.tangent);

					// fetch the rest of the VPL
					const cugar::Vector3f	light_in	= unpack_direction(context.vpls.in_dir[light_pixel]);
					const cugar::Vector3f	light_in2	= unpack_direction(context.vpls.in_dir2[light_pixel]);
					const Bsdf				light_bsdf	= unpack_bsdf(renderer, context.vpls.gbuffer[light_pixel]);

					// fetch the probability with which this light has been sampled
					const float p1   = light_weight.w;
					const float p1_2 = light_alpha2.w;

					// compute the probability with which this light could have been sampled by the central (i.e. current) pixel
					const cugar::Vector3f in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - geom.position);

					// compute the connecting edge
					cugar::Vector3f out = light_geom.position - geom.position;

					// check whether we are on the cusp of a singularity...
					if (cugar::square_length(out) == 0.0f)
						continue;

					const float d2 = cugar::max(1.0e-8f, cugar::square_length(out));
					const float d  = sqrtf(d2);

					// normalize the outgoing direction
					out /= d;

					// compute the G' term
					const float G_prime = fabsf(cugar::dot(out, light_geom.normal_s)) / d2;

					// and put it all together to get the desired probability
					const float p2_2 = bsdf.p(geom, in, out, cugar::kSolidAngle, true, Bsdf::ComponentType(component_mask)) * G_prime;
					const float p2   = p2_2 * light_bsdf.p(light_geom, -out, light_in, cugar::kProjectedSolidAngle);

					// now compute the MIS weight for this sample
					const float mis_w   = mis_heuristic<REUSE_HEURISTIC>(p1,p2);
					const float mis_w_2 = mis_heuristic<REUSE_HEURISTIC>(p1_2,p2_2);

					// compute the total path throughput
					const float G = G_prime * fabsf(cugar::dot(out, geom.normal_s));

					const cugar::Vector3f f_s  = bsdf.f(geom, in, out, Bsdf::ComponentType(component_mask));
					const cugar::Vector3f f_L  = light_bsdf.f(light_geom, -out, light_in, Bsdf::ComponentType(component_mask));
					const cugar::Vector3f f_L2 = light_bsdf.f(light_geom, -out, light_in2, Bsdf::ComponentType(component_mask));
					const cugar::Vector3f f	   = light_alpha.xyz() * f_s * f_L * G * (mis_w / p1);

					if (cugar::is_finite(f))
					{
						//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y))/float(FILTER_WIDTH*FILTER_WIDTH));
						const float w_filter = 1.0f;

						// and accumulate with the proper filtering weight
						f_sum += f * w_filter * (1.0f - CENTRAL_WEIGHT) * 2.0f;
						w_sum += w_filter;
					}
					
					
					const cugar::Vector3f f2 = light_alpha2.xyz() * f_s * f_L2 * G * (mis_w_2 / p1_2);

					if (cugar::is_finite(f2))
					{
						//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y))/float(FILTER_WIDTH*FILTER_WIDTH));
						const float w_filter = 1.0f;

						// and accumulate with the proper filtering weight
						f_sum += f2 * w_filter * (1.0f - CENTRAL_WEIGHT) * 2.0f;
						w_sum += w_filter;
					}
				}
				// accumulate to the image
				renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f_sum, 0.0f) / ((NUM_SAMPLES+1)*(renderer.instance + 1));
			  #endif

			  #if 1
				// fetch the VPL energy and weight for the central pixel
				const cugar::Vector4f light_alpha   = context.vpls.in_alpha[pixel];
				const cugar::Vector4f light_alpha2  = context.vpls.in_alpha2[pixel];
				const cugar::Vector4f light_weight  = context.vpls.weight[pixel];
				const cugar::Vector4f light_weight2 = context.vpls.weight2[pixel];

				// check whether it is worth processing
				if (cugar::max_comp(light_alpha) == 0.0f || light_weight.w == 0.0f)
					return;

				// fetch the rest of the VPL
				const cugar::Vector3f	light_in = unpack_direction(context.vpls.in_dir[pixel]);
				const cugar::Vector3f	light_in2 = unpack_direction(context.vpls.in_dir2[pixel]);
				const Bsdf				light_bsdf = unpack_bsdf(renderer, context.vpls.gbuffer[pixel]);

				// reconstruct the VPL geometry
				VertexGeometry light_geom;
				const cugar::Vector4f packed_geo = context.vpls.pos[pixel];
				light_geom.position = packed_geo.xyz();
				light_geom.normal_s = unpack_direction(cugar::binary_cast<uint32>(packed_geo.w));
				light_geom.normal_g = light_geom.normal_s;
				light_geom.tangent = cugar::orthogonal(light_geom.normal_s);
				light_geom.binormal = cugar::cross(light_geom.normal_s, light_geom.tangent);

				float p_sum = light_weight.w;

				f_sum = cugar::Vector3f(0.0f);
				w_sum = 1.0f;

				sampler = cugar::LFSRRandomStream( &context.generator, 1u, cugar::hash(pixel) );

				// and otherwise add its contribution weighted by the probability of sampling it by any of the neighboring pixels
				for (uint32 s = 0; s < NUM_SAMPLES; ++s)
				{
					const float u = sampler.next();
					const float v = sampler.next();

					const uint32 xx = lx + cugar::quantize( u, (rx - lx) );
					const uint32 yy = ly + cugar::quantize( v, (ry - ly) );
					{
						const uint32 current_pixel = xx + yy * renderer.res_x;

						if (xx == pixel_x &&
							yy == pixel_y)
							continue;

						// reconstruct the local geometry
						const VertexGeometryId v_id(
							renderer.fb.gbuffer.tri(current_pixel),
							renderer.fb.gbuffer.uv(current_pixel).x,
							renderer.fb.gbuffer.uv(current_pixel).y);

						// check whether it is valid
						if (v_id.prim_id == uint32(-1))
							continue;

						VertexGeometry geom;
						setup_differential_geometry(renderer.mesh, v_id, &geom);

						// reconstruct the local bsdf
						const int material_id = renderer.mesh.material_indices[v_id.prim_id];

						FERMAT_ASSERT(material_id < renderer.mesh.num_materials);
						MeshMaterial material = renderer.mesh.materials[material_id];

						// perform all texture lookups
						material.diffuse		*= texture_lookup(geom.texture_coords, material.diffuse_map,		renderer.textures, cugar::Vector4f(1.0f));
						material.specular		*= texture_lookup(geom.texture_coords, material.specular_map,		renderer.textures, cugar::Vector4f(1.0f));
						material.emissive		*= texture_lookup(geom.texture_coords, material.emissive_map,		renderer.textures, cugar::Vector4f(1.0f));
						material.diffuse_trans	*= texture_lookup(geom.texture_coords, material.diffuse_trans_map,	renderer.textures, cugar::Vector4f(1.0f));

						FERMAT_ASSERT(material.roughness > 0.0f);
						const Bsdf bsdf(kRadianceTransport, renderer, material);

						// fetch the probability with which this light has been sampled
						const float p1   = light_weight.w;
						const float p1_2 = light_alpha2.w;

						// compute the probability with which this light could have been sampled by the current pixel
						const cugar::Vector3f in = cugar::normalize(cugar::Vector3f(renderer.camera.eye) - geom.position);

						// compute the connecting edge
						cugar::Vector3f out = light_geom.position - geom.position;

						// check whether we are on the cusp of a singularity...
						if (cugar::square_length(out) == 0.0f)
							continue;

						const float d2 = cugar::max(1.0e-8f, cugar::square_length(out));
						const float d = sqrtf(d2);

						// normalize the outgoing direction
						out /= d;

						// compute the G' term
						const float G_prime = fabsf(cugar::dot(out, light_geom.normal_s)) / d2;

						// and put it all together to get the desired probability
						const float p2_2 = bsdf.p(geom, in, out, cugar::kSolidAngle, true, Bsdf::ComponentType(component_mask)) * G_prime;
						const float p2   = p2_2 * light_bsdf.p(light_geom, -out, light_in, cugar::kProjectedSolidAngle, true, Bsdf::ComponentType(component_mask));

						// now compute the MIS weight for this sample
						const float mis_w = mis_heuristic<REUSE_HEURISTIC>(p1,p2);
						const float mis_w_2 = mis_heuristic<REUSE_HEURISTIC>(p1_2,p2_2);

						// compute the total path throughput
						const float G = G_prime * fabsf(cugar::dot(out, geom.normal_s));

						// in this case we already know what f/p is, since we stored it while sampling the VPL itself...
						const cugar::Vector3f f = light_alpha.xyz() * light_weight.xyz() * mis_w;

						if (cugar::is_finite(f))
						{
							//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y)) / float(FILTER_WIDTH*FILTER_WIDTH));
							const float w_filter = 1.0f;

							// and accumulate with the proper filtering weight
							f_sum += f * w_filter * CENTRAL_WEIGHT * 2.0f;
							w_sum += w_filter;
						}

						// in this case we already know what f/p is, since we stored it while sampling the VPL itself...
						const cugar::Vector3f f2 = light_alpha2.xyz() * light_weight2.xyz() * mis_w_2; // NEE contribution at bounce 1

						if (cugar::is_finite(f2))
						{
							//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y)) / float(FILTER_WIDTH*FILTER_WIDTH));
							const float w_filter = 1.0f;

							// and accumulate with the proper filtering weight
							f_sum += f2 * w_filter * CENTRAL_WEIGHT * 2.0f;
							w_sum += w_filter;
						}

						p_sum += p2;
					}
				}
				// accumulate to the image
				renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f_sum, 0.0f) / ((NUM_SAMPLES+1)*(renderer.instance + 1));
			  #endif

			#endif
			}
		}
	}
	void reuse_vpls(RPTContext context, RendererView renderer)
	{
		const dim3 blockSize(32, 4); // 4 32-wide warps
		const dim3 gridSize(
			cugar::divide_ri(renderer.res_x, blockSize.x),
			cugar::divide_ri(renderer.res_y, blockSize.y));
		reuse_vpls_kernel << < gridSize, blockSize >> > (context, renderer);
	}

	__global__
	void solve_occlusion_kernel(const uint32 in_queue_size, RPTContext context, RendererView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		if (thread_id < in_queue_size) // *context.shadow_queue.size
		{
			const PixelInfo		  pixel_info = context.shadow_queue.pixels[thread_id];
			const Hit			  hit		 = context.shadow_queue.hits[thread_id];
			const cugar::Vector4f w			 = context.shadow_queue.weights[thread_id];
			const cugar::Vector4f w2		 = context.shadow_queue.weights2[thread_id];

			// TODO: break this up in separate diffuse and specular components
			if (hit.t < 0.0f)
			{
				if (context.in_bounce > 1)
				{
					// accumulate this contribution to the VPL's incoming radiance
					context.vpls.in_alpha[pixel_info.pixel] += w2;
				}
				else if (context.in_bounce == 1)
				{
					// set this contribution to the second VPL's incoming radiance
					context.vpls.in_alpha2[pixel_info.pixel] = w2;
				}
				else
					renderer.fb(pixel_info.channel, pixel_info.pixel) += cugar::Vector4f(w.xyz(), 0.0f) / float(renderer.instance + 1);
			}
		}
	}

	void solve_occlusion(const uint32 in_queue_size, RPTContext context, RendererView renderer)
	{
		const uint32 blockSize(128);
		const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
		solve_occlusion_kernel << < gridSize, blockSize >> > (in_queue_size, context, renderer);
	}

} // anonymous namespace

RPT::RPT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void RPT::init(int argc, char** argv, Renderer& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	// parse the options
	m_options.parse(argc, argv);

	fprintf(stderr, "  RPT settings:\n");
	fprintf(stderr, "    path-length     : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee      : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf     : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee    : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf   : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights  : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    direct lighting : %u\n", m_options.direct_lighting ? 1 : 0);
	fprintf(stderr, "    diffuse         : %u\n", m_options.diffuse_scattering ? 1 : 0);
	fprintf(stderr, "    glossy          : %u\n", m_options.glossy_scattering ? 1 : 0);
	fprintf(stderr, "    indirect glossy : %u\n", m_options.indirect_glossy ? 1 : 0);
	fprintf(stderr, "    filter width    : %u\n", m_options.filter_width);

	// pre-alloc all buffers
	m_rays.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_hits.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_pixels.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_weights.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_weights2.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_probs.alloc(n_pixels * (2 + REUSE_SHADOW_SAMPLES));
	m_counters.alloc(3);

	m_vpls.alloc(n_pixels);

	// build the set of shifts
	const uint32 n_dimensions = 6 * (m_options.max_path_length + 1);
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);
	m_sequence.setup(n_dimensions, SHIFT_RES);

	fprintf(stderr, "  creatign mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.m_mesh_lights.init( n_pixels, renderer.m_mesh.view(), renderer.m_mesh_d.view(), renderer.m_texture_views_h.ptr(), renderer.m_texture_views_d.ptr() );

	fprintf(stderr, "  creatign mesh lights... done\n");
}

void RPT::render(const uint32 instance, Renderer& renderer)
{
	// pre-multiply the previous frame for blending
	renderer.multiply_frame(float(instance) / float(instance + 1));

	//fprintf(stderr, "render started (%u)\n", instance);
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	RPTRayQueue queue1;
	RPTRayQueue queue2;
	RPTRayQueue shadow_queue;

	queue1.rays			= m_rays.ptr();
	queue1.hits			= m_hits.ptr();
	queue1.weights		= m_weights.ptr();
	queue1.weights2		= m_weights2.ptr();
	queue1.probs		= m_probs.ptr();
	queue1.pixels		= m_pixels.ptr();
	queue1.size			= m_counters.ptr();
	queue1.max_size     = n_pixels;

	queue2.rays			= queue1.rays + n_pixels;
	queue2.hits			= queue1.hits + n_pixels;
	queue2.weights		= queue1.weights + n_pixels;
	queue2.weights2		= queue1.weights2 + n_pixels;
	queue2.probs		= queue1.probs + n_pixels;
	queue2.pixels		= queue1.pixels + n_pixels;
	queue2.size			= queue1.size + 1;
	queue2.max_size     = n_pixels;

	shadow_queue.rays		= queue2.rays + n_pixels;
	shadow_queue.hits		= queue2.hits + n_pixels;
	shadow_queue.weights	= queue2.weights + n_pixels;
	shadow_queue.weights2	= queue2.weights2 + n_pixels;
	shadow_queue.probs		= queue2.probs + n_pixels;
	shadow_queue.pixels		= queue2.pixels + n_pixels;
	shadow_queue.size		= queue2.size + 1;
	shadow_queue.max_size   = n_pixels * REUSE_SHADOW_SAMPLES;

	cugar::Timer timer;
	timer.start();

	// fetch a view of the renderer
	RendererView renderer_view = renderer.view(instance);

	// setup the samples for this frame
	m_sequence.set_instance(instance);

	RPTContext context;
	context.options			= m_options;
	context.in_bounce		= 0;
	context.in_queue		= queue1;
	context.scatter_queue	= queue2;
	context.shadow_queue	= shadow_queue;
	context.sequence		= m_sequence.view();
	context.vpls			= m_vpls.view();
	context.generator		= m_generator;

	// zero out the vpls' energy and weight
	cudaMemset(context.vpls.in_alpha,  0x00, sizeof(float4)*n_pixels);
	cudaMemset(context.vpls.in_alpha2, 0x00, sizeof(float4)*n_pixels);
	cudaMemset(context.vpls.weight, 0x00, sizeof(float4)*n_pixels);

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
		shade_hits(in_queue_size, context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("shade hits"));

		// trace & accumulate occlusion queries
		{
			uint32 shadow_queue_size;
			cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

			// trace the rays
			//
			if (shadow_queue_size)
			{
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
				solve_occlusion(shadow_queue_size, context, renderer_view);
				CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));
			}
		}

		std::swap(context.in_queue, context.scatter_queue);
	}

	// reset the output queue counters
	cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));

	// reuse all local vpls
	if (m_options.tiled_reuse)
		tiled_reuse_vpls(context, renderer_view);
	else
		reuse_vpls(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("reuse vpls"));

	// trace & accumulate occlusion queries
	{
		// reset the bounce to zero to make sure the results go to the framebuffer
		context.in_bounce = 0;

		uint32 shadow_queue_size;
		cudaMemcpy(&shadow_queue_size, context.shadow_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

		// trace the rays
		//
		if (shadow_queue_size)
		{
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
			solve_occlusion(shadow_queue_size, context, renderer_view);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));
		}
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
