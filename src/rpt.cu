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
#include <tiled_sampling.h>
#include <vector>

#define SHIFT_RES	64u

#define MAX_DEPTH				5
#define DIRECT_LIGHTING_NEE		1
#define DIRECT_LIGHTING_BSDF	1

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

		FERMAT_DEVICE
		void warp_append(const PixelInfo pixel, const Ray& ray, const float4 weight, const float4 weight2, const float p)
		{
			cugar::cuda::warp_static_atomic atomic_adder(size);

			uint32 slot;

			atomic_adder.add<1>(true, &slot);

			rays[slot]		= ray;
			weights[slot]	= weight;
			weights2[slot]	= weight2;
			probs[slot]		= p;
			pixels[slot]	= pixel.packed;
		}
	};

	struct RPTContext
	{
		uint32				in_bounce;

		TiledSequenceView	sequence;

		RPTRayQueue			in_queue;
		RPTRayQueue			shadow_queue;
		RPTRayQueue			scatter_queue;

		RPTVPLView			vpls;
	};

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float mis_heuristic(const float p1, const float p2)
	{
#if 0
		// balance heuristic
		return p1 / (p1 + p2);
#elif 1
		// power heuristic
		return (p1 * p1) / (p1 * p1 + p2 * p2);
#elif 0
		// cutoff heuristic
		const float alpha = 0.1f;
		const float p_max = fmaxf(p1, p2);
		const float q1 = p1 < alpha * p_max ? 0.0f : p1;
		const float q2 = p2 < alpha * p_max ? 0.0f : p2;
		return q1 / (q1 + q2);
#endif
	}

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
	void shade_hits_kernel(const uint32 in_queue_size, RPTContext context, RendererView renderer)
	{
		const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

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
				}

				cugar::Vector3f in = -cugar::normalize(cugar::Vector3f(ray.dir));

				// perform next-event estimation to compute direct lighting
				if (DIRECT_LIGHTING_NEE && context.in_bounce < MAX_DEPTH)
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
						
					const float d2 = fmaxf(1.0e-8f, cugar::square_length(out));
					const float d = sqrtf(d2);

					// normalize the outgoing direction
					out /= d;

					// evaluate the light's EDF and the surface BSDF
					const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;
					const cugar::Vector3f f_s = ev.bsdf.f(ev.geom, ev.in, out);
					// TODO: for the first bounce (i.e. direct-lighting), break this up in separate diffuse and specular components

					// evaluate the geometric term
					const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

					// TODO: perform MIS with the possibility of directly hitting the light source
					const float p1 = light_pdf;
					//const float p2 = bsdf.p(geom, in, out, cugar::kProjectedSolidAngle) * G;
					const float p2 = (ev.bsdf.p(ev.geom, ev.in, out, cugar::kSolidAngle) * fabsf(cugar::dot(out, light_vertex_geom.normal_s))) / d2;
					const float mis_w = DIRECT_LIGHTING_BSDF ? mis_heuristic(p1, p2) : 1.0f;

					// calculate the cumulative sample weight, equal to f_L * f_s * G / p
					const cugar::Vector4f out_w  = cugar::Vector4f(w.xyz() * f_L * f_s * G * mis_w, w.w);
					const cugar::Vector4f out_w2 = cugar::Vector4f(w2.xyz() * f_L * f_s * G * mis_w, w2.w);

					if (cugar::max_comp(out_w.xyz()) > 0.0f && cugar::is_finite(out_w.xyz()))
					{
						// enqueue the output ray
						Ray out_ray;
						out_ray.origin	= ev.geom.position;
						out_ray.dir		= out;
						out_ray.tmin	= 1.0e-3f;
						out_ray.tmax	= d * 0.9999f;

						const PixelInfo out_pixel = context.in_bounce ?
							pixel_info :										// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, FBufferDesc::COMPOSITED_C);	// otherwise (i.e. this is the first bounce) choose the direct-lighting output channel

						context.shadow_queue.warp_append(out_pixel, out_ray, out_w, out_w2, 1.0f);
					}
				}

				// accumulate the emissive component along the incoming direction
				if (DIRECT_LIGHTING_BSDF)
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
					const float mis_w = (DIRECT_LIGHTING_NEE && context.in_bounce) ? mis_heuristic(p1, p2) : 1.0f;

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
							renderer.fb(context.in_bounce == 0 ? FBufferDesc::COMPOSITED_C : pixel_info.channel, pixel_info.pixel) += out_w / float(renderer.instance + 1);
					}
				}

				// trace a bounce ray
				if (context.in_bounce < MAX_DEPTH)
				{
					// fetch the sampling dimensions
					const float z[3] = { samples[3], samples[4], samples[5] }; // use dimensions 3,4,5

					// sample a scattering event
					cugar::Vector3f		out(0.0f);
					cugar::Vector3f		out_g(0.0f);
					float				p(0.0f);
					float				p_proj(0.0f);
					Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

					scatter(ev, z, out_comp, out, p, p_proj, out_g, true, false);

					const cugar::Vector3f out_w  = out_g * ev.alpha;
					const cugar::Vector3f out_w2 = ev.depth == 1 ? cugar::Vector3f(1.0f) : out_g * w2.xyz();

					if (context.in_bounce == 0) // accumulate the average albedo of visible surfaces
					{
						if (out_comp == Bsdf::kDiffuseReflection)
							renderer.fb(FBufferDesc::DIFFUSE_A, pixel_info.pixel) += cugar::Vector4f(out_w, w.w) / float(renderer.instance + 1);
						else if (out_comp == Bsdf::kGlossyReflection)
							renderer.fb(FBufferDesc::SPECULAR_A, pixel_info.pixel) += cugar::Vector4f(out_w, w.w) / float(renderer.instance + 1);
					}

					if (context.in_bounce == 0)
					{
						// store the weight and probability for this scattering event
						context.vpls.weight[pixel_info.pixel] = cugar::Vector4f(out_w, p_proj);
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
							cugar::Vector4f(out_w2, w2.w),
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
		shade_hits_kernel<blockSize / 32> << < gridSize, blockSize >> > (in_queue_size, context, renderer);
	}

	__global__
	void reuse_vpls_kernel(RPTContext context, RendererView renderer)
	{
		const uint32 pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
		const uint32 pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

		const uint32 pixel = pixel_x + pixel_y * renderer.res_x;

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

				const uint32 FILTER_WIDTH = 3;

				// accumulate the contributions from all the VPLs in a (FILTER_WIDTH*2+1) x (FILTER_WIDTH*2+1) area
				const uint32 lx = pixel_x > FILTER_WIDTH ? pixel_x - FILTER_WIDTH : 0;
				const uint32 ly = pixel_y > FILTER_WIDTH ? pixel_y - FILTER_WIDTH : 0;

				const uint32 rx = pixel_x + FILTER_WIDTH < renderer.res_x ? pixel_x + FILTER_WIDTH : renderer.res_x - 1;
				const uint32 ry = pixel_y + FILTER_WIDTH < renderer.res_y ? pixel_y + FILTER_WIDTH : renderer.res_y - 1;

				cugar::Vector3f f_sum = 0.0f;
				float			w_sum = 1.0f;

			  #if 1 // enable/disable the contribution from the neighbors

				for (uint32 yy = ly; yy <= ry; ++yy)
				{
					for (uint32 xx = lx; xx <= rx; ++xx)
					{
						const uint32 light_pixel = xx + yy * renderer.res_x;

						// fetch the VPL energy and weight
						const cugar::Vector4f light_alpha  = context.vpls.in_alpha[light_pixel];
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
						const Bsdf				light_bsdf	= unpack_bsdf(renderer, context.vpls.gbuffer[light_pixel]);

						// fetch the probability with which this light has been sampled
						const float p1 = light_weight.w;

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
						const float p2 = bsdf.p(geom, in, out) * G_prime * light_bsdf.p(light_geom, -out, light_in, cugar::kProjectedSolidAngle);

						// now compute the MIS weight for this sample
						const float mis_w = p1 / (p1 + p2);

						// compute the total path throughput
						const float G = G_prime * fabsf(cugar::dot(out, geom.normal_s));

						const cugar::Vector3f f_s = bsdf.f(geom, in, out);
						const cugar::Vector3f f_L = light_bsdf.f(light_geom, -out, light_in);
						const cugar::Vector3f f	  = light_alpha.xyz() * f_s * f_L * G * (mis_w / p1);

						if (cugar::is_finite(f))
						{
							//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y))/float(FILTER_WIDTH*FILTER_WIDTH));
							const float w_filter = 1.0f;

							// and accumulate with the proper filtering weight
							f_sum += f * w_filter;
							w_sum += w_filter;
						}
					}
				}
				// accumulate to the image
				renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f_sum, 0.0f) / ((rx - lx + 1)*(ry - ly + 1)*(renderer.instance + 1));
			  #endif
			  #if 1 // enable/disable the contribution from the central pixel
				// fetch the VPL energy and weight for the central pixel
				const cugar::Vector4f light_alpha = context.vpls.in_alpha[pixel];
				const cugar::Vector4f light_weight = context.vpls.weight[pixel];

				// check whether it is worth processing
				if (cugar::max_comp(light_alpha) == 0.0f || light_weight.w == 0.0f)
					return;

				// fetch the rest of the VPL
				const cugar::Vector3f	light_in = unpack_direction(context.vpls.in_dir[pixel]);
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

				// and otherwise add its contribution weighted by the probability of sampling it by any of the neighboring pixels
				for (uint32 yy = ly; yy <= ry; ++yy)
				{
					for (uint32 xx = lx; xx <= rx; ++xx)
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


						// fetch the probability with which this light has been sampled
						const float p1 = light_weight.w;

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
						const float p2 = bsdf.p(geom, in, out) * G_prime * light_bsdf.p(light_geom, -out, light_in, cugar::kProjectedSolidAngle);

					  #if 1
						// now compute the MIS weight for this sample
						const float mis_w = p1 / (p1 + p2);

						// compute the total path throughput
						const float G = G_prime * fabsf(cugar::dot(out, geom.normal_s));

						//const cugar::Vector3f f_s = bsdf.f(geom, in, out);
						//const cugar::Vector3f f_L = light_bsdf.f(light_geom, -out, light_in);
						//const cugar::Vector3f f   = light_alpha.xyz() * f_s * f_L * G * (mis_w / p1);
						// in this case we already know what f/p is, since we stored it while sampling the VPL itself...
						const cugar::Vector3f f = light_alpha.xyz() * light_weight.xyz() * mis_w;

						if (cugar::is_finite(f))
						{
							//const float w_filter = expf(-2.0f * float((xx - pixel_x)*(xx - pixel_x) + (yy - pixel_y)*(yy - pixel_y)) / float(FILTER_WIDTH*FILTER_WIDTH));
							const float w_filter = 1.0f;

							// and accumulate with the proper filtering weight
							f_sum += f * w_filter;
							w_sum += w_filter;
						}
					  #endif

						p_sum += p2;
					}
				}
				// accumulate to the image
				renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f_sum, 0.0f) / ((rx - lx + 1)*(ry - ly + 1)*(renderer.instance + 1));

			  #if 0 // this uses an optimal "global" MIS, considering all the pixels in the filter as samplers
				const float mis_w = light_weight.w / p_sum;

				const cugar::Vector3f f = light_alpha.xyz() * light_weight.xyz() * mis_w;

				if (cugar::is_finite(f))
				{
					// and accumulate with the proper filtering weight
					renderer.fb(FBufferDesc::COMPOSITED_C, pixel) += cugar::Vector4f(f, 0.0f) / float(renderer.instance + 1);
				}
			  #endif

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

	// pre-alloc all buffers
	m_rays.alloc(n_pixels * 3);
	m_hits.alloc(n_pixels * 3);
	m_pixels.alloc(n_pixels * 3);
	m_weights.alloc(n_pixels * 3);
	m_weights2.alloc(n_pixels * 3);
	m_probs.alloc(n_pixels * 3);
	m_counters.alloc(3);

	m_vpls.alloc(n_pixels);

	// build the set of shifts
	const uint32 n_dimensions = 6 * (MAX_DEPTH + 2);
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

	queue1.rays = m_rays.ptr();
	queue1.hits = m_hits.ptr();
	queue1.weights = m_weights.ptr();
	queue1.weights2 = m_weights2.ptr();
	queue1.probs = m_probs.ptr();
	queue1.pixels = m_pixels.ptr();
	queue1.size = m_counters.ptr();

	queue2.rays = queue1.rays + n_pixels;
	queue2.hits = queue1.hits + n_pixels;
	queue2.weights = queue1.weights + n_pixels;
	queue2.weights2 = queue1.weights2 + n_pixels;
	queue2.probs = queue1.probs + n_pixels;
	queue2.pixels = queue1.pixels + n_pixels;
	queue2.size = queue1.size + 1;

	shadow_queue.rays = queue2.rays + n_pixels;
	shadow_queue.hits = queue2.hits + n_pixels;
	shadow_queue.weights = queue2.weights + n_pixels;
	shadow_queue.weights2 = queue2.weights2 + n_pixels;
	shadow_queue.probs = queue2.probs + n_pixels;
	shadow_queue.pixels = queue2.pixels + n_pixels;
	shadow_queue.size = queue2.size + 1;

	cugar::Timer timer;
	timer.start();

	// fetch a view of the renderer
	RendererView renderer_view = renderer.view(instance);

	// setup the samples for this frame
	m_sequence.set_instance(instance);

	RPTContext context;
	context.in_bounce = 0;
	context.in_queue = queue1;
	context.scatter_queue = queue2;
	context.shadow_queue = shadow_queue;
	context.sequence = m_sequence.view();
	context.vpls = m_vpls.view();

	// zero out the vpls' energy and weight
	cudaMemset(context.vpls.in_alpha, 0x00, sizeof(float4)*n_pixels);
	cudaMemset(context.vpls.weight, 0x00, sizeof(float4)*n_pixels);

	generate_primary_rays(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	for (context.in_bounce = 0;
		context.in_bounce <= MAX_DEPTH;
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

	// reuse all local vpls
	reuse_vpls(context, renderer_view);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("reuse vpls"));

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
