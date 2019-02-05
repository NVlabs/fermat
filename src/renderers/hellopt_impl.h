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

#include <renderer.h>
#include <rt.h>
#include <mesh_utils.h>
#include <pathtracer_queues.h>
#include <pathtracer_core.h>
#include <cugar/basic/memory_arena.h>

namespace {

//! [HelloPT: HelloPTContext]
// the context class we'll use on the device
//
struct HelloPTContext
{
	HelloPTOptions		options;		// the options
	TiledSequenceView	sequence;		// the sampling sequence
	float				frame_weight;	// the weight given to samples in this frame
	uint32				in_bounce;		// the current path tracing bounce
	
	PTRayQueue			in_queue;		// the input queue
	PTRayQueue			shadow_queue;	// the scattering queue
	PTRayQueue			scatter_queue;	// the shadow queue
};
//! [HelloPT: HelloPTContext]

//! [HelloPT: alloc_queues]
void alloc_queues(
	const uint32			n_pixels,
	PTRayQueue&				input_queue,
	PTRayQueue&				scatter_queue,
	PTRayQueue&				shadow_queue,
	cugar::memory_arena&	arena)
{	
	input_queue.rays		= arena.alloc<MaskedRay>(n_pixels);
	input_queue.hits		= arena.alloc<Hit>(n_pixels);
	input_queue.weights		= arena.alloc<float4>(n_pixels);
	input_queue.pixels		= arena.alloc<uint4>(n_pixels);
	input_queue.size		= arena.alloc<uint32>(1);

	scatter_queue.rays		= arena.alloc<MaskedRay>(n_pixels);
	scatter_queue.hits		= arena.alloc<Hit>(n_pixels);
	scatter_queue.weights	= arena.alloc<float4>(n_pixels);
	scatter_queue.pixels	= arena.alloc<uint4>(n_pixels);
	scatter_queue.size		= arena.alloc<uint32>(1);

	shadow_queue.rays		= arena.alloc<MaskedRay>(n_pixels);
	shadow_queue.hits		= arena.alloc<Hit>(n_pixels);
	shadow_queue.weights	= arena.alloc<float4>(n_pixels);
	shadow_queue.pixels		= arena.alloc<uint4>(n_pixels);
	shadow_queue.size		= arena.alloc<uint32>(1);
}
//! [HelloPT: alloc_queues]

//! [HelloPT: generate_primary_rays]
// a kernel to generate the primary rays 
//
__global__
void generate_primary_rays_kernel(
	HelloPTContext			context,
	RenderingContextView	renderer)
{
	// calculate the 2d pixel index given from the thread id
	const uint2 pixel = make_uint2(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y );

	// check whether the pixel/thread is inside the render target
	if (pixel.x >= renderer.res_x || pixel.y >= renderer.res_y)
		return;

	// calculate a 1d pixel index
	const int idx = pixel.x + pixel.y*renderer.res_x;

	const MaskedRay ray = generate_primary_ray( context, renderer, pixel );

	// write the output ray
	context.in_queue.rays[idx] = ray;

	// write the path weight
	context.in_queue.weights[idx] = cugar::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

	// write the pixel index
	context.in_queue.pixels[idx] = make_uint4( idx, uint32(-1), uint32(-1), uint32(-1) );

	// use thread 0 write out the total number of primary rays in the queue descriptor
	if (pixel.x == 0 && pixel.y)
		*context.in_queue.size = renderer.res_x * renderer.res_y;
}

// dispatch the generate_primary_rays kernel
//
void generate_primary_rays(
	HelloPTContext			context,
	RenderingContextView	renderer)
{
	dim3 blockSize(32, 16);
	dim3 gridSize(cugar::divide_ri(renderer.res_x, blockSize.x), cugar::divide_ri(renderer.res_y, blockSize.y));
	generate_primary_rays_kernel << < gridSize, blockSize >> > (context, renderer);
}
//! [HelloPT: generate_primary_rays]

// shade a path vertex
//
// \param pixel_index		the 1d pixel index associated with this path
// \param pixel				the 2d pixel coordinates
// \param ray				the incoming ray direction
// \param hit				the hit point defining this vertex
// \param w					the current path weight
// \param p_prev			the solid angle probability of the last scattering event
//
// \return					true if the path is continued, false if it terminates here
FERMAT_DEVICE
bool shade_vertex(
	HelloPTContext&			context,
	RenderingContextView&	renderer,
	const uint32			pixel_index,
	const uint2				pixel,
	const MaskedRay&		ray,
	const Hit				hit,
	const cugar::Vector3f	w,
	const float				p_prev)
{
	// check if this is a valid hit
	if (hit.t > 0.0f && hit.triId >= 0)
	{
		// setup an eye-vertex given the input ray, hit point, and path weight
		EyeVertex ev;
		ev.setup(ray, hit, w.xyz(), cugar::Vector4f(0.0f), context.in_bounce, renderer);

		// write out gbuffer information
		if (context.in_bounce == 0)
		{
			renderer.fb.gbuffer.geo(pixel_index) = GBufferView::pack_geometry(ev.geom.position, ev.geom.normal_s);
			renderer.fb.gbuffer.uv(pixel_index)  = make_float4(hit.u, hit.v, ev.geom.texture_coords.x, ev.geom.texture_coords.y);
			renderer.fb.gbuffer.tri(pixel_index) = hit.triId;
			renderer.fb.gbuffer.depth(pixel_index) = hit.t;
		}

		// initialize our shifted sampling sequence
		float samples[6];
		for (uint32 i = 0; i < 6; ++i)
			samples[i] = vertex_sample(pixel, context, i);
		
		// perform next-event estimation to compute direct lighting
		if (context.in_bounce + 2 <= context.options.max_path_length)
		{
			// fetch the sampling dimensions
			const float z[3] = { samples[0], samples[1], samples[2] }; // use dimensions 0,1,2

			VertexGeometryId light_vertex;
			VertexGeometry   light_vertex_geom;
			float			 light_pdf;
			Edf				 light_edf;

			// sample the light source surface
			renderer.mesh_light.sample(z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);

			// join the light sample with the current vertex
			cugar::Vector3f out = (light_vertex_geom.position - ev.geom.position);
						
			const float d2 = fmaxf(1.0e-8f, cugar::square_length(out));

			// normalize the outgoing direction
			out *= rsqrtf(d2);

			// evaluate the light's EDF, predivided by the sample pdf
			const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;

			cugar::Vector3f f_s(0.0f);
			float			p_s(0.0f);

			// evaluate the surface BSDF f() and its sampling pdf p() in one go
			ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s, p_s, cugar::kProjectedSolidAngle);

			//! [HelloPT: compute the sample value]
			// evaluate the geometric term
			const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

			// perform MIS with the possibility of directly hitting the light source
			const float p1 = light_pdf;
			const float p2 = p_s * G;
			const float mis_w = context.in_bounce > 0 ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

			// calculate the cumulative sample weight, equal to f_L * f_s * G / p
			const cugar::Vector3f out_w = w.xyz() * f_L * f_s * G * mis_w;
			//! [HelloPT: compute the sample value]

			//! [HelloPT: enqueue a shadow ray]
			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				// enqueue the output ray
				MaskedRay out_ray;
				out_ray.origin	= ev.geom.position - ray.dir * 1.0e-4f; // shift back in space along the viewing direction
				out_ray.dir		= (light_vertex_geom.position - out_ray.origin); //out;
				out_ray.mask    = 0x2u;
				out_ray.tmax	= 0.9999f; //d * 0.9999f;

				// append the ray to the shadow queue
				context.shadow_queue.warp_append( pixel_index, out_ray, cugar::Vector4f(out_w, 0.0f) );
			}
			//! [HelloPT: enqueue a shadow ray]
		}

		//! [HelloPT: accumulate emissive]
		// accumulate the emissive component along the incoming direction
		{
			VertexGeometry	light_vertex_geom = ev.geom; // the light source geometry IS the current vertex geometry
			float			light_pdf;
			Edf				light_edf;

			// calculate the pdf of sampling this point on the light source
			renderer.mesh_light.map(hit.triId, cugar::Vector2f(hit.u, hit.v), light_vertex_geom, &light_pdf, &light_edf );

			// evaluate the edf's output along the incoming direction
			const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, ev.in);

			const float d2 = fmaxf(1.0e-10f, hit.t * hit.t);

			// compute the MIS weight with next event estimation at the previous vertex
			const float G_partial = fabsf(cugar::dot(ev.in, light_vertex_geom.normal_s)) / d2;
				// NOTE: G_partial doesn't include the dot product between 'in and the normal at the previous vertex

			const float p1 = G_partial * p_prev; // NOTE: p_prev = p_proj * dot(in,normal)
			const float p2 = light_pdf;
			const float mis_w = context.in_bounce > 0 ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

			// and accumulate the weighted contribution
			const cugar::Vector3f out_w	= w.xyz() * f_L * mis_w;

			// and accumulate the weighted contribution
			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				// accumulate the sample contribution to the image
				add_in<false>(renderer.fb(FBufferDesc::COMPOSITED_C), pixel_index, out_w, context.frame_weight);
			}
		}
		//! [HelloPT: accumulate emissive]

		//! [HelloPT: evaluate scattering/absorption]
		// compute a scattering event / trace a bounce ray
		if (context.in_bounce + 1 < context.options.max_path_length)
		{
			// fetch the sampling dimensions
			const float z[3] = { samples[3], samples[4], samples[5] }; // use dimensions 3,4,5

			// sample a scattering event
			cugar::Vector3f		out(0.0f);
			cugar::Vector3f		g(0.0f);
			float				p(0.0f);
			float				p_proj(0.0f);
			Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

			// compute a scattering direction
			scatter(ev, z, out_comp, out, p, p_proj, g, true, false, false, Bsdf::kAllComponents);

			// compute the output weight
			cugar::Vector3f	out_w = g * w.xyz();

			if (p != 0.0f && cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				// enqueue the output ray
				MaskedRay out_ray;
				out_ray.origin	= ev.geom.position;
				out_ray.dir		= out;
				out_ray.mask	= __float_as_uint(1.0e-3f);
				out_ray.tmax	= 1.0e8f;

				// track the solid angle probability of this scattering event
				const float out_p = p;

				// append the ray to the scattering queue
				//
				// notice that we pack the sample probability together with the sample value in a single
				// float4, so as to allow a single 16-byte write into the output queue.
				context.scatter_queue.warp_append( pixel_index, out_ray, cugar::Vector4f( out_w, out_p ) );
				return true;	// continue the path
			}
		}
		//! [HelloPT: evaluate scattering/absorption]
	}
	else
	{
		// hit the environment - do nothing (or, eventually, calculate any sky-lighting)
	}
	return false;	// stop the path
}

//! [HelloPT: shade_vertices]
// shade vertices kernel
//
__global__
void shade_vertices_kernel(const uint32 in_queue_size, HelloPTContext context, RenderingContextView renderer)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.in_queue.size
	{
		const uint32		  pixel_index		= context.in_queue.pixels[thread_id].x;
		const MaskedRay		  ray				= context.in_queue.rays[thread_id];
		const Hit			  hit				= context.in_queue.hits[thread_id];
		const cugar::Vector4f w					= context.in_queue.weights[thread_id];

		const uint2 pixel = make_uint2(
			pixel_index % renderer.res_x,
			pixel_index / renderer.res_x
		);

		shade_vertex(
			context,
			renderer,
			pixel_index,
			pixel,
			ray,
			hit,
			w.xyz(),
			w.w );
	}
}

// dispatch the shade hits kernel
//
void shade_vertices(const uint32 in_queue_size, HelloPTContext context, RenderingContextView renderer)
{
	const uint32 blockSize(64);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));

	shade_vertices_kernel<<< gridSize, blockSize >>>( in_queue_size, context, renderer );
}
//! [HelloPT: shade_vertices]

//! [HelloPT: resolve_occlusion]
// a kernel to resolve NEE samples' occlusion
//
__global__
void resolve_occlusion_kernel(
	const uint32			shadow_queue_size,
	HelloPTContext			context,
	RenderingContextView	renderer)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < shadow_queue_size) // *context.shadow_queue.size
	{
		const uint32		  pixel_index	= context.shadow_queue.pixels[thread_id].x;
		const Hit			  hit			= context.shadow_queue.hits[thread_id];
		const cugar::Vector4f w				= context.shadow_queue.weights[thread_id];

		if (hit.t < 0.0f) // add this sample if and only if there was no intersection
			add_in<false>( renderer.fb(FBufferDesc::COMPOSITED_C), pixel_index, w.xyz(), context.frame_weight );
	}
}

// dispatch the resolve_occlusion kernel
//
void resolve_occlusion(
	const uint32			shadow_queue_size,
	HelloPTContext			context,
	RenderingContextView	renderer)
{
	const uint32 blockSize(64);
	const dim3 gridSize(cugar::divide_ri(shadow_queue_size, blockSize));

	resolve_occlusion_kernel<<< gridSize, blockSize >>>( shadow_queue_size, context, renderer );
}
//! [HelloPT: resolve_occlusion]

} // anonymous namespace
