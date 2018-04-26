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

#include <bpt_context.h>
#include <bpt_utils.h>
#include <bpt_options.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

#define SECONDARY_EYE_VERTICES_BLOCKSIZE		128
#define SECONDARY_EYE_VERTICES_CTA_BLOCKS		6

#define SECONDARY_LIGHT_VERTICES_BLOCKSIZE		128
#define SECONDARY_LIGHT_VERTICES_CTA_BLOCKS		6

///
/// This class implements a "policy" template for bidirectional path tracing: all bidirectional path tracing kernels are templated over a TBPTConfig class
/// that must implement this interface.
/// In order to change the policy, it is sufficient to inherit from this class and override the methods. Note that the methods are not virtual, but the code is written so
/// as to use the implementation of the last class that overrides them, mantaining the inlining efficiency.
///
struct BPTConfigBase
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	BPTConfigBase(
		const BPTOptionsBase&	_options,
		VertexSampling			_light_sampling = VertexSampling::kAll,
		VertexOrdering			_light_ordering = VertexOrdering::kRandomOrdering,
		VertexSampling			_eye_sampling	= VertexSampling::kAll,
		bool					_use_rr			= true) :
		max_path_length(_options.max_path_length),
		light_sampling(uint32(_light_sampling)),
		light_ordering(uint32(_light_ordering)),
		eye_sampling(uint32(_eye_sampling)),
		use_vpls(_options.use_vpls),
		use_rr(_use_rr),
		light_tracing(_options.light_tracing),
		direct_lighting_nee(_options.direct_lighting_nee),
		direct_lighting_bsdf(_options.direct_lighting_bsdf),
		indirect_lighting_nee(_options.indirect_lighting_nee),
		indirect_lighting_bsdf(_options.indirect_lighting_bsdf),
		visible_lights(_options.visible_lights) {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	BPTConfigBase(
		uint32			_max_path_length		= 6,
		VertexSampling	_light_sampling			= VertexSampling::kAll,
		VertexOrdering	_light_ordering			= VertexOrdering::kRandomOrdering, 
		VertexSampling	_eye_sampling			= VertexSampling::kAll,
		bool			_use_vpls				= true,
		bool			_use_rr					= true,
		float			_light_tracing			= 0.0f,
		bool			_direct_lighting_nee	= true,
		bool			_direct_lighting_bsdf	= true,
		bool			_indirect_lighting_nee  = true,
		bool			_indirect_lighting_bsdf = true,
		bool			_visible_lights			= true) :
		max_path_length(_max_path_length),
		light_sampling(uint32(_light_sampling)),
		light_ordering(uint32(_light_ordering)),
		eye_sampling(uint32(_eye_sampling)),
		use_vpls(_use_vpls),
		use_rr(_use_rr),
		light_tracing(_light_tracing),
		direct_lighting_nee(_direct_lighting_nee),
		direct_lighting_bsdf(_direct_lighting_bsdf),
		indirect_lighting_nee(_indirect_lighting_nee),
		indirect_lighting_bsdf(_indirect_lighting_bsdf),
		visible_lights(_visible_lights) {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	BPTConfigBase(const BPTConfigBase& other) :
		max_path_length(other.max_path_length),
		light_sampling(other.light_sampling),
		light_ordering(other.light_ordering),
		eye_sampling(other.eye_sampling),
		use_vpls(other.use_vpls),
		use_rr(other.use_rr),
		light_tracing(other.light_tracing),
		direct_lighting_nee(other.direct_lighting_nee),
		direct_lighting_bsdf(other.direct_lighting_bsdf),
		indirect_lighting_nee(other.indirect_lighting_nee),
		indirect_lighting_bsdf(other.indirect_lighting_bsdf),
		visible_lights(other.visible_lights) {}

	uint32  max_path_length			: 10;
	uint32	light_sampling			: 1;
	uint32  light_ordering			: 1;
	uint32  eye_sampling			: 1;
	uint32	use_vpls				: 1;
	uint32	use_rr					: 1;
	uint32	direct_lighting_nee		: 1;
	uint32	direct_lighting_bsdf	: 1;
	uint32  indirect_lighting_nee	: 1;
	uint32  indirect_lighting_bsdf  : 1;
	uint32	visible_lights			: 1;
	float   light_tracing;

	/// decide whether to terminate a given light subpath
	///
	/// \param path_id			index of the light subpath
	/// \param s				vertex number along the light subpath
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool terminate_light_subpath(const uint32 path_id, const uint32 s) const { return s >= max_path_length + 1; }

	/// decide whether to terminate a given eye subpath
	///
	/// \param path_id			index of the eye subpath
	/// \param s				vertex number along the eye subpath
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool terminate_eye_subpath(const uint32 path_id, const uint32 t) const { return t >= max_path_length + 1; }

	/// decide whether to store a given light vertex
	///
	/// \param path_id			index of the light subpath
	/// \param s				vertex number along the light subpath
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool store_light_vertex(const uint32 path_id, const uint32 s, const bool absorbed) const
	{
		return (VertexSampling(light_sampling) == VertexSampling::kAll) || absorbed;
	}

	/// decide whether to perform a bidirectional connection
	///
	/// \param eye_path_id		index of the eye subpath
	/// \param t				vertex number along the eye subpath
	/// \param absorbed			true if the eye subpath has been absorbed/terminated here
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool perform_connection(const uint32 eye_path_id, const uint32 t, const bool absorbed) const
	{
		return
			(t == 1 && direct_lighting_nee) ||
			(t >  1 && indirect_lighting_nee);
	}

	/// decide whether to accumulate an emissive sample from a pure (eye) path tracing estimator
	///
	/// \param eye_path_id		index of the eye subpath
	/// \param t				vertex number along the eye subpath
	/// \param absorbed			true if the eye subpath has been absorbed/terminated here
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool accumulate_emissive(const uint32 eye_path_id, const uint32 t, const bool absorbed) const
	{
		return
			(t == 2 && visible_lights) ||
			(t == 3 && direct_lighting_bsdf) ||
			(t  > 3 && indirect_lighting_bsdf);
	}

	/// allow to process/store the given light vertex
	///
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void visit_light_vertex(
		const uint32			light_path_id,
		const uint32			depth,
		const VertexGeometryId	v_id,
		BPTContextBase&			context,
		RendererView&			renderer) const
	{}

	/// allow to process/store the given eye vertex
	///
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	void visit_eye_vertex(
		const uint32			eye_path_id,
		const uint32			depth,
		const VertexGeometryId	v_id,
		const EyeVertex&		v,
		BPTContextBase&			context,
		RendererView&			renderer) const
	{}
};

///
/// Base class for sample sinks, a class specifying how to handle final samples produced by the bidirectional path tracing kernels
///
/// \anchor SampleSinkAnchor
///
struct SampleSinkBase
{
	///
	/// Sink a full path sample
	///
	FERMAT_HOST_DEVICE
	void sink(
		const uint32			channel,
		const cugar::Vector4f	value,
		const uint32			light_path_id,
		const uint32			eye_path_id,
		const uint32			s,
		const uint32			t,
		BPTContextBase&			context,
		RendererView&			renderer)
	{}

	///
	/// record an eye scattering event
	///
	FERMAT_HOST_DEVICE
	void sink_eye_scattering_event(
		const Bsdf::ComponentType	component,
		const cugar::Vector4f		value,
		const uint32				eye_path_id,
		const uint32				t,
		BPTContextBase&				context,
		RendererView&				renderer)
	{}
};

///@} BPTLib
///@} Fermat

namespace bpt {

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

///
/// This function generates the primary light vertex for a given path, expressed by the path id, and its primary sample space coordinates.
///
/// \param light_path_id		the id of the light subpath
/// \param n_light_paths		the total number of generated light subpaths
/// \param primary_coords		the primary sample space coordinates generator
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
/// \param config				the bidirectional path tracing configuration policy
///
template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
FERMAT_HOST_DEVICE
void generate_primary_light_vertex(
	const uint32				light_path_id,
	const uint32				n_light_paths,
	const TPrimaryCoordinates&	primary_coords,
	TBPTContext&				context,
	RendererView&				renderer,
	TBPTConfig&					config)
{
	//if (light_path_id == 0)
	//{
	//	if (config.light_sampling == VertexSampling::kAll &&
	//		config.light_ordering == VertexOrdering::kRandomOrdering)
	//		*context.light_vertices.vertex_counter = n_light_paths;
	//}

	if (VertexOrdering(config.light_ordering) == VertexOrdering::kPathOrdering)
	{
		// initialize the number of vertices for this path
		context.light_vertices.vertex_counts[light_path_id] = 0;

		// temporarily store an invalid light vertex - used in case the light subpath gets terminated early due to RR
		context.light_vertices.vertex_path_id[light_path_id] = uint32(-1);
	}

	// check whether we have anything to do
	if (config.terminate_light_subpath(light_path_id, 0) == true)
		return;

	VPL				light_vertex;
	VertexGeometry  geom;
	float			pdf;
	Edf			    edf;

	if (config.use_vpls)
	{
		light_vertex = context.light_vertices.vertex[light_path_id];

		renderer.mesh_vpls.map(light_vertex.prim_id, light_vertex.uv, &geom, &pdf, &edf);
	}
	else
	{
		float samples[3];
		for (uint32 i = 0; i < 3; ++i)
			samples[i] = primary_coords.sample(light_path_id, 0, i);

		renderer.mesh_light.sample(samples, &light_vertex.prim_id, &light_vertex.uv, &geom, &pdf, &edf);
	}

	// store the compact vertex information
	config.visit_light_vertex(
		light_path_id,
		0,
		light_vertex,
		context,
		renderer);

	const bool terminate = config.terminate_light_subpath(light_path_id, 1);

	if (terminate || (VertexSampling(config.light_sampling) == VertexSampling::kAll))
	{
		const uint32 slot = (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering) ?
			cugar::atomic_add(context.light_vertices.vertex_counter, 1u) :
			light_path_id;

		const uint32 packed_normal = pack_direction(geom.normal_s);

		// store the light vertex
		context.light_vertices.vertex_gbuffer[slot] = pack_edf(edf);
		context.light_vertices.vertex_pos[slot]		= cugar::Vector4f(geom.position, cugar::binary_cast<float>(packed_normal));
		context.light_vertices.vertex_input[slot]	= make_uint2(0, cugar::to_rgbe(cugar::Vector3f(1.0f) / pdf)); // the material emission factor
		context.light_vertices.vertex_weights[slot] = PathWeights(
			0.0f,			// p(-2)g(-2)p(-1)
			1.0f * pdf);	// p(-1)g(-1) = pdf								: we want p(-1)g(-1)p(0) = p(0)*pdf - which will happen because we are setting p(0) = p(0) and g(-1) = pdf

		// (over-)write the path id
		context.light_vertices.vertex_path_id[slot] = light_path_id;

		if (VertexOrdering(config.light_ordering) == VertexOrdering::kPathOrdering)
		{
			// set the number of path vertices to one
			context.light_vertices.vertex_counts[light_path_id] = 1;
		}
	}

	if (terminate == false)
	{
		float samples[3];
		for (uint32 i = 0; i < 3; ++i)
			samples[i] = primary_coords.sample(light_path_id, 1, i);

		cugar::Vector3f out;
		cugar::Vector3f f;
		cugar::Vector3f g;
		float			p;
		float			p_proj;

		// sample an outgoing direction
		edf.sample(cugar::Vector2f(samples[0], samples[1]), geom, geom.position, out, g, p, p_proj);

		f = g * p_proj;
		g /= pdf;

		Ray out_ray;
		out_ray.origin	= geom.position;
		out_ray.dir		= out;
		out_ray.tmin	= 1.0e-4f;
		out_ray.tmax	= 1.0e8f;

		// fetch the output slot
		const uint32 slot = context.scatter_queue.append_slot();

		// write the output ray/info
		context.scatter_queue.rays[slot]			= out_ray;
		context.scatter_queue.weights[slot]			= cugar::Vector4f(g, 0.0f);
		context.scatter_queue.probs[slot]			= p; // we need to track the solid angle probability of the last vertex
		context.scatter_queue.pixels[slot]			= PixelInfo(light_path_id, FBufferDesc::DIFFUSE_C).packed;
		context.scatter_queue.path_weights[slot]	= TempPathWeights(
			0.0f,								// p(-2)g(-2)p(-1)
			1.0f * pdf,							// p(-1)g(-1) = 1			: we want p(-1)g(-1)p(0) = p(0) - which will happen because we are setting p(0) = p(0) and g(-1) = pdf
			p_proj,								// p(0)
			fabsf(dot(geom.normal_s, out)));	// cos(theta_0)
	}
}

///
/// This function processes the secondary light vertex corresponding to a given entry in the path tracing queue,
/// stored in the bidirectional path tracing context.
///
/// Specifically, processing a queue entry means performing the following operations:
///
///	- fetching the corresponding ray and hit (or miss) information from the queue
/// - interpolating the local geometry of the hit (or that of the environment on a miss)
/// - reconstructing the local BSDF
/// - potentially storing the resulting light vertex, based on the passed configuration/policy
/// - sampling another scattering/absorption event
///
/// \param queue_idx			the index of the queue entry
/// \param n_light_paths		the total number of generated light subpaths
/// \param primary_coords		the primary sample space coordinates generator
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
/// \param config				the bidirectional path tracing configuration policy
///
template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
FERMAT_HOST_DEVICE
void process_secondary_light_vertex(
	const uint32				queue_idx,
	const uint32				n_light_paths,
	const TPrimaryCoordinates&	primary_coords,
	TBPTContext&				context,
	RendererView&				renderer,
	TBPTConfig&					config)
{
	const PixelInfo		  pixel_info	= context.in_queue.pixels[queue_idx];
	const Ray			  ray			= context.in_queue.rays[queue_idx];
	const Hit			  hit			= context.in_queue.hits[queue_idx];
	const cugar::Vector4f w				= context.in_queue.weights[queue_idx];
	const TempPathWeights path_weights	= context.in_queue.path_weights[queue_idx];

	const uint32 light_path_id = pixel_info.pixel;

	if (hit.t > 0.0f && hit.triId >= 0)
	{
		// store the compact vertex information
		config.visit_light_vertex(
			light_path_id,
			context.in_bounce + 1,
			VertexGeometryId(hit.triId, hit.u, hit.v),
			context,
			renderer);

		// setup the light vertex
		LightVertex lv;
		lv.setup(ray, hit, w.xyz(), path_weights, context.in_bounce + 1, renderer);

		bool absorbed = true;

		// trace a bounce ray
		if (config.terminate_light_subpath(light_path_id, context.in_bounce + 2) == false)
		{
			// initialize our sampling sequence
			float z[3];
			for (uint32 i = 0; i < 3; ++i)
				z[i] = primary_coords.sample(light_path_id, context.in_bounce + 2, i);

			// sample a scattering event
			cugar::Vector3f		out(0.0f);
			cugar::Vector3f		out_w(0.0f);
			cugar::Vector3f		g(0.0f);
			float				p(0.0f);
			float				p_proj(0.0f);
			Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

			scatter(lv, z, out_comp, out, p, p_proj, out_w, config.use_rr, true, true);

			if (cugar::max_comp(out_w) > 0.0f)
			{
				// enqueue the output ray
				Ray out_ray;
				out_ray.origin	= lv.geom.position;
				out_ray.dir		= out;
				out_ray.tmin	= 1.0e-4f;
				out_ray.tmax	= 1.0e8f;

				const PixelInfo out_pixel = pixel_info;

				context.scatter_queue.append(
					out_pixel,
					out_ray,
					cugar::Vector4f(out_w, w.w),
					0.0f,
					TempPathWeights(
						lv.pGp_sum,									// p(i-2)g(i-2)p(i-1)
						lv.prev_pG,									// p(i-1)g(i-1)
						p_proj,										// p(i)
						fabsf(dot(lv.geom.normal_s, out))));		// cos(theta_i)

				absorbed = false;
			}
		}

		// store the light vertex
		if (config.store_light_vertex(light_path_id, context.in_bounce + 2, absorbed))
		{
			const uint32 slot = (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering) ?
				cugar::atomic_add(context.light_vertices.vertex_counter, 1u) :
				light_path_id + context.light_vertices.vertex_counts[light_path_id] * n_light_paths;	// store all vertices: use the global vertex index

			const uint32 packed_normal	  = pack_direction(lv.geom.normal_s);
			const uint32 packed_direction = pack_direction(lv.in);

			context.light_vertices.vertex_gbuffer[slot] = pack_bsdf(lv.material);
			context.light_vertices.vertex_pos[slot]		= cugar::Vector4f(lv.geom.position, cugar::binary_cast<float>(packed_normal));
			context.light_vertices.vertex_input[slot]	= make_uint2(packed_direction, cugar::to_rgbe(w.xyz()));
			context.light_vertices.vertex_weights[slot]	= PathWeights(
				lv.pGp_sum,					// f(i-2)g(i-2)f(i-1)
				lv.prev_pG);				// f(i-1)g(i-1)

			context.light_vertices.vertex_path_id[slot] = light_path_id | ((context.in_bounce + 1) << 24);

			if (VertexOrdering(config.light_ordering) == VertexOrdering::kPathOrdering)
			{
				// keep track of how many vertices this path has
				context.light_vertices.vertex_counts[light_path_id]++;
			}
		}
	}
	else
	{
		// hit the environment - nothing to do
	}
}

///
/// This function generates the primary eye vertex for a given path, expressed by the path id, and its primary sample space coordinates.
///
/// \param light_path_id		the id of the light subpath
/// \param n_light_paths		the total number of generated light subpaths
/// \param primary_coords		the primary sample space coordinates generator
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
/// \param config				the bidirectional path tracing configuration policy
///
template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
FERMAT_HOST_DEVICE
void generate_primary_eye_vertex(
	const uint32				idx,
	const uint32				n_eye_paths,
	const uint32				n_light_paths,
	const TPrimaryCoordinates&	primary_coords,
	TBPTContext&				context,
	const RendererView&			renderer,
	TBPTConfig&					config)
{
	const cugar::Vector2f uv(
		primary_coords.sample(idx, 1, 0),
		primary_coords.sample(idx, 1, 1));

	const cugar::Vector2f d = uv * 2.f - cugar::Vector2f(1.f);

	// write the pixel index
	context.in_queue.pixels[idx] = idx;

	cugar::Vector3f ray_origin	  = renderer.camera.eye;
	cugar::Vector3f ray_direction = d.x*context.camera_U + d.y*context.camera_V + context.camera_W;

	((float4*)context.in_queue.rays)[2 * idx + 0] = make_float4(ray_origin.x, ray_origin.y, ray_origin.z, 0.0f);			// origin, tmin
	((float4*)context.in_queue.rays)[2 * idx + 1] = make_float4(ray_direction.x, ray_direction.y, ray_direction.z, 1e34f);	// dir, tmax

	// write the filter weight
	context.in_queue.weights[idx] = cugar::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

	const float p_e			= camera_direction_pdf(context.camera_U, context.camera_V, context.camera_W, context.camera_W_len, context.camera_square_focal_length, cugar::normalize(ray_direction), true);
	const float cos_theta	= dot(cugar::normalize(ray_direction), context.camera_W) / context.camera_W_len;

	// write the path weights
	context.in_queue.path_weights[idx] = TempPathWeights(
		0.0f,		// 1.0 / p(-2)g(-2)p(-1)
		1.0e8f,		// p(-1)g(-1)		= +inf : hitting the camera is impossible
		config.light_tracing ? p_e / (/*n_light_paths * */config.light_tracing)	: 1.0f,		// out_p = p(0)
		config.light_tracing ? cos_theta									: 1.0e8f);	// out_cos_theta : so that the term f_0 g_0 f_1 (i.e. connection to the lens) gets the proper weight
																						//			+inf : so that the term f_0 g_0 f_1(i.e.connection to the lens) gets zero weight

	if (idx == 0)
		*context.in_queue.size = n_eye_paths;
}

///
/// This function processes the secondary eye vertex corresponding to a given entry in the path tracing queue,
/// stored in the bidirectional path tracing context.
///
/// Specifically, processing a queue entry means performing the following operations:
///
///	- fetching the corresponding ray and hit (or miss) information from the queue
/// - interpolating the local geometry of the hit (or that of the environment on a miss)
/// - reconstructing the local BSDF
/// - performing Next-Event Estimation, if enabled, and handing the resulting sample (a full path) to the output sample sink
/// - computing the local emission at the hit towards the incoming direction, i.e. forming a full path through pure forward path tracing,
///   and passing the resulting sample to the output sample sink
/// - sampling another scattering/absorption event
///
/// \param queue_idx			the index of the queue entry
/// \param n_eye_paths			the total number of generated eye subpaths
/// \param n_light_paths		the total number of generated light subpaths
/// \param sample_sink			the output sample sink, processing all bidirectional paths formed processing this vertex
/// \param primary_coords		the primary sample space coordinates generator
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
/// \param config				the bidirectional path tracing configuration policy
///
template <typename TSampleSink, typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
FERMAT_HOST_DEVICE
void process_secondary_eye_vertex(
	const uint32				queue_idx,
	const uint32				n_eye_paths,
	const uint32				n_light_paths,
	TSampleSink&				sample_sink,
	const TPrimaryCoordinates&	primary_coords,
	TBPTContext&				context,
	RendererView&				renderer,
	TBPTConfig&					config)
{
	const PixelInfo		  pixel_info	= context.in_queue.pixels[queue_idx];
	const Ray			  ray			= context.in_queue.rays[queue_idx];
	const Hit			  hit			= context.in_queue.hits[queue_idx];
	const cugar::Vector4f w				= context.in_queue.weights[queue_idx];
	const TempPathWeights path_weights	= context.in_queue.path_weights[queue_idx];

	const uint32 eye_path_id = pixel_info.pixel;

	//bool sinked_path = false;

	if (hit.t > 0.0f && hit.triId >= 0)
	{
		// setup the eye vertex
		EyeVertex ev;
		ev.setup(ray, hit, w.xyz(), path_weights, context.in_bounce, renderer);

		// store the compact vertex information
		config.visit_eye_vertex(
			eye_path_id,
			context.in_bounce + 1,
			VertexGeometryId(hit.triId, cugar::Vector2f(hit.u, hit.v)),
			ev,
			context,
			renderer);

		bool absorbed = true;

		// trace a bounce ray
		if (config.terminate_eye_subpath(eye_path_id, context.in_bounce + 2) == false)
		{
			// fetch the sampling dimensions
			float z[3];
			for (uint32 i = 0; i < 3; ++i)
				z[i] = primary_coords.sample(pixel_info.pixel, context.in_bounce + 2, i);

			// sample a scattering event
			cugar::Vector3f		out(0.0f);
			cugar::Vector3f		out_w(0.0f);
			float				p(0.0f);
			float				p_proj(0.0f);
			Bsdf::ComponentType out_comp(Bsdf::kAbsorption);

			scatter(ev, z, out_comp, out, p, p_proj, out_w, config.use_rr, true, true);

			if (cugar::max_comp(out_w) > 0.0f)
			{
				// record an eye scattering event
				sample_sink.sink_eye_scattering_event(
					out_comp,
					cugar::Vector4f(out_w, w.w),
					pixel_info.pixel,
					context.in_bounce + 2,
					context,
					renderer);

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

				context.scatter_queue.append(
					out_pixel, out_ray,
					cugar::Vector4f(out_w, w.w),
					out_p,
					TempPathWeights(
						ev.pGp_sum,																// p_(i-2)g_(i-2)p_(i-1)
						ev.prev_pG,																// p_(i-1)g_(i-1)
						p_proj,																	// p_(i)
						fabsf(dot(ev.geom.normal_s, out))));									// cos(theta_i)

				//sinked_path = true;
				absorbed = false;
			}
		}

		// compute the maximum depth a light vertex might have
		const int32 max_light_depth = config.max_path_length - ev.depth - 2;

		// perform a bidirectional connection
		if (max_light_depth >= 0 &&
			config.perform_connection(eye_path_id, context.in_bounce + 2, absorbed))
		{
			const bool single_connection =
				VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering ||
				VertexSampling(config.light_sampling) == VertexSampling::kEnd;

			if (single_connection)
			{
				uint32 light_idx;
				float  light_weight;

				if (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering)
				{
					// fetch the sampling dimensions
					float z[3];
					for (uint32 i = 0; i < 3; ++i)
						z[i] = primary_coords.sample(pixel_info.pixel, context.in_bounce + 2, 3 + i);

					const uint32 n_light_vertices	  = context.light_vertices.vertex_counts[max_light_depth];
					const uint32 n_light_vertex_paths = context.light_vertices.vertex_counts[0];

					//
					// The theory: we want to accumulate all VPLs at each depth, weighted by 1/#(light_vertex_paths);
					//             an alternative estimator is to pick one at each depth, and weight its contribution by w_depth = #photon(depth)/#light_vertex_paths.
					// The practice: we pick 1 VPL out of all of them, with a probability of picking it at a given depth of p_depth = #photon(depth)/#photon(total);
					//               hence, we need to reweight this by w_depth / p_depth = 
					//						#photon(depth) / #light_vertex_paths / (#photon(depth)/#photon(total)) =
					//						#photon(depth) / #light_vertex_paths * #photon(total) / #photon(depth) =
					//						#photon(total) / #light_vertex_paths;

					// select a VPL with z[2]
					light_idx = cugar::quantize(z[2], n_light_vertices);
					light_weight = float(n_light_vertices) / float(n_light_vertex_paths);
				}
				else
				{
					light_idx = eye_path_id;
					light_weight = 1.0f;
				}

				// setup a light vertex
				cugar::Vector4f	 light_pos			= context.light_vertices.vertex_pos[light_idx];
				const uint2		 light_in			= context.light_vertices.vertex_input[light_idx];
				uint4            light_gbuffer		= context.light_vertices.vertex_gbuffer[light_idx];
				PathWeights      light_weights		= context.light_vertices.vertex_weights[light_idx];
				const uint32	 light_vertex_id    = context.light_vertices.vertex_path_id[light_idx];
				const uint32     light_path_id		= light_vertex_id & 0xFFFFFF;
				const uint32     light_depth		= light_vertex_id >> 24;

				// make sure the light vertex is valid
				if (light_vertex_id != uint32(-1))
				{
					// setup the light vertex
					LightVertex lv;
					lv.setup(light_pos, light_in, light_gbuffer, light_weights, light_depth, renderer);

					// evaluate the connection
					cugar::Vector3f out;
					cugar::Vector3f out_w;
					float			d;

					eval_connection(ev, lv, out, out_w, d, config.use_rr, config.direct_lighting_nee, config.direct_lighting_bsdf);

					// multiply by the light vertex weight
					out_w *= light_weight;

					if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
					{
						// recompute d for intersection calculations
						if (SHADOW_BIAS) d = cugar::length(lv.geom.position - (ev.geom.position + ev.in * SHADOW_BIAS));

						// enqueue the output ray
						Ray out_ray;
						//out_ray.origin	= ev.geom.position + ev.in * SHADOW_BIAS; // move the origin slightly towards the viewer
						//out_ray.dir		= out;
						//out_ray.tmin	= SHADOW_TMIN;
						//out_ray.tmax	= d * 0.9999f;
						out_ray.origin	= ev.geom.position + ev.in * SHADOW_BIAS; // shift back in space along the viewing direction
						out_ray.dir		= (lv.geom.position - out_ray.origin); //out;
						out_ray.tmin	= SHADOW_TMIN;
						out_ray.tmax	= 0.9999f;

						const PixelInfo out_pixel = context.in_bounce ?
							pixel_info :										// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, FBufferDesc::DIRECT_C);	// otherwise (i.e. this is the first bounce) choose the direct-lighting output channel

						const uint32 slot = context.shadow_queue.append_slot();

						context.shadow_queue.pixels[slot]		 = out_pixel.packed;
						context.shadow_queue.rays[slot]			 = out_ray;
						context.shadow_queue.weights[slot]		 = cugar::Vector4f(out_w, w.w);
						context.shadow_queue.light_path_id[slot] = light_path_id | ((light_depth + 1) << 24) | ((ev.depth + 2) << 28); // NOTE: light_depth + 1 represents the technique 's, i.e. the number of light subpath vertices

						//sinked_path = true;
					}
				}
			}
			else
			{
				const uint32 eye_to_light_paths = n_eye_paths / n_light_paths;

				const uint32 light_path_id =
					(n_light_paths == n_eye_paths) ? pixel_info.pixel : // use the same light subpath index as this eye subpath
					pixel_info.pixel / eye_to_light_paths; // pick one at random

				// compute the maximum depth a light vertex might have
				const int32 n_light_vertices = context.light_vertices.vertex_counts[light_path_id];

				// perform a bidirectional connection for each light vertex
				for (uint32 light_depth = config.direct_lighting_nee ? 0 : 1;
							light_depth < cugar::min(n_light_vertices, max_light_depth + 1);
							light_depth++)
				{
					const float light_weight = 1.0f;

					const uint32 light_idx = light_path_id + light_depth * n_light_paths;

					// setup a light vertex
					cugar::Vector4f	 light_pos		= context.light_vertices.vertex_pos[light_idx];
					const uint2		 light_in		= context.light_vertices.vertex_input[light_idx];
					uint4            light_gbuffer	= context.light_vertices.vertex_gbuffer[light_idx];
					PathWeights      light_weights	= context.light_vertices.vertex_weights[light_idx];

					// setup the light vertex
					LightVertex lv;
					lv.setup(light_pos, light_in, light_gbuffer, light_weights, light_depth, renderer);

					// evaluate the connection
					cugar::Vector3f out;
					cugar::Vector3f out_w;
					float			d;

					eval_connection(ev, lv, out, out_w, d, config.use_rr, config.direct_lighting_nee, config.direct_lighting_bsdf);

					// multiply by the light vertex weight
					out_w *= light_weight;

					if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
					{
						// recompute d for intersection calculations
						if (SHADOW_BIAS) d = cugar::length(lv.geom.position - (ev.geom.position + ev.in * SHADOW_BIAS));

						// enqueue the output ray
						Ray out_ray;
						//out_ray.origin	= ev.geom.position + ev.in * SHADOW_BIAS; // move the origin slightly towards the viewer
						//out_ray.dir		= out;
						//out_ray.tmin	= SHADOW_TMIN;
						//out_ray.tmax	= d * 0.9999f;
						out_ray.origin	= ev.geom.position + ev.in * SHADOW_BIAS; // shift back in space along the viewing direction
						out_ray.dir		= (lv.geom.position - out_ray.origin); //out;
						out_ray.tmin	= SHADOW_TMIN;
						out_ray.tmax	= 0.9999f;

						const PixelInfo out_pixel = context.in_bounce ?
							pixel_info :										// if this sample is a secondary bounce, use the previously selected channel
							PixelInfo(pixel_info.pixel, FBufferDesc::DIRECT_C);	// otherwise (i.e. this is the first bounce) choose the direct-lighting output channel

						const uint32 slot = context.shadow_queue.append_slot();

						context.shadow_queue.pixels[slot]			= out_pixel.packed;
						context.shadow_queue.rays[slot]				= out_ray;
						context.shadow_queue.weights[slot]			= cugar::Vector4f(out_w, w.w);
						context.shadow_queue.light_path_id[slot]	= light_path_id | ((light_depth + 1) << 24) | ((ev.depth + 2) << 28); // NOTE: light_depth + 1 represents the technique 's, i.e. the number of light subpath vertices

						//sinked_path = true;
					}
				}
			}
		}

		// accumulate the emissive component along the incoming direction
		if (config.accumulate_emissive(eye_path_id, context.in_bounce + 2, absorbed))
		{
			cugar::Vector3f out_w = eval_incoming_emission(ev, renderer, config.direct_lighting_nee, config.indirect_lighting_nee, config.use_vpls);

			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				sample_sink.sink(
					pixel_info.channel,
					cugar::Vector4f(out_w, w.w),
					0,
					pixel_info.pixel,
					0,
					context.in_bounce + 2,
					context,
					renderer);

				//sinked_path = true;
			}
		}
	}
	else
	{
		// hit the environment - perform sky lighting
	}

	//if (sinked_path == false)
	//	sample_sink.null_path(eye_path_id, context, renderer);
}

///
/// This function connects the given light vertex to the camera.
/// Valid connections with non-zero contribution get enqueued in the shadow queue for occlusion testing.
///
/// \param light_idx			the index of the light vertex in context.light_vertices
/// \param n_light_paths		the total number of generated light subpaths
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
/// \param config				the bidirectional path tracing configuration policy
///
template <typename TBPTContext, typename TBPTConfig>
FERMAT_HOST_DEVICE
void connect_to_camera(const uint32 light_idx, const uint32 n_light_paths, TBPTContext& context, RendererView& renderer, const TBPTConfig& config)
{
	// pure light tracing: connect with a light vertex chosen at random
	{
		const uint32 light_depth = context.light_vertices.vertex_path_id[ light_idx ] >> 24;
		const float light_weight = 1.0f / float(n_light_paths);

		const uint2  light_in = context.light_vertices.vertex_input[light_idx];

		//VertexGeometryId light_vertex = context.light_vertices.vertex[light_idx];
		VertexGeometry   light_vertex_geom;
		cugar::Vector4f	 light_pos		= context.light_vertices.vertex_pos[light_idx];
		uint4            light_gbuffer	= context.light_vertices.vertex_gbuffer[light_idx];
		cugar::Vector3f  light_in_dir	= unpack_direction(light_in.x);
		cugar::Vector3f  light_in_alpha = cugar::from_rgbe(light_in.y);
		PathWeights      light_weights	= context.light_vertices.vertex_weights[light_idx];
		//float			 light_pdf		= cugar::binary_cast<float>(light_gbuffer.w);

	#if 0
		// evaluate the differential geometry at the light vertex (TODO: replace using pre-encoded position and normal (and later on tangents?))
		setup_differential_geometry(renderer.mesh, light_vertex.prim_id, light_vertex.uv.x, light_vertex.uv.y, &light_vertex_geom);
	#else
		light_vertex_geom.position = light_pos.xyz();
		light_vertex_geom.normal_s = unpack_direction(cugar::binary_cast<uint32>(light_pos.w));
		light_vertex_geom.normal_g = light_vertex_geom.normal_s;
		light_vertex_geom.tangent  = cugar::orthogonal(light_vertex_geom.normal_s);
		light_vertex_geom.binormal = cugar::cross(light_vertex_geom.normal_s, light_vertex_geom.tangent);
	#endif

		// start evaluating the geometric term
		const float d2 = fmaxf(1.0e-8f, cugar::square_length(light_vertex_geom.position - cugar::Vector3f(renderer.camera.eye)));
		const float d = sqrtf(d2);

		// join the light sample with the current vertex
		const cugar::Vector3f out = (light_vertex_geom.position - cugar::Vector3f(renderer.camera.eye)) / d;

		// evaluate the geometric term
		const float cos_theta = cugar::dot(out, context.camera_W) / context.camera_W_len;
		const float G = fabsf(cos_theta * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

		// evaluate the camera BSDF
		float out_x;
		float out_y;

		const float p_s = camera_direction_pdf(context.camera_U, context.camera_V, context.camera_W, context.camera_W_len, context.camera_square_focal_length, out, &out_x, &out_y);
		const float f_s = p_s * float(renderer.res_x * renderer.res_y);

		if (f_s)
		{
			cugar::Vector4f out_w;

			if (light_depth == 0) // this is a primary VPL / light vertex
			{
				if (0) // visible lights (a very silly strategy)
				{
					// build the local BSDF (EDF)
					//Edf light_bsdf(light_material);
					Edf light_bsdf(cugar::from_rgbe(light_gbuffer.x));

					// evaluate the light's EDF and the surface BSDF
					const cugar::Vector3f f_L = light_bsdf.f(light_vertex_geom, light_vertex_geom.position, -out);
					const float           p_L = light_bsdf.p(light_vertex_geom, light_vertex_geom.position, -out, cugar::kProjectedSolidAngle);

					const float pGp = p_s * G * p_L;
					const float next_pGp = p_L * light_weights.pG;
					const float mis_w =
						(config.visible_lights == 0) ? 1.0f :
						bpt_mis(pGp / (/*n_light_paths * */config.light_tracing), next_pGp, light_weights.pGp_sum);

					// calculate the cumulative sample weight, equal to f_L * f_s * G / p
					out_w = cugar::Vector4f(light_in_alpha * f_L * f_s * G * mis_w, 1.0f) * light_weight;
				}
				else
					out_w = cugar::Vector4f(0.0f);
			}
			else
			{
				// build the local BSDF
				//Bsdf light_bsdf(light_material);
				Bsdf light_bsdf = unpack_bsdf(renderer, light_gbuffer );

				// evaluate the light's EDF and the surface BSDF
				const cugar::Vector3f f_L = light_bsdf.f(light_vertex_geom, light_in_dir, -out);
				const float           p_L = light_bsdf.p(light_vertex_geom, light_in_dir, -out, cugar::kProjectedSolidAngle);

				const float pGp = p_s * G * p_L;
				const float next_pGp = cugar::max_comp(f_L) * light_weights.pG;
				const float mis_w =
					(light_depth == 1 &&
						config.direct_lighting_nee == 0 &&
						config.direct_lighting_bsdf == 0) ? 1.0f :
						(light_depth > 1 &&
							config.indirect_lighting_nee == 0 &&
							config.indirect_lighting_bsdf == 0) ? 1.0f :
					bpt_mis(pGp / (/*n_light_paths * */config.light_tracing), next_pGp, light_weights.pGp_sum);

				// calculate the cumulative sample weight, equal to f_L * f_s * G / p
				out_w = cugar::Vector4f(light_in_alpha * f_L * f_s * G * mis_w, 1.0f) * light_weight;
			}

			if (cugar::max_comp(out_w.xyz()) > 0.0f && cugar::is_finite(out_w.xyz()))
			{
				// enqueue the output ray
				Ray out_ray;
				out_ray.origin = renderer.camera.eye;
				out_ray.dir = out;
				out_ray.tmin = SHADOW_TMIN;
				out_ray.tmax = d * 0.9999f;

				// compute the pixel index
				const PixelInfo out_pixel = PixelInfo(
					cugar::quantize(out_x*0.5f + 0.5f, renderer.res_x) +
					cugar::quantize(out_y*0.5f + 0.5f, renderer.res_y) * renderer.res_x,
					FBufferDesc::DIRECT_C);

				context.shadow_queue.append(out_pixel, out_ray, out_w, 1.0f);
			}
		}
	}
}

///
/// Resolve the occlusion for the specified entry in the shadow queue, and pass the resulting sample to the sink.
/// Specifically, if the queue entry contains a (ray tracing) hit, the sample will be considered shadowed and its contribution will be set to zero.
/// Otherwise, the sample's contribution will be left unmodified.
///
/// \param queue_idx			the index of the queue entry
/// \param sample_sink			the output sample sink, processing all bidirectional paths formed processing this vertex
/// \param context				the bidirectional path tracing context
/// \param renderer				the rendering context
///
template <typename TSampleSink, typename TBPTContext>
FERMAT_HOST_DEVICE
void solve_occlusion(const uint32 queue_idx, TSampleSink& sample_sink, TBPTContext& context, RendererView& renderer)
{
	const PixelInfo		  pixel_info	= context.shadow_queue.pixels[queue_idx];
	const Hit			  hit			= context.shadow_queue.hits[queue_idx];
	const cugar::Vector4f w				= context.shadow_queue.weights[queue_idx];
	const uint32		  light_path_id = context.shadow_queue.light_path_id[queue_idx];

	// TODO: break this up in separate diffuse and specular components
	const float vis = (hit.t < 0.0f) ? 1.0f : 0.0f;

	const uint32 s = (light_path_id >> 24) & 0xF;
	const uint32 t = (light_path_id >> 28) & 0xF;

	sample_sink.sink(pixel_info.channel, w * vis, light_path_id & 0xFFFFFF, pixel_info.pixel, s, t, context, renderer);
}

template <typename TBPTContext, typename TBPTConfig>
__global__
void light_tracing_kernel(const uint32 n_light_paths, TBPTContext context, RendererView renderer, TBPTConfig config)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering)
	{
		// compute the maximum depth a VPL/photon light might have
		const int32 max_light_depth = config.max_path_length - 1;

		const uint32 n_light_vertices = context.light_vertices.vertex_counts[max_light_depth];

		const uint32 light_idx = thread_id;;

		if (light_idx < n_light_vertices)
			connect_to_camera(light_idx, n_light_paths, context, renderer, config);
	}
	else
	{
		const uint32 light_path_id = thread_id;

		if (light_path_id < n_light_paths)
		{
			const uint32 vertex_count = context.light_vertices.vertex_counts[light_path_id];
			for (uint32 i = 0; i < vertex_count; ++i)
				connect_to_camera(light_path_id + i * n_light_paths, n_light_paths, context, renderer, config);
		}
	}
}

template <typename TBPTContext, typename TBPTConfig>
void light_tracing(const uint32 n_light_paths, TBPTContext& context, RendererView& renderer, TBPTConfig& config)
{
	uint32 n_threads;

	if (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering)
	{
		// compute the maximum depth a VPL/photon light might have
		const int32 max_light_depth = config.max_path_length - 1;

		cudaMemcpy(&n_threads, &context.light_vertices.vertex_counts[max_light_depth], sizeof(uint32), cudaMemcpyDeviceToHost);
	}
	else
		n_threads = n_light_paths;
		
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(n_threads, blockSize));

	light_tracing_kernel << < gridSize, blockSize >> > (n_light_paths, context, renderer, config);
}

template <typename TSampleSink, typename TBPTContext>
__global__
void solve_occlusions_kernel(const uint32 in_queue_size, TSampleSink sample_sink, TBPTContext context, RendererView renderer)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.shadow_queue.size
		solve_occlusion(thread_id, sample_sink, context, renderer);
}

template <typename TSampleSink, typename TBPTContext>
void solve_occlusions(const uint32 in_queue_size, TSampleSink sample_sink, TBPTContext context, RendererView renderer)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
	solve_occlusions_kernel << < gridSize, blockSize >> > (in_queue_size, sample_sink, context, renderer);
}

template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
__global__
void generate_primary_light_vertices_kernel(const uint32 n_light_paths, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 light_path_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (light_path_id < n_light_paths)
		generate_primary_light_vertex(light_path_id, n_light_paths, primary_coords, context, renderer, config);
}

template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
void generate_primary_light_vertices(const uint32 n_light_paths, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(n_light_paths, blockSize));
	generate_primary_light_vertices_kernel << < gridSize, blockSize >> > (n_light_paths, primary_coords, context, renderer, config);

	// update the per-level cumulative vertex counts
	if (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering)
		cudaMemcpy(context.light_vertices.vertex_counts, context.light_vertices.vertex_counter, sizeof(uint32), cudaMemcpyDeviceToDevice);
}


template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
__global__
__launch_bounds__(SECONDARY_LIGHT_VERTICES_BLOCKSIZE, SECONDARY_LIGHT_VERTICES_CTA_BLOCKS)
void process_secondary_light_vertices_kernel(const uint32 in_queue_size, const uint32 n_light_paths, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.in_queue.size
		process_secondary_light_vertex(thread_id, n_light_paths, primary_coords, context, renderer, config);
}

template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
void process_secondary_light_vertices(const uint32 in_queue_size, const uint32 n_light_paths, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 blockSize(SECONDARY_LIGHT_VERTICES_BLOCKSIZE);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
	process_secondary_light_vertices_kernel << < gridSize, blockSize >> > (in_queue_size, n_light_paths, primary_coords, context, renderer, config);

	// update the per-level cumulative vertex counts
	if (VertexOrdering(config.light_ordering) == VertexOrdering::kRandomOrdering)
		cudaMemcpy(context.light_vertices.vertex_counts + context.in_bounce + 1, context.light_vertices.vertex_counter, sizeof(uint32), cudaMemcpyDeviceToDevice);
}


template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
__global__
void generate_primary_eye_vertices_kernel(const uint32 n_eye_paths, const uint32 n_light_paths, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 eye_path_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (eye_path_id < n_eye_paths)
		generate_primary_eye_vertex(eye_path_id, n_eye_paths, n_light_paths, primary_coords, context, renderer, config);
}

template <typename TPrimaryCoordinates, typename TBPTConfig>
void generate_primary_eye_vertices(const uint32 n_eye_paths, const uint32 n_light_paths, TPrimaryCoordinates primary_coords, BPTContextBase context, RendererView renderer, const TBPTConfig config)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(n_eye_paths, blockSize));
	generate_primary_eye_vertices_kernel << < gridSize, blockSize >> > (n_eye_paths, n_light_paths, primary_coords, context, renderer, config);
}

template <typename TSampleSink, typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
__global__
__launch_bounds__(SECONDARY_EYE_VERTICES_BLOCKSIZE, SECONDARY_EYE_VERTICES_CTA_BLOCKS)
void process_secondary_eye_vertices_kernel(const uint32 in_queue_size, const uint32 n_eye_paths, const uint32 n_light_paths, TSampleSink sink, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id < in_queue_size) // *context.in_queue.size
		process_secondary_eye_vertex(thread_id, n_eye_paths, n_light_paths, sink, primary_coords, context, renderer, config);
}

template <typename TSampleSink, typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
void process_secondary_eye_vertices(const uint32 in_queue_size, const uint32 n_eye_paths, const uint32 n_light_paths, TSampleSink sink, TPrimaryCoordinates primary_coords, TBPTContext context, RendererView renderer, const TBPTConfig config)
{
	const uint32 blockSize(SECONDARY_EYE_VERTICES_BLOCKSIZE);
	const dim3 gridSize(cugar::divide_ri(in_queue_size, blockSize));
	process_secondary_eye_vertices_kernel << < gridSize, blockSize >> > (in_queue_size, n_eye_paths, n_light_paths, sink, primary_coords, context, renderer, config);
}

///@} BPTLib
///@} Fermat

} // namespace bpt