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

#include <pathtracer.h>
#include <tiled_sequence.h>
#include <bsdf.h>
#include <edf.h>
#include <mis_utils.h>
#include <bpt_utils.h>
#include <eaw.h>
#include <direct_lighting_mesh.h>
#include <direct_lighting_rl.h>

/// \page PTLibPage PTLib
/// Top: \ref OvertureContentsPage
///
/// \ref PTLib is a flexible path tracing library, thought to be as performant as possible and yet vastly configurable at compile-time.
/// The module is organized into a host library of parallel kernels, \ref PTLib, and a core module of device-side functions, \ref PTLibCore.
/// The latter provides functions to generate primary rays, process path vertices, sample Next-Event Estimation and emissive surface hits
/// at each of them, and process all the generated samples.
/// In order to make the whole process configurable, all the functions accept three template interfaces:
///\par
/// \anchor TPTContext
/// 1. a context interface, holding members describing the current path tracer state, 
///    and providing two trace methods, for scattering and shadow rays respectively.
///    The basic path tracer state can be inherited from the \ref PTContextBase class.
///    On top of that, this class has to provide the following interface:
///\n
///\code
///  struct TPTContext
///  {
///  	FERMAT_DEVICE
///  	void trace_ray(
///  		TPTVertexProcessor&		vertex_processor, 
///  		RenderingContextView&	renderer,
///  		const PixelInfo			pixel,
///  		const MaskedRay			ray,
///  		const cugar::Vector4f	weight,
///  		const cugar::Vector2f	cone,
///  		const uint32			nee_vertex_id);
///  
///  	FERMAT_DEVICE
///  	void trace_shadow_ray(
///  		TPTVertexProcessor&		vertex_processor,
///  		RenderingContextView&	renderer,
///  		const PixelInfo			pixel,
///  		const MaskedRay			ray,
///  		const cugar::Vector3f	weight,
///  		const cugar::Vector3f	weight_d,
///  		const cugar::Vector3f	weight_g,
///  		const uint32			nee_vertex_id,
///  		const uint32			nee_sample_id);
///  };
///\endcode
///    Note that for the purpose of the \ref PTLibCore module, a given implementation is free to define
///    the trace methods in any arbitrary manner, since the result of tracing a ray is not used directly.
///    This topic will be covered in more detail later on.
///\n
///\n
/// \anchor TPTVertexProcessor
/// 2. a user defined vertex processor, determining what to do with each generated path vertex;
///    this is the class responsible for weighting each sample and accumulating them to the image.
///    It has to provide the following interface:
///\n
///\code
///  struct TPTVertexProcessor
///  {
///  	// preprocess a vertex and return some packed vertex info - this is useful since this
///  	// information bit is automatically propagated through the entire path tracing pipeline,
///  	// and might be used, for example, to implement user defined caching strategies like
///  	// those employed in path space filtering, where the packed info would e.g. encode a
///  	// spatial hash.
///  	//
///  	// \param context				the current context
///  	// \param renderer				the current renderer
///  	// \param pixel_info			packed pixel info
///  	// \param ev					the current vertex
///  	// \param cone_radius			the current cone radius
///  	// \param scene_bbox			the scene bounding box
///  	// \param prev_vertex_info		the vertex info at the previous path vertex
///  	FERMAT_DEVICE
///  	uint32 preprocess_vertex(
///  		const TPTContext&			context,
///  		const RenderingContextView&	renderer,
///  		const PixelInfo				pixel_info,
///  		const EyeVertex&			ev,
///  		const float					cone_radius,
///  		const cugar::Bbox3f			scene_bbox,
///  		const uint32				prev_vertex_info);
///  
///  	// compute NEE weights given a vertex and a light sample
///  	//
///  	// \param context				the current context
///  	// \param renderer				the current renderer
///  	// \param pixel_info			packed pixel info
///  	// \param prev_vertex_info		packed vertex info at the previous path vertex
///  	// \param vertex_info			packed vertex info
///  	// \param ev					the current vertex
///  	// \param f_d					the diffuse brdf
///  	// \param f_g					the glossy brdf
///  	// \param w						the current path weight
///  	// \param f_L					the current sample contribution, including the MIS weight
///  	// \param out_w_d				the output diffuse weight
///  	// \param out_w_g				the output glossy weight
///  	// \param out_vertex_info		the output packed vertex info
///  	FERMAT_DEVICE
///  	void compute_nee_weights(
///  		const TPTContext&			context,
///  		const RenderingContextView&	renderer,
///  		const PixelInfo				pixel_info,
///  		const uint32				prev_vertex_info,
///  		const uint32				vertex_info,
///  		const EyeVertex&			ev,
///  		const cugar::Vector3f&		f_d,
///  		const cugar::Vector3f&		f_g,
///  		const cugar::Vector3f&		w,
///  		const cugar::Vector3f&		f_L,
///  			  cugar::Vector3f&		out_w_d,
///  			  cugar::Vector3f&		out_w_g,
///  			  uint32&				out_vertex_info);
///  
///  	// compute scattering weights given a vertex
///  	//
///  	// \param context				the current context
///  	// \param renderer				the current renderer
///  	// \param pixel_info			packed pixel info
///  	// \param prev_vertex_info		packed vertex info at the previous path vertex
///  	// \param vertex_info			packed vertex info
///  	// \param ev					the current vertex
///  	// \param out_comp				the brdf scattering component
///  	// \param g						the brdf scattering weight (= f/p)
///  	// \param w						the current path weight
///  	// \param out_w					the output weight
///  	// \param out_vertex_info		the output vertex info
///  	//
///  	FERMAT_DEVICE
///  	void compute_scattering_weights(
///  		const TPTContext&			context,
///  		const RenderingContextView&	renderer,
///  		const PixelInfo				pixel_info,
///  		const uint32				prev_vertex_info,
///  		const uint32				vertex_info,
///  		const EyeVertex&			ev,
///  		const uint32				out_comp,
///  		const cugar::Vector3f&		g,
///  		const cugar::Vector3f&		w,
///  			  cugar::Vector3f&		out_w,
///  			  uint32&				out_vertex_info);
///  
///  	// accumulate an emissive surface hit
///  	//
///  	// \param context				the current context
///  	// \param renderer				the current renderer
///  	// \param pixel_info			packed pixel info
///  	// \param prev_vertex_info		packed vertex info at the previous path vertex
///  	// \param vertex_info			packed vertex info
///  	// \param ev					the current vertex
///  	// \param w						the emissive sample weight
///  	//
///  	FERMAT_DEVICE
///  	void accumulate_emissive(
///  		const TPTContext&			context,
///  			  RenderingContextView&	renderer,
///  		const PixelInfo				pixel_info,
///  		const uint32				prev_vertex_info,
///  		const uint32				vertex_info,
///  		const EyeVertex&			ev,
///  		const cugar::Vector3f&		w);
///  
///  
///  	// accumulate a NEE sample
///  	//
///  	// \param context				the current context
///  	// \param renderer				the current renderer
///  	// \param pixel_info			packed pixel info
///  	// \param vertex_info			packed vertex info
///  	// \param hit					the hit information
///  	// \param w_d					the diffuse nee weight
///  	// \param w_g					the glossy nee weight
///  	//
///  	FERMAT_DEVICE
///  	void accumulate_nee(
///  		const TPTContext&			context,
///  			  RenderingContextView&	renderer,
///  		const PixelInfo				pixel_info,
///  		const uint32				vertex_info,
///  		const bool					shadow_hit,
///  		const cugar::Vector3f&		w_d,
///  		const cugar::Vector3f&		w_g);
///  };
///\endcode
///\n
/// \anchor TPTDirectLightingSampler
/// 3. a user defined direct lighting engine, responsible to generate NEE samples.
///    It has to provide the following interface:
///\n
///\code
///  struct TDirectLightingSampler
///  {
///  	// preprocess a path vertex and return a user defined hash integer key,
///  	// called <i>nee_vertex_id</i>,
///  	// used for all subsequent NEE computations at this vertex;
///  	// this packed integer is useful to implement things like spatial hashing, where the current
///  	// vertex is hashed to a slot in a hash table, e.g. storing reinforcement-learning data.
///  	//
///  	FERMAT_DEVICE
///  	uint32 preprocess_vertex(
///  		const RenderingContextView&	renderer,
///  		const EyeVertex&	ev,
///  		const uint32		pixel,
///  		const uint32		bounce,
///  		const bool			is_secondary_diffuse,
///  		const float			cone_radius,
///  		const cugar::Bbox3f	scene_bbox);
///  
///  	// sample a light vertex at a given slot, and return a user defined sample index,
///  	// called <i>nee_sample_id</i>: this integer may encode any arbitrary data the sampler
///  	// might need to later on address the generated sample in the update() method.
///  	//
///  	FERMAT_DEVICE
///  	uint32 sample(
///  		const uint32		nee_vertex_id,
///  		const float			z[3],
///  		VertexGeometryId*	light_vertex,
///  		VertexGeometry*		light_vertex_geom,
///  		float*				light_pdf,
///  		Edf*				light_edf);
///  
///  	// map a light vertex defined by its triangle id and uv barycentric coordinates to the
///  	// corresponding differential geometry, computing its EDF and sampling PDF, using the
///  	// slot information computed at the previous vertex along the path; this method is called
///  	// on emissive surface hits to figure out the PDF with which the hits would have been
///  	// generated by NEE.
///  	//
///  	FERMAT_DEVICE
///  	void map(
///  		const uint32			prev_nee_vertex_id,
///  		const uint32			triId,
///  		const cugar::Vector2f	uv,
///  		const VertexGeometry	light_vertex_geom,
///  		float*					light_pdf,
///  		Edf*					light_edf);
///  
///  	// update the internal state of the sampler with the resulting NEE sample
///  	// information, useful to e.g. implement reinforcement-learning strategies.
///  	//
///  	FERMAT_DEVICE
///  	void update(
///  		const uint32			nee_vertex_id,
///  		const uint32			nee_sample_id,
///  		const cugar::Vector3f	w,
///  		const bool				occluded);
///  };
///\endcode
///\par
/// You'll notice this is just a slight generalization of the Light interface, providing more controls for
/// preprocessing and updating some per-vertex information.
/// At the moment, Fermat provides two different implementations of this interface:
///\n
/// - \ref DirectLightingMesh : a simple wrapper around the MeshLight class;
/// - \ref DirectLightingRL : a more advanced sampler based on <a href="https://en.wikipedia.org/wiki/Reinforcement_learning">Reinforcement Learning</a>.
/// 
///\par
/// The most important functions implemented by the \ref PTLibCore module are:
///\n
///\code
///  // generate a primary ray based on the given pixel index
///  //
///  // \tparam TPTContext				A path tracing context
///  //
///  // \param context                  the path tracing context
///  // \param renderer                 the rendering context
///  // \param pixel                    the unpacked 2d pixel index
///  // \param U                        the horizontal (+X) camera frame vector
///  // \param V                        the vertical (+Y) camera frame vector
///  // \param W                        the depth (+Z) camera frame vector
///  template <typename TPTContext>
///  FERMAT_DEVICE
///  MaskedRay generate_primary_ray(
///  	TPTContext&				context,
///  	RenderingContextView&	renderer,
///  	const uint2				pixel,
///  	cugar::Vector3f			U,
///  	cugar::Vector3f			V,
///  	cugar::Vector3f			W);
///
///  // processes a NEE sample, using already computed occlusion information
///  //
///  // \tparam TPTContext			A path tracing context, which must adhere to the TPTContext interface
///  // \tparam TPTVertexProcessor	A vertex processor, which must adhere to the TPTVertexProcessor interface
///  //
///  // \param context                  the path tracing context
///  // \param vertex_processor         the vertex processor
///  // \param renderer                 the rendering context
///  // \param shadow_hit               a bit indicating whether the sample is occluded or not
///  // \param pixel_info               the packed pixel info
///  // \param w                        the total sample weight
///  // \param w_d                      the diffuse sample weight
///  // \param w_g                      the glossy sample weight
///  // \param vertex_info              the current vertex info produced by the vertex processor
///  // \param nee_vertex_id            the current NEE slot computed by the direct lighting sampler
///  // \param nee_sample_id            the current NEE sample info computed by the direct lighting sampler
///  template <typename TPTContext, typename TPTVertexProcessor>
///  FERMAT_DEVICE
///  void solve_occlusion(
///  	TPTContext&				context,
///  	TPTVertexProcessor&		vertex_processor,
///  	RenderingContextView&	renderer,
///  	const bool				shadow_hit,
///  	const PixelInfo			pixel_info,
///  	const cugar::Vector3f	w,
///  	const cugar::Vector3f	w_d,
///  	const cugar::Vector3f	w_g
///  	const uint32			vertex_info    = uint32(-1),
///  	const uint32			nee_vertex_id  = uint32(-1),
///  	const uint32			nee_sample_id  = uint32(-1));
///
///  // processes a path vertex, performing these three key steps:
///  // - sampling NEE
///  // - accumulating emissive surface hits
///  // - scattering
///  //
///  // \tparam TPTContext			A path tracing context, which must adhere to the TPTContext interface
///  // \tparam TPTVertexProcessor	A vertex processor, which must adhere to the TPTVertexProcessor interface
///  //
///  // \param context                  the path tracing context
///  // \param vertex_processor         the vertex processor
///  // \param renderer                 the rendering context
///  // \param pixel_info               the packed pixel info
///  // \param pixel                    the unpacked 2d pixel index
///  // \param ray                      the incoming ray
///  // \param hit                      the hit information
///  // \param w                        the current path weight
///  // \param prev_vertex_info         the vertex info produced by the vertex processor at the previous vertex
///  // \param prev_nee_vertex_id       the NEE slot corresponding to the previous vertex
///  // \param cone                     the incoming ray cone
///  template <typename TPTContext, typename TPTVertexProcessor>
///  FERMAT_DEVICE
///  bool shade_vertex(
///  	TPTContext&				context,
///  	TPTVertexProcessor&		vertex_processor,
///  	RenderingContextView&	renderer,
///  	const uint32			bounce,
///  	const PixelInfo			pixel_info,
///  	const uint2				pixel,
///  	const MaskedRay&		ray,
///  	const Hit				hit,
///  	const cugar::Vector4f	w,
///  	const uint32			prev_vertex_info = uint32(-1),
///  	const uint32			prev_nee_vertex_id    = uint32(-1),
///  	const cugar::Vector2f	cone             = cugar::Vector2f(0));
///\endcode
///\par
/// Note that all of these are device-side functions meant to be called by individual CUDA threads.
/// The underlying idea is that all of them might call into \ref TPTContext trace() methods, but that the implementation
/// of the \ref TPTContext class might decide whether to perform the trace calls in-place, or rather enqueue them.
/// The latter approach is called "wavefront" scheduling, and is the one favored in Fermat, as so far it has proven most
/// efficient.
///
///\section PTWavefrontSchedulingSection	Wavefront Scheduling
///\par
/// \ref PTLib implements a series of kernels to execute all of the above functions in massively parallel waves,
/// assuming their inputs can be fetched from some \ref TPTContext -defined queues.
/// In order for these wavefronts kernels to work, it is sufficient to have the \ref TPTContext implementation inherit from
/// the prepackaged \ref PTContextQueues class, containing all necessary queue storage.
/// Together with the kernels themselves, the corresponding host dispatch functions are provided as well.
/// These are the following:
///
///\code
///  	// dispatch the kernel to generate primary rays: the number of rays is defined by the resolution
///  	// parameters provided by the rendering context.
///  	//
///  	// \tparam TPTContext			A path tracing context
///  	//
///  	// \param context				the path tracing context
///  	// \param renderer				the rendering context
///  	template <typename TPTContext>
///  	void generate_primary_rays(
///  		TPTContext				context,
///  		RenderingContextView	renderer);
///
///  	// dispatch the shade hits kernel
///  	//
///  	// \tparam TPTContext			A path tracing context, which must adhere to the TPTContext interface
///  	// \tparam TPTVertexProcessor	A vertex processor, which must adhere to the TPTVertexProcessor interface
///  	//
///  	// \param in_queue_size			the size of the input queue containing all path vertices in the current wave
///  	// \param context				the path tracing context
///  	// \param vertex_processor		the vertex_processor
///  	// \param renderer				the rendering context
///  	template <typename TPTContext, typename TPTVertexProcessor>
///  	void shade_hits(
///  		const uint32			in_queue_size,
///  		TPTContext				context,
///  		TPTVertexProcessor		vertex_processor,
///  		RenderingContextView	renderer);
/// 
///  	// dispatch a kernel to process NEE samples using computed occlusion information
///  	//
///  	// \tparam TPTContext			A path tracing context, which must adhere to the TPTContext interface
///  	// \tparam TPTVertexProcessor	A vertex processor, which must adhere to the TPTVertexProcessor interface
///  	//
///  	// \param in_queue_size			the size of the input queue containing all processed NEE samples
///  	// \param context				the path tracing context
///  	// \param vertex_processor		the vertex_processor
///  	// \param renderer				the rendering context
///  	template <typename TPTContext, typename TPTVertexProcessor>
///  	void solve_occlusion(
///  		const uint32			in_queue_size,
///  		TPTContext				context,
///  		TPTVertexProcessor		vertex_processor,
///  		RenderingContextView	renderer);
///\endcode
///\anchor path_trace_loop_anchor
///\par
/// The last key function provided by this module is the one assembling all of the above into a single loop, or rather, into a complete
/// pipeline that generates primary rays, traces them, shades the resulting vertices, traces any generated shadow and scattering rays,
/// shades the results, and so, and so on, until all generated paths are terminated:
///
///\code
///  	// main path tracing loop
///  	//
///  	template <typename TPTContext, typename TPTVertexProcessor>
///  	void path_trace_loop(
///  		TPTContext&				context,
///  		TPTVertexProcessor&		vertex_processor,
///  		RenderingContext&		renderer,
///  		RenderingContextView&	renderer_view,
///  		PTLoopStats&			stats);
///\endcode
///
///\par
/// In the next chapter we'll see how all of this can be used to write a very compact path tracer.
///
/// <img src="staircase2-sunset.jpg" style="position:relative; bottom:-10px; border:0px; width:740px;"/>
///\par
/// <small>'Modern Hall, at sunset', based on a <a href="http://www.blendswap.com/blends/view/51997">model</a> by <i>NewSee2l035</i></small>
///\n
///
///
/// Next: \ref PTPage

#define MIS_HEURISTIC	POWER_HEURISTIC

#define VTL_RL_HASH_SIZE	(512u * 1024u)

#if !defined(DEVICE_TIMING) || (DEVICE_TIMING == 0)
#define DEVICE_TIME(x)
#else
#define DEVICE_TIME(x) x
#endif

enum PTDeviceTimers
{
	SETUP_TIME					= 0,
	BRDF_EVAL_TIME				= 1,
	DIRLIGHT_SAMPLE_TIME		= 2,
	DIRLIGHT_EVAL_TIME			= 3,
	LIGHTS_PREPROCESS_TIME		= 4,
	LIGHTS_SAMPLE_TIME			= 5,
	LIGHTS_EVAL_TIME			= 6,
	LIGHTS_MAPPING_TIME			= 7,
	LIGHTS_UPDATE_TIME			= 8,
	TRACE_SHADOW_TIME			= 9,
	TRACE_SHADED_TIME			= 10,
	BRDF_SAMPLE_TIME			= 11,
	FBUFFER_WRITES_TIME			= 12,
	PREPROCESS_VERTEX_TIME		= 13,
	NEE_WEIGHTS_TIME			= 14,
	SCATTERING_WEIGHTS_TIME		= 15,
	TOTAL_TIME					= 16
};

// a simple device-side timer
// 
struct DeviceTimer
{
	FERMAT_DEVICE void start() { last = clock64(); }
	FERMAT_DEVICE void restart() { last = clock64(); }
	FERMAT_DEVICE uint64 take() { int64 first = last; last = clock64(); return uint64( last - first ); }

	int64 last;
};

///@addtogroup Fermat
///@{

///@addtogroup PTLib
///@{

///@defgroup PTLibCore PTLibCore
/// A module defining core path tracing functions to process path vertices, perform NEE and process its generated samples
///@{

/// Store packed pixel information, including:
/// - the pixel index
/// - the current path type (diffuse or glossy)
/// - a bit indicating whether the path went through a diffuse bounce
///
union PixelInfo
{
	FERMAT_HOST_DEVICE PixelInfo() {}
	FERMAT_HOST_DEVICE PixelInfo(const uint32 _packed) : packed(_packed) {}
	FERMAT_HOST_DEVICE PixelInfo(const uint32 _pixel, const uint32 _comp, const uint32 _diffuse = 0) : pixel(_pixel), comp(_comp), diffuse(_diffuse) {}

	FERMAT_HOST_DEVICE operator uint32() const { return packed; }

	uint32	packed;
	struct
	{
		uint32 pixel : 27;
		uint32 comp  : 4;
		uint32 diffuse : 1;
	};
};

FERMAT_DEVICE FERMAT_FORCEINLINE
void per_warp_atomic_add(uint64* ptr, uint64 val) // NOTE: ptr needs to be the same across the warp!
{
	const unsigned int lane_id = threadIdx.x & 31;

	int pred;
	int mask = __match_all_sync(__activemask(), val, &pred);
	int leader = __ffs(mask) - 1;	// select a leader

	if (lane_id == leader)			// only the leader does the update
		atomicAdd(ptr, val);	
}

/// Base path tracing context class
///
template <typename TPTOptions>
struct PTContextBase
{
	TPTOptions			options;
	TiledSequenceView	sequence;
	float				frame_weight;

	uint32				in_bounce				: 27;
	uint32				do_nee					: 1;
	uint32				do_accumulate_emissive	: 1;
	uint32				do_scatter				: 1;

	cugar::Bbox3f		bbox;

	uint64*				device_timers;
};

//------------------------------------------------------------------------------

/// compute options relative to the current bounce and set them in the \ref TPTContext class:
///\par
/// - TPTContext::do_nee                 :		whether to perform NEE
/// - TPTContext::do_accumulate_emissive :		whether to accumulate emissive surface hits
/// - TPTContext::do_scatter             :		whether to perform scattering
///
template <typename TPTContext>
FERMAT_HOST_DEVICE
void compute_per_bounce_options(
	TPTContext&			context,
	const RenderingContextView&	renderer)
{
	// decide whether to perform next-event estimation
	context.do_nee =
		renderer.mesh_vpls.n_vpls &&
		((context.in_bounce + 2 <= context.options.max_path_length) &&
		((context.in_bounce == 0 && context.options.direct_lighting_nee && context.options.direct_lighting) ||
		 (context.in_bounce >  0 && context.options.indirect_lighting_nee)));

	// decide whether to evaluate and accumulate emissive surfaces
	context.do_accumulate_emissive =
		((context.in_bounce == 0 && context.options.visible_lights) ||
		 (context.in_bounce == 1 && context.options.direct_lighting_bsdf && context.options.direct_lighting) ||
		 (context.in_bounce >  1 && context.options.indirect_lighting_bsdf));

	// compute the number of path vertices we want to generate from the eye
	const uint32 max_path_vertices = context.options.max_path_length +
		((context.options.max_path_length == 2 && context.options.direct_lighting_bsdf) ||
		 (context.options.max_path_length >  2 && context.options.indirect_lighting_bsdf) ? 1 : 0);
		
	// decide whether to perform scattering
	context.do_scatter = (context.in_bounce + 2 < max_path_vertices);
}

//------------------------------------------------------------------------------
/// generate a primary ray based on the given pixel index
///
/// \tparam TPTContext				A path tracing context
///
/// \param context                  the path tracing context
/// \param renderer                 the rendering context
/// \param pixel                    the unpacked 2d pixel index
/// \param U						the horizontal (+X) camera frame vector
/// \param V						the vertical (+Y) camera frame vector
/// \param W						the depth (+Z) camera frame vector
template <typename TPTContext>
FERMAT_DEVICE
MaskedRay generate_primary_ray(
	TPTContext&				context,
	RenderingContextView&	renderer,
	const uint2				pixel,
	cugar::Vector3f			U,
	cugar::Vector3f			V,
	cugar::Vector3f			W)
{
	// use an optimized sampling pattern to rotate a Halton sequence
	const cugar::Vector2f uv(
		context.sequence.sample_2d(pixel.x, pixel.y, 0),
		context.sequence.sample_2d(pixel.x, pixel.y, 1));

	const float2 d = make_float2(
		(pixel.x + uv.x) / float(renderer.res_x),
		(pixel.y + uv.y) / float(renderer.res_y)) * 2.f - 1.f;

	float3 ray_origin	 = renderer.camera.eye;
	float3 ray_direction = d.x*U + d.y*V + W;

	return make_ray( ray_origin, ray_direction, 0u, 1e34f );
}

//------------------------------------------------------------------------------
/// generate a primary ray based on the given pixel index
///
/// \tparam TPTContext				A path tracing context
///
/// \param context                  the path tracing context
/// \param renderer                 the rendering context
/// \param pixel                    the unpacked 2d pixel index
template <typename TPTContext>
FERMAT_DEVICE
MaskedRay generate_primary_ray(
	TPTContext&				context,
	RenderingContextView&	renderer,
	const uint2				pixel)
{
	// use an optimized sampling pattern to rotate a Halton sequence
	const cugar::Vector2f uv(
		context.sequence.sample_2d(pixel.x, pixel.y, 0),
		context.sequence.sample_2d(pixel.x, pixel.y, 1));

	const float2 d = make_float2(
		(pixel.x + uv.x) / float(renderer.res_x),
		(pixel.y + uv.y) / float(renderer.res_y));

	float3 ray_origin	 = renderer.camera.eye;
	float3 ray_direction = renderer.camera_sampler.sample_direction( d );

	return make_ray( ray_origin, ray_direction, 0u, 1e34f );
}

//------------------------------------------------------------------------------

/// processes a NEE sample, using occlusion information
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
/// \param context					the path tracing context
/// \param vertex_processor			the vertex processor
/// \param renderer					the rendering context
/// \param shadow_hit				a bit indicating whether the sample is occluded or not
/// \param pixel_info				the packed pixel info
/// \param w						the total sample weight
/// \param w_d						the diffuse sample weight
/// \param w_g						the glossy sample weight
/// \param nee_vertex_id			the current NEE slot computed by the direct lighting engine
/// \param nee_sample_id			the current NEE cluster computed by the direct lighting engine
template <typename TPTContext, typename TPTVertexProcessor>
FERMAT_DEVICE
void solve_occlusion(
	TPTContext&				context,
	TPTVertexProcessor&		vertex_processor,
	RenderingContextView&	renderer,
	const bool				shadow_hit,
	const PixelInfo			pixel_info,
	const cugar::Vector3f	w,
	const cugar::Vector3f	w_d,
	const cugar::Vector3f	w_g,
	const uint32			vertex_info   = uint32(-1),
	const uint32			nee_vertex_id = uint32(-1),
	const uint32			nee_sample_id = uint32(-1))
{
	DEVICE_TIME( DeviceTimer timer );
	DEVICE_TIME( timer.start() );

	// update the DL sampler state
	context.dl.update( nee_vertex_id, nee_sample_id, w, shadow_hit == true );

	DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_UPDATE_TIME, timer.take() ) );
	
	vertex_processor.accumulate_nee(
		context,
		renderer,
		pixel_info,
		vertex_info,
		shadow_hit,
		w_d,
		w_g );

	DEVICE_TIME( per_warp_atomic_add( context.device_timers + FBUFFER_WRITES_TIME, timer.take() ) );
}

//------------------------------------------------------------------------------

/// return the i-th dimensional sample for this vertex, with i in [0,6[
///
template <typename TPTContext>
FERMAT_DEVICE
float vertex_sample(const uint2 pixel, TPTContext& context, const uint32 i)
{
	return context.sequence.sample_2d(pixel.x, pixel.y, (context.in_bounce + 1) * 6 + i);
}

//------------------------------------------------------------------------------
/// processes a path vertex, performing the following three steps:
///\par
/// - sampling NEE
/// - accumulating emissive surface hits
/// - scattering
///
/// \tparam TPTContext				A path tracing context, which must adhere to the \ref TPTContext interface
/// \tparam TPTVertexProcessor		A vertex processor, which must adhere to the \ref TPTVertexProcessor interface
///
/// \param context					the path tracing context
/// \param vertex_processor			the vertex processor
/// \param renderer					the rendering context
/// \param pixel_info				the packed pixel info
/// \param pixel					the unpacked 2d pixel index
/// \param ray						the incoming ray
/// \param hit						the hit information
/// \param w						the current path weight
/// \param prev_nee_vertex_id			the NEE slot corresponding to the previous vertex
/// \param cone						the incoming ray cone
template <typename TPTContext, typename TPTVertexProcessor>
FERMAT_DEVICE
bool shade_vertex(
	TPTContext&				context,
	TPTVertexProcessor&		vertex_processor,
	RenderingContextView&	renderer,
	const uint32			bounce,
	const PixelInfo			pixel_info,
	const uint2				pixel,
	const MaskedRay&		ray,
	const Hit				hit,
	const cugar::Vector4f	w,
	const uint32			prev_vertex_info = uint32(-1),
	const uint32			prev_nee_vertex_id = uint32(-1),
	const cugar::Vector2f	cone = cugar::Vector2f(0))
{
	const float p_prev = w.w;

	const uint32 pixel_index = pixel_info.pixel;
		
	if (hit.t > 0.0f && hit.triId >= 0)
	{
		DEVICE_TIME( DeviceTimer timer );
		DEVICE_TIME( timer.start() );

		EyeVertex ev;
		ev.setup(ray, hit, w.xyz(), cugar::Vector4f(0.0f), bounce, renderer);

		DEVICE_TIME( per_warp_atomic_add( context.device_timers + SETUP_TIME, timer.take() ) );

		// write out gbuffer information
		if (bounce == 0)
		{
			renderer.fb.gbuffer.geo(pixel_index) = GBufferView::pack_geometry(ev.geom.position, ev.geom.normal_s);
			renderer.fb.gbuffer.uv(pixel_index)  = make_float4(hit.u, hit.v, ev.geom.texture_coords.x, ev.geom.texture_coords.y);
			renderer.fb.gbuffer.tri(pixel_index) = hit.triId;
			renderer.fb.gbuffer.depth(pixel_index) = hit.t;

			// write surface albedos
			renderer.fb(FBufferDesc::DIFFUSE_A,  pixel_index) += cugar::Vector4f(ev.material.diffuse)  * context.frame_weight;
			renderer.fb(FBufferDesc::SPECULAR_A, pixel_index) += (cugar::Vector4f(ev.material.specular) + cugar::Vector4f(1.0f))*0.5f * context.frame_weight;
		}

		DEVICE_TIME( per_warp_atomic_add( context.device_timers + FBUFFER_WRITES_TIME, timer.take() ) );

		// in order to select the footprint at the intersection, we use the formulation proposed by Bekaert:
		// R(x_k) = h/sqrtf(p(x_k|x_[k-1])) + R(x_[k-1])
		const float area_prob = cugar::rsqrtf(cone.y * ev.prev_G_prime);
		const float cone_radius = cone.x + area_prob;
				
		// lookup / insert an NEE RL entry
		uint32 nee_vertex_id = uint32(-1);
		if (context.do_nee)
		{
			bool is_secondary_diffuse = pixel_info.diffuse;

			nee_vertex_id = context.dl.preprocess_vertex(
				renderer,
				ev,
				pixel_info.pixel,
				context.in_bounce,
				is_secondary_diffuse,
				cone_radius,
				context.bbox );

			#if 0
			// debug visualization
			{
				cugar::Vector3f c;
				c.x = cugar::randfloat(0, nee_vertex_id);
				c.y = cugar::randfloat(1, nee_vertex_id);
				c.z = cugar::randfloat(2, nee_vertex_id);
				add_in<false>(renderer.fb(FBufferDesc::COMPOSITED_C), pixel_info.pixel, c, context.frame_weight);
				return;
			}
			#endif
		}

		DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_PREPROCESS_TIME, timer.take() ) );

		const uint32 vertex_info = vertex_processor.preprocess_vertex(
			context,
			renderer,
			pixel_info,
			ev,
			cone_radius,
			context.bbox,
			prev_vertex_info,
			w.xyz(),
			p_prev );

		DEVICE_TIME( per_warp_atomic_add( context.device_timers + PREPROCESS_VERTEX_TIME, timer.take() ) );

		// initialize our shifted sampling sequence
		float samples[6];
		for (uint32 i = 0; i < 6; ++i)
			samples[i] = vertex_sample(pixel, context, i);

		// directional-lighting
		if ((context.in_bounce + 2 <= context.options.max_path_length) &&
			(context.in_bounce > 0 || context.options.direct_lighting) &&
			renderer.dir_lights_count)
		{
			DEVICE_TIME( timer.restart() );

			// fetch the sampling dimensions
			const float z[3] = { samples[0], samples[1], samples[2] }; // use dimensions 0,1,2

			VertexGeometryId light_vertex;
			VertexGeometry   light_vertex_geom;
			float			 light_pdf;
			Edf				 light_edf;

			// use the third dimension to select a light source
			const uint32 light_idx = cugar::quantize( z[2], renderer.dir_lights_count );

			// sample the light source surface
			renderer.dir_lights[ light_idx ].sample(ev.geom.position, z, &light_vertex.prim_id, &light_vertex.uv, &light_vertex_geom, &light_pdf, &light_edf);

			// multiply by the light selection probability
			light_pdf /= renderer.dir_lights_count;

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + DIRLIGHT_SAMPLE_TIME, timer.take() ) );

			// join the light sample with the current vertex
			cugar::Vector3f out = (light_vertex_geom.position - ev.geom.position);
						
			const float d2 = fmaxf(1.0e-8f, cugar::square_length(out));

			// normalize the outgoing direction
			out *= rsqrtf(d2);

			cugar::Vector3f f_s_comp[Bsdf::kNumComponents];
			float			p_s_comp[Bsdf::kNumComponents];

			ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s_comp, p_s_comp, cugar::kProjectedSolidAngle);

			// check which paths are enabled
			const bool eval_diffuse = context.options.diffuse_scattering;
			const bool eval_glossy  = context.options.glossy_scattering; // TODO: handle the indirect_glossy toggle here

		#if 0
			cugar::Vector3f f_s(0.0f);
			float			p_s(0.0f);

			if (eval_diffuse)
			{
				f_s += f_s_comp[Bsdf::kDiffuseReflectionIndex] + f_s_comp[Bsdf::kDiffuseTransmissionIndex];
				p_s += p_s_comp[Bsdf::kDiffuseReflectionIndex] + p_s_comp[Bsdf::kDiffuseTransmissionIndex];
			}
			if (eval_glossy)
			{
				f_s += f_s_comp[Bsdf::kGlossyReflectionIndex] + f_s_comp[Bsdf::kGlossyTransmissionIndex];
				p_s += p_s_comp[Bsdf::kGlossyReflectionIndex] + p_s_comp[Bsdf::kGlossyTransmissionIndex];
			}
		#endif
			DEVICE_TIME( per_warp_atomic_add( context.device_timers + BRDF_EVAL_TIME, timer.take() ) );

			// evaluate the light's EDF and the surface BSDF
			const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + DIRLIGHT_EVAL_TIME, timer.take() ) );

			// evaluate the geometric term
			const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

			// TODO: perform MIS with the possibility of directly hitting the light source
			const float mis_w = 1.0f;

			// calculate the output weights
			cugar::Vector3f	out_w_d;
			cugar::Vector3f	out_w_g;
			uint32			out_vertex_info;

			vertex_processor.compute_nee_weights(
				context,
				renderer,
				pixel_info,
				prev_vertex_info,
				vertex_info,
				ev,
				eval_diffuse ? f_s_comp[Bsdf::kDiffuseReflectionIndex] + f_s_comp[Bsdf::kDiffuseTransmissionIndex] : cugar::Vector3f(0.0f),
				eval_glossy  ? f_s_comp[Bsdf::kGlossyReflectionIndex]  + f_s_comp[Bsdf::kGlossyTransmissionIndex]  : cugar::Vector3f(0.0f),
				w.xyz(),
				f_L * G * mis_w,
				out_w_d,
				out_w_g,
				out_vertex_info );

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + NEE_WEIGHTS_TIME, timer.take() ) );
		
		#if 0
			// calculate the cumulative sample weight, equal to f_L * f_s * G / p
			const cugar::Vector3f out_w	= w.xyz() * f_L * f_s * G * mis_w;
		#else
			// calculate the cumulative sample weight
			const cugar::Vector3f out_w = out_w_d + out_w_g;
		#endif

			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				DEVICE_TIME( timer.restart() );

				// find the right side of the normal
				const cugar::Vector3f N = dot(ev.geom.normal_s,ray.dir) > 0.0f ? -ev.geom.normal_s : ev.geom.normal_s;

				// enqueue the output ray
				MaskedRay out_ray;
				out_ray.origin	= ev.geom.position - ray.dir * 1.0e-3f; // shift back in space along the viewing direction
				out_ray.dir		= (light_vertex_geom.position - out_ray.origin); //out;
				out_ray.mask	= 0x1u; // shadow flag
				out_ray.tmax	= 0.9999f; //d * 0.9999f;

				context.trace_shadow_ray( vertex_processor, renderer, pixel_info, out_ray, out_w, out_w_d, out_w_g, vertex_info );

				DEVICE_TIME( per_warp_atomic_add( context.device_timers + TRACE_SHADOW_TIME, timer.take() ) );
			}
		}
				
		// perform next-event estimation to compute direct lighting
		if (context.do_nee)
		{
			DEVICE_TIME( timer.restart() );

			// fetch the sampling dimensions
			const float z[3] = { samples[0], samples[1], samples[2] }; // use dimensions 0,1,2
			//const float z[3] = {
			//	vertex_sample(pixel, bounce, 0u),
			//	vertex_sample(pixel, bounce, 1u),
			//	vertex_sample(pixel, bounce, 2u)
			//}; // use dimensions 0,1,2

			VertexGeometryId light_vertex;
			VertexGeometry   light_vertex_geom;
			float			 light_pdf;
			Edf				 light_edf;

			// sample the light source surface
			const uint32 nee_sample_id = context.dl.sample( nee_vertex_id, z, &light_vertex, &light_vertex_geom, &light_pdf, &light_edf );

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_SAMPLE_TIME, timer.take() ) );

			// join the light sample with the current vertex
			cugar::Vector3f out = (light_vertex_geom.position - ev.geom.position);
						
			const float d2 = fmaxf(1.0e-8f, cugar::square_length(out));

			// normalize the outgoing direction
			out *= rsqrtf(d2);

			cugar::Vector3f f_s_comp[Bsdf::kNumComponents];
			float			p_s_comp[Bsdf::kNumComponents];

			ev.bsdf.f_and_p(ev.geom, ev.in, out, f_s_comp, p_s_comp, cugar::kProjectedSolidAngle);

			// check which paths are enabled
			const bool eval_diffuse = context.options.diffuse_scattering;
			const bool eval_glossy  = context.options.glossy_scattering; // TODO: handle the indirect_glossy toggle here

			cugar::Vector3f f_s(0.0f);
			float			p_s(0.0f);

			if (eval_diffuse)
			{
				f_s += f_s_comp[Bsdf::kDiffuseReflectionIndex] + f_s_comp[Bsdf::kDiffuseTransmissionIndex];
				p_s += p_s_comp[Bsdf::kDiffuseReflectionIndex] + p_s_comp[Bsdf::kDiffuseTransmissionIndex];
			}
			if (eval_glossy)
			{
				f_s += f_s_comp[Bsdf::kGlossyReflectionIndex] + f_s_comp[Bsdf::kGlossyTransmissionIndex];
				p_s += p_s_comp[Bsdf::kGlossyReflectionIndex] + p_s_comp[Bsdf::kGlossyTransmissionIndex];
			}

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + BRDF_EVAL_TIME, timer.take() ) );

			// evaluate the light's EDF and the surface BSDF
			const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, -out) / light_pdf;

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_EVAL_TIME, timer.take() ) );

			// evaluate the geometric term
			const float G = fabsf(cugar::dot(out, ev.geom.normal_s) * cugar::dot(out, light_vertex_geom.normal_s)) / d2;

			// TODO: perform MIS with the possibility of directly hitting the light source
			const float p1 = light_pdf;
			const float p2 = p_s * G;
			const float mis_w =
				(bounce == 0 && context.options.direct_lighting_bsdf) ||
				(bounce >  0 && context.options.indirect_lighting_bsdf) ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

			// calculate the output weights
			cugar::Vector3f	out_w_d;
			cugar::Vector3f	out_w_g;
			uint32			out_vertex_info;

			vertex_processor.compute_nee_weights(
				context,
				renderer,
				pixel_info,
				prev_vertex_info,
				vertex_info,
				ev,
				eval_diffuse ? f_s_comp[Bsdf::kDiffuseReflectionIndex] + f_s_comp[Bsdf::kDiffuseTransmissionIndex] : cugar::Vector3f(0.0f),
				eval_glossy  ? f_s_comp[Bsdf::kGlossyReflectionIndex]  + f_s_comp[Bsdf::kGlossyTransmissionIndex]  : cugar::Vector3f(0.0f),
				w.xyz(),
				f_L * G * mis_w,
				out_w_d,
				out_w_g,
				out_vertex_info );

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + NEE_WEIGHTS_TIME, timer.take() ) );

		#if 0
			// calculate the cumulative sample weight, equal to f_L * f_s * G / p
			const cugar::Vector3f out_w = w.xyz() * f_L * f_s * G * mis_w;
		#else
			// calculate the cumulative sample weight
			const cugar::Vector3f out_w = out_w_d + out_w_g;
		#endif

			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				DEVICE_TIME( timer.restart() );

				// enqueue the output ray
				MaskedRay out_ray;
				out_ray.origin	= ev.geom.position - ray.dir * 1.0e-4f; // shift back in space along the viewing direction
				out_ray.dir		= (light_vertex_geom.position - out_ray.origin); //out;
				out_ray.mask    = 0x2u;
				out_ray.tmax	= 0.9999f; //d * 0.9999f;

				context.trace_shadow_ray( vertex_processor, renderer, pixel_info, out_ray, out_w, out_w_d, out_w_g, vertex_info, nee_vertex_id, nee_sample_id );

				DEVICE_TIME( per_warp_atomic_add( context.device_timers + TRACE_SHADOW_TIME, timer.take() ) );
			}
		}

		// accumulate the emissive component along the incoming direction
		if (context.do_accumulate_emissive)
		{
			DEVICE_TIME( timer.restart() );

			VertexGeometry	light_vertex_geom = ev.geom;
			float			light_pdf;
			Edf				light_edf;

			context.dl.map( prev_nee_vertex_id, hit.triId, cugar::Vector2f(hit.u, hit.v), light_vertex_geom, &light_pdf, &light_edf );

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_MAPPING_TIME, timer.take() ) );

			// evaluate the edf's output along the incoming direction
			const cugar::Vector3f f_L = light_edf.f(light_vertex_geom, light_vertex_geom.position, ev.in);

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + LIGHTS_EVAL_TIME, timer.take() ) );

			const float d2 = fmaxf(1.0e-10f, hit.t * hit.t);

			// compute the MIS weight with next event estimation at the previous vertex
			const float G_partial = fabsf(cugar::dot(ev.in, light_vertex_geom.normal_s)) / d2; // NOTE: G_partial doesn't include the dot product between 'in and the normal at the previous vertex

			const float p1 = G_partial * p_prev;											// NOTE: p_prev is the solid angle probability of sampling the BSDF at the previous vertex, i.e. p_proj * dot(in,normal)
			const float p2 = light_pdf;
			const float mis_w =
				(bounce == 1 && context.options.direct_lighting_nee) ||
				(bounce >  1 && context.options.indirect_lighting_nee) ? mis_heuristic<MIS_HEURISTIC>(p1, p2) : 1.0f;

			// and accumulate the weighted contribution
			const cugar::Vector3f out_w	= w.xyz() * f_L * mis_w;

			// and accumulate the weighted contribution
			if (cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				vertex_processor.accumulate_emissive(
					context,
					renderer,
					pixel_info,
					prev_vertex_info,
					vertex_info,
					ev,
					out_w );
			}

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + FBUFFER_WRITES_TIME, timer.take() ) );
		}

		// trace a bounce ray
		if (context.do_scatter)
		{
			DEVICE_TIME( timer.restart() );

			// fetch the sampling dimensions
			const float z[3] = { samples[3], samples[4], samples[5] }; // use dimensions 3,4,5
			//const float z[3] = {
			//	vertex_sample(pixel, bounce, 3u),
			//	vertex_sample(pixel, bounce, 4u),
			//	vertex_sample(pixel, bounce, 5u)
			//}; // use dimensions 3,4,5

			// sample a scattering event
			cugar::Vector3f		out(0.0f);
			cugar::Vector3f		g(0.0f);
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
				// 1. indirect glossy scattering is disabled, OR
				// 2. we have sampled a diffuse reflection and indirect_glossy == false (TODO)
				if (context.options.glossy_scattering == false)
					component_mask &= ~uint32(Bsdf::kGlossyMask);
			}

			scatter(ev, z, out_comp, out, p, p_proj, g, true, false, false, Bsdf::ComponentType(component_mask));

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + BRDF_SAMPLE_TIME, timer.take() ) );

			// compute the output weight
			cugar::Vector3f	out_w;
			uint32			out_vertex_info;

			vertex_processor.compute_scattering_weights(
				context,
				renderer,
				pixel_info,
				prev_vertex_info,
				vertex_info,
				ev,
				out_comp,
				g,
				w.xyz(),
				out_w,
				out_vertex_info );

			DEVICE_TIME( per_warp_atomic_add( context.device_timers + SCATTERING_WEIGHTS_TIME, timer.take() ) );

			if (p != 0.0f && cugar::max_comp(out_w) > 0.0f && cugar::is_finite(out_w))
			{
				// enqueue the output ray
				MaskedRay out_ray;
				out_ray.origin	= ev.geom.position;
				out_ray.dir		= out;
				out_ray.mask	= __float_as_uint(1.0e-3f);
				out_ray.tmax	= 1.0e8f;

				const float out_p = p;

				// in order to select the footprint, we use the formulation proposed by Bekaert:
				// R(x_k) = h/sqrtf(p(x_k|x_[k-1])) + R(x_[k-1])
				const float min_p = 32.0f; // 32 corresponds to a maximum angle of ~10 degrees
				const cugar::Vector2f out_cone(cone_radius, cugar::max(out_p, min_p));
					// tan(alpha) = 1/sqrt(out_p) => out_p = 1/tan(alpha)^2
					// out_p > min_p => 1/tan(alpha)^2 > min_p => tan(alpha)^2 < 1/min_p => tan(alpha) < 1/min_p^2

				// mark if the path ever went through a diffuse bounce
				bool is_secondary_diffuse = pixel_info.diffuse || (out_comp & Bsdf::kDiffuseMask);

				context.trace_ray(
					vertex_processor, 
					renderer,
					PixelInfo(pixel_index, out_comp, is_secondary_diffuse),
					out_ray,
					cugar::Vector4f(out_w, out_p),
					out_cone,
					out_vertex_info,
					nee_vertex_id );

				DEVICE_TIME( per_warp_atomic_add( context.device_timers + TRACE_SHADED_TIME, timer.take() ) );
				return true;	// continue the path
			}
		}
	}
	else
	{
		// hit the environment - perform sky lighting
	}
	return false;	// stop the path
}
//------------------------------------------------------------------------------

///@} PTLibCore
///@} PTLib
///@} Fermat
