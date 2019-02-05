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

#include <bpt_context.h>
#include <bpt_utils.h>
#include <bpt_options.h>
#include <bpt_kernels.h>
#include <rt.h>

/// \page BPTLibPage BPTLib
/// Top: \ref OvertureContentsPage
/// 
/// \ref BPTLib is a flexible bidirectional path tracing library, thought to be as performant as possible and yet vastly configurable at compile-time.
/// The module is organized into a host library of parallel kernels, \ref BPTLib, and a core module of device-side functions, \ref BPTLibCore.
/// The underlying idea is that all the bidirectional sampling functions and kernels are designed to use a wavefront scheduling approach,
/// in which ray tracing queries are queued from shading kernels, and get processed in separate waves.
///
///\section BPTLibCoreSection BPTLibCore Description
///
///\par
/// \ref BPTLibCore provides functions to:
/// - generate primary light vertices, i.e. vertices on the light source surfaces, each accompanied by a sampled outgoing direction
/// - process secondary light vertices, starting from ray hits corresponding to the sampled outgoing direction at the previous vertex
/// - generate primary eye vertices, i.e. vertices on the camera, each accompanied by a sampled outgoing direction
/// - process secondary eye vertices, starting from ray hits corresponding to the sampled outgoing direction at the previous vertex
///\par
/// In order to make the whole process configurable, all the functions accept the following template interfaces:
///\par
/// \anchor TBPTContext
/// 1. a context interface, holding members describing the current path tracer state, 
///    including all the necessary queues, a set of options, and the storage for recording all generated light vertices;
///    this class needs to inherit from \ref BPTContextBase :
///\n
///\snippet bpt_context.h BPTContextBaseBlock
///\n
/// \anchor TBPTConfig
/// 2. a user defined "policy" class, configuring the path sampling process;
///    this class is responsible for deciding what exactly to do at and with each eye and light subpath vertex,
///    and needs to provide the following interface:
///\n
///\code
/// struct TBPTConfig
/// {
/// 	uint32  max_path_length			: 10;
/// 	uint32	light_sampling			: 1;
/// 	uint32  light_ordering			: 1;
/// 	uint32  eye_sampling			: 1;
/// 	uint32	use_vpls				: 1;
/// 	uint32	use_rr					: 1;
/// 	uint32	direct_lighting_nee		: 1;
/// 	uint32	direct_lighting_bsdf	: 1;
/// 	uint32  indirect_lighting_nee	: 1;
/// 	uint32  indirect_lighting_bsdf  : 1;
/// 	uint32	visible_lights			: 1;
/// 	float   light_tracing;
/// 
/// 	// decide whether to terminate a given light subpath
/// 	//
/// 	// \param path_id			index of the light subpath
/// 	// \param s				vertex number along the light subpath
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	bool terminate_light_subpath(const uint32 path_id, const uint32 s) const;
/// 
/// 	// decide whether to terminate a given eye subpath
/// 	//
/// 	// \param path_id			index of the eye subpath
/// 	// \param s				vertex number along the eye subpath
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	bool terminate_eye_subpath(const uint32 path_id, const uint32 t) const;
/// 
/// 	// decide whether to store a given light vertex
/// 	//
/// 	// \param path_id			index of the light subpath
/// 	// \param s				vertex number along the light subpath
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	bool store_light_vertex(const uint32 path_id, const uint32 s, const bool absorbed) const;
/// 
/// 	// decide whether to perform a bidirectional connection
/// 	//
/// 	// \param eye_path_id		index of the eye subpath
/// 	// \param t				vertex number along the eye subpath
/// 	// \param absorbed			true if the eye subpath has been absorbed/terminated here
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	bool perform_connection(const uint32 eye_path_id, const uint32 t, const bool absorbed) const;
/// 
/// 	// decide whether to accumulate an emissive sample from a pure (eye) path tracing estimator
/// 	//
/// 	// \param eye_path_id		index of the eye subpath
/// 	// \param t				vertex number along the eye subpath
/// 	// \param absorbed			true if the eye subpath has been absorbed/terminated here
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	bool accumulate_emissive(const uint32 eye_path_id, const uint32 t, const bool absorbed) const;
/// 
/// 	// process/store the given light vertex
/// 	//
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	void visit_light_vertex(
/// 		const uint32			light_path_id,
/// 		const uint32			depth,
/// 		const VertexGeometryId	v_id,
/// 		TBPTContext&			context,
/// 		RenderingContextView&	renderer) const;
/// 
/// 	// process/store the given eye vertex
/// 	//
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	void visit_eye_vertex(
/// 		const uint32			eye_path_id,
/// 		const uint32			depth,
/// 		const VertexGeometryId	v_id,
/// 		const EyeVertex&		v,
/// 		TBPTContext&			context,
/// 		RenderingContextView&	renderer) const;
/// };
///\endcode
///\n
///In practice, an implementation can inherit from the pre-packaged \ref BPTConfigBase class and override any of its methods.
///\n
///\n
/// \anchor TSampleSink
/// 3. a user defined sample "sink" class, specifying what to do with all the generated bidirectional path samples (i.e. full paths);
///    this class needs to expose the same interface as \ref SampleSinkBase :
///\n
///\snippet bpt_kernels.h SampleSinkBaseBlock
///\n
/// \anchor TPrimaryCoordinates
/// 4. a user defined class specifying the primary sample space coordinates of the generated subpaths;
///    this class needs to expose the following itnerface:
///\n
///\code
/// struct TPrimaryCoords
/// {
/// 	// return the primary sample space coordinate of the d-th component of the j-th vertex
/// 	// of the i-th subpath
/// 	//
/// 	// \param idx		the subpath index 'i'
/// 	// \param vertex	the vertex index 'j' in the given subpath
/// 	// \param dim		the index of the dimension 'd' of the given subpath vertex
/// 	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
/// 	float sample(const uint32 idx, const uint32 vertex, const uint32 dim) const;
/// };
///\endcode
///\par
/// The complete list of functions can be found in the \ref BPTLibCore module documentation.
///
///\section BPTLibSection BPTLib Description
///
///\par
/// \ref BPTLib contains the definition of the full bidirectional path tracing pipeline;
/// as for the lower level \ref BPTLibCore functions, the pipeline is customizable through a \ref TBPTConfig policy
/// class, a \ref TSampleSink, and a set of \ref TPrimaryCoordinates.
///\par
/// While the module itself defines all separate stages of the pipeline, the entire pipeline can be instanced
/// with a single host function call to:
///
///\anchor sample_paths_anchor
///\code
/// // A host function dispatching a series of kernels to sample a given number of full paths.
/// // The generated paths are controlled by two user-defined sets of primary space coordinates, one
/// // for eye and light subpaths sampling.
/// // Specifically, this function executes the following two functions:
/// //
/// // - \ref sample_light_subpaths()
/// // - \ref sample_eye_subpaths()
/// //
/// // \tparam TEyePrimaryCoordinates       a set of primary space coordinates, see TPrimaryCoordinates
/// // \tparam TLightPrimaryCoordinates     a set of primary space coordinates, see TPrimaryCoordinates
/// // \tparam TSampleSink                  a sample sink, specifying what to do with each generated path sample
/// // \tparam TBPTContext                  a bidirectional path tracing context clas
/// // \tparam TBPTConfig                   a policy class controlling the behaviour of the path sampling process
/// //
/// // \param n_eye_paths               the number of eye subpaths to sample
/// // \param n_light_paths             the number of light subpaths to sample
/// // \param eye_primary_coords        the set of primary sample space coordinates used to generate eye subpaths
/// // \param light_primary_coords      the set of primary sample space coordinates used to generate light subpaths
/// // \param sample_sink               the sample sink
/// // \param context                   the bidirectional path tracing context
/// // \param config                    the config policy
/// // \param renderer                  the host-side rendering context
/// // \param renderer_view             a view of the rendering context
/// // \param lazy_shadows              a flag indicating whether to resolve shadows lazily, after generating
/// //                                  all light and eye vertices, or right away as each wave of new vertices is processed
/// template <
///   typename TEyePrimaryCoordinates,
///   typename TLightPrimaryCoordinates,
///   typename TSampleSink,
///   typename TBPTContext,
///   typename TBPTConfig>
/// void sample_paths(
/// 	const uint32				n_eye_paths,
/// 	const uint32				n_light_paths,
/// 	TEyePrimaryCoordinates		eye_primary_coords,
/// 	TLightPrimaryCoordinates	light_primary_coords,
/// 	TSampleSink					sample_sink,
/// 	TBPTContext&				context,
/// 	const TBPTConfig&			config,
/// 	RenderingContext&			renderer,
/// 	RenderingContextView&		renderer_view,
/// 	const bool					lazy_shadows = false)
///\endcode
///
///\par
/// \ref sample_paths() generates bidirectional paths with at least two eye vertices, i.e. t=2 in Veach's terminology.
/// A separate function allows to process paths with t=1, connecting directly to a vertex on the lens:
///
///\anchor light_tracing_anchor
///\code
/// // A host function dispatching a series of kernels to process pure light tracing paths.
/// // Specifically, this function executes the following two functions:
/// //
/// // - \ref light_tracing()
/// // - \ref solve_shadows()
/// //
/// // This function needs to be called <i>after</i> a previous call to \ref generate_light_subpaths(), as it assumes
/// // a set of light subpaths have already been sampled and it is possible to connect them to the camera.
/// //
/// // \tparam TSampleSink					a sample sink, specifying what to do with each generated path sample, see \ref SampleSinkAnchor
/// // \tparam TBPTContext					a bidirectional path tracing context class, see \ref BPTContextBase
/// // \tparam TBPTConfig					a policy class controlling the behaviour of the path sampling process, see \ref BPTConfigBase
/// //
/// template <
///   typename TSampleSink,
///   typename TBPTContext,
///   typename TBPTConfig>
/// void light_tracing(
/// 	const uint32			n_light_paths,
/// 	TSampleSink				sample_sink,
/// 	TBPTContext&			context,
/// 	const TBPTConfig&		config,
/// 	RenderingContext&		renderer,
/// 	RenderingContextView&	renderer_view)
///\endcode
///
///\section BPTExampleSection An Example
///
///\par
/// At this point, it might be useful to take a look at the implementation of the \ref BPT renderer to see how this is used.
/// We'll start from the implementation of the render method:
///
///\snippet bpt_impl.h BPT::render
///
///\par
/// Besides some boilerplate, this function instantiates a context, a config, some light and eye primary sample coordinate 
/// generators (\ref TiledLightSubpathPrimaryCoords and \ref PerPixelEyeSubpathPrimaryCoords), and executes the \ref sample_paths()
/// and \ref light_tracing() functions above.
/// What is interesting now is taking a look at the definition of the sample sink class:
///
///\snippet bpt_impl.h BPTConnectionsSink
///
///\par
/// As you may notice, this implementation is simply taking each sample, and accumulating its contribution to the corresponding
/// pixel in the target framebuffer.
/// Here, we are using the fact that the eye path index corresponds exactly to the pixel index, a consequence of using
/// the \ref PerPixelEyeSubpathPrimaryCoords class.
///
/// <img src="water_caustic.jpg" style="position:relative; bottom:-10px; border:0px; width:698px;"/>
///


namespace bpt {

///@addtogroup Fermat
///@{

///@defgroup BPTLib
/// This module provides a flexible bidirectional path tracing library, thought to be as performant as possible and yet vastly configurable at compile-time.
/// For more information, see the \ref BPTLibPage page.
///@{

///\par
/// A host function dispatching a series of kernels to sample a given number of light subpaths.
/// The generated subpaths are controlled by a user-defined set of primary space coordinates.
/// Specifically, this function executes a pipeline comprised of the following kernels:
///\n
/// - \ref generate_primary_light_vertices()
/// - \ref RTContext::trace()
/// - \ref process_secondary_light_vertices()
///\par
/// A templated policy config class specifies what to do with the generated light vertices.
///
/// \tparam TPrimaryCoordinates		a set of primary space coordinates, \ref TPrimaryCoordinatesAnchor
/// \tparam TBPTContext				a bidirectional path tracing context class, see \ref BPTContextBase
/// \tparam TBPTConfig				a policy class controlling the behaviour of the path sampling process, see \ref BPTConfigBase
///
template <typename TPrimaryCoordinates, typename TBPTContext, typename TBPTConfig>
void sample_light_subpaths(
	const uint32			n_light_paths,
	TPrimaryCoordinates		primary_coords,
	TBPTContext&			context,
	const TBPTConfig&		config,
	RenderingContext&		renderer,
	RenderingContextView&	renderer_view)
{
	// reset the output queue size
	cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

	// reset the output vertex counter
	cudaMemset(context.light_vertices.vertex_counter, 0x00, sizeof(uint32));

	// generate primary light vertices from the mesh light samples, including the sampling of a direction
	bpt::generate_primary_light_vertices(n_light_paths, primary_coords, context, renderer_view, config);

	// swap the input and output queues
	std::swap(context.in_queue, context.scatter_queue);

	// for each bounce: trace rays, process hits (store light_vertices, generate sampling directions)
	for (context.in_bounce = 0;
		context.in_bounce < context.options.max_path_length - 1;
		context.in_bounce++)
	{
		uint32 in_queue_size;

		// read out the number of output rays produced
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		//fprintf(stderr, "  bounce %u:\n    rays: %u\n", bounce, in_queue_size);

		// reset the output queue counters
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// trace the rays generated at the previous bounce
		//
		//fprintf(stderr, "    trace\n");
		{
			renderer.get_rt_context()->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
		}

		// process the light_vertex hits at this bounce
		//
		bpt::process_secondary_light_vertices(in_queue_size, n_light_paths, primary_coords, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("process light vertices"));

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);
	}

	//uint32 vertex_counts[1024];
	//cudaMemcpy(&vertex_counts[0], context.light_vertices.vertex_counts, sizeof(uint32) * context.options.max_path_length, cudaMemcpyDeviceToHost);
	//for (uint32 s = 0; s < context.options.max_path_length; ++s)
	//	fprintf(stderr, "  light vertices[%u] : %u (%u))\n", s, vertex_counts[s] - (s ? vertex_counts[s-1] : 0), vertex_counts[s]);
}

///\par
/// A host function dispatching a series of kernels to resolve shadows / occlusion of the queued bidirectional connection samples
/// generated by the \ref sample_eye_subpaths function.
///
/// \tparam TSampleSink				a sample sink, specifying what to do with each generated path sample, see \ref SampleSinkAnchor
/// \tparam TBPTContext				a bidirectional path tracing context class, see \ref BPTContextBase
///
template <typename TSampleSink, typename TBPTContext>
void solve_shadows(TSampleSink sample_sink, TBPTContext& context, RenderingContext& renderer, RenderingContextView& renderer_view)
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
		//fprintf(stderr, "      solve occlusion\n");

		bpt::solve_occlusions(shadow_queue_size, sample_sink, context, renderer_view);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("solve occlusion"));

		// reset the shadow queue counter
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));
	}
}

///\par
/// A host function dispatching a series of kernels to sample a given number of eye subpaths.
/// The generated subpaths are controlled by a user-defined set of primary space coordinates.
/// Specifically, this function executes a pipeline comprised of the following kernels:
///\n
/// - \ref generate_primary_eye_vertices()
/// - \ref RTContext::trace()
/// - \ref process_secondary_eye_vertices()
/// - \ref solve_shadows()
///\par
/// This function needs to be called <i>after</i> a previous call to \ref generate_light_subpaths(), as it assumes
/// a set of light subpaths have already been sampled and it is possible to sample and perform connections between the pre-existing light vertices and
/// each new eye vertex.
/// A templated sample sink class specifies what to do with the full light path samples generated in this phase (i.e. the paths
/// hitting an emissive surface), while the generated "connection" samples are stored in a queue controlled by the templated policy config class.
/// These connections are processed by the \ref solve_shadows() function.
///
/// \tparam TPrimaryCoordinates		a set of primary space coordinates, \ref TPrimaryCoordinatesAnchor
/// \tparam TSampleSink				a sample sink, specifying what to do with each generated path sample, see \ref SampleSinkAnchor
/// \tparam TBPTContext				a bidirectional path tracing context class, see \ref BPTContextBase
/// \tparam TBPTConfig				a policy class controlling the behaviour of the path sampling process, see \ref BPTConfigBase
///
template <typename TPrimaryCoordinates, typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void sample_eye_subpaths(
	const uint32			n_eye_paths,
	const uint32			n_light_paths,
	TPrimaryCoordinates		primary_coords,
	TSampleSink				sample_sink,
	TBPTContext&			context,
	const TBPTConfig&		config,
	RenderingContext&		renderer,
	RenderingContextView&	renderer_view,
	const bool				lazy_shadows = false)
{
	// reset the output queue counters
	cudaMemset(context.shadow_queue.size,  0x00, sizeof(uint32));
	cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));

	//fprintf(stderr, "    trace primary rays : started\n");

	// generate the primary rays
	bpt::generate_primary_eye_vertices(n_eye_paths, n_light_paths, primary_coords, context, renderer_view, config);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("generate primary rays"));

	//fprintf(stderr, "    trace primary rays: done\n");

	for (context.in_bounce = 0;
		 context.in_bounce < context.options.max_path_length;
		 context.in_bounce++)
	{
		uint32 in_queue_size;

		// read out the number of output rays produced by the previous pass
		cudaMemcpy(&in_queue_size, context.in_queue.size, sizeof(uint32), cudaMemcpyDeviceToHost);
		CUDA_CHECK(cugar::cuda::check_error("memcpy"));

		//fprintf(stderr, "    input queue size: %u\n", in_queue_size);

		// check whether there's still any work left
		if (in_queue_size == 0)
			break;

		//fprintf(stderr, "  bounce %u:\n    rays: %u\n", bounce, in_queue_size);

		// reset the output queue counters
		cudaMemset(context.scatter_queue.size, 0x00, sizeof(uint32));
		CUDA_CHECK(cugar::cuda::check_error("memset"));

		// trace the rays generated at the previous bounce
		//
		//fprintf(stderr, "    trace\n");
		{
			renderer.get_rt_context()->trace(in_queue_size, (Ray*)context.in_queue.rays, context.in_queue.hits);
			CUDA_CHECK(cugar::cuda::sync_and_check_error("trace"));
		}

		//fprintf(stderr, "    shade hits(%u)\n", context.in_bounce);

		// perform lighting at this bounce
		//
		bpt::process_secondary_eye_vertices(in_queue_size, n_eye_paths, n_light_paths, sample_sink, primary_coords, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("process eye vertices"));

		if (lazy_shadows == false)
		{
			// trace & accumulate occlusion queries
			solve_shadows(sample_sink, context, renderer, renderer_view);
		}

		//fprintf(stderr, "    finish pass\n");

		// swap the input and output queues
		std::swap(context.in_queue, context.scatter_queue);
	}

	if (lazy_shadows)
	{
		// trace & accumulate occlusion queries
		solve_shadows(sample_sink, context, renderer, renderer_view);
	}
}

///\par
/// A host function dispatching a series of kernels to sample a given number of full paths.
/// The generated paths are controlled by two user-defined sets of primary space coordinates, one
/// for eye and light subpaths sampling.
/// Specifically, this function executes the following two functions:
///\n
/// - \ref sample_light_subpaths()
/// - \ref sample_eye_subpaths()
///
/// \tparam TEyePrimaryCoordinates		a set of primary space coordinates, \ref TPrimaryCoordinatesAnchor
/// \tparam TLightPrimaryCoordinates	a set of primary space coordinates, \ref TPrimaryCoordinatesAnchor
/// \tparam TSampleSink					a sample sink, specifying what to do with each generated path sample, see \ref SampleSinkAnchor
/// \tparam TBPTContext					a bidirectional path tracing context class, see \ref BPTContextBase
/// \tparam TBPTConfig					a policy class controlling the behaviour of the path sampling process, see \ref BPTConfigBase
///
template <typename TEyePrimaryCoordinates, typename TLightPrimaryCoordinates, typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void sample_paths(
	const uint32				n_eye_paths,
	const uint32				n_light_paths,
	TEyePrimaryCoordinates		eye_primary_coords,
	TLightPrimaryCoordinates	light_primary_coords,
	TSampleSink					sample_sink,
	TBPTContext&				context,
	const TBPTConfig&			config,
	RenderingContext&			renderer,
	RenderingContextView&		renderer_view,
	const bool					lazy_shadows = false)
{
	sample_light_subpaths(
		n_light_paths,
		light_primary_coords,
		context,
		config,
		renderer,
		renderer_view);

	sample_eye_subpaths(
		n_eye_paths,
		n_light_paths,
		eye_primary_coords,
		sample_sink,
		context,
		config,
		renderer,
		renderer_view,
		lazy_shadows);
}

///\par
/// A host function dispatching a series of kernels to process pure light tracing paths.
/// Specifically, this function executes the following two functions:
///\n
/// - \ref light_tracing()
/// - \ref solve_shadows()
///\par
/// This function needs to be called <i>after</i> a previous call to \ref generate_light_subpaths(), as it assumes
/// a set of light subpaths have already been sampled and it is possible to connect them to the camera.
///
/// \tparam TSampleSink					a sample sink, specifying what to do with each generated path sample, see \ref SampleSinkAnchor
/// \tparam TBPTContext					a bidirectional path tracing context class, see \ref BPTContextBase
/// \tparam TBPTConfig					a policy class controlling the behaviour of the path sampling process, see \ref BPTConfigBase
///
template <typename TSampleSink, typename TBPTContext, typename TBPTConfig>
void light_tracing(
	const uint32			n_light_paths,
	TSampleSink				sample_sink,
	TBPTContext&			context,
	const TBPTConfig&		config,
	RenderingContext&		renderer,
	RenderingContextView&	renderer_view)
{
	// solve pure light tracing occlusions
	if (context.options.light_tracing)
	{
		//fprintf(stderr, "  light tracing : started\n");

		// reset the output queue counters
		cudaMemset(context.shadow_queue.size, 0x00, sizeof(uint32));

		bpt::light_tracing(n_light_paths, context, renderer_view, config);
		CUDA_CHECK(cugar::cuda::sync_and_check_error("light tracing"));

		bpt::solve_shadows(sample_sink, context, renderer, renderer_view);
		//fprintf(stderr, "  light tracing : done\n");
	}
}

///@} BPTLib
///@} Fermat

} // namespace bpt