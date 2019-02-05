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
#include <types.h>
#include <buffers.h>
#include <ray.h>
#include <tiled_sequence.h>
#include <cugar/sampling/lfsr.h>
#include <cugar/linalg/bbox.h>
#include <renderer_interface.h>
#include <hashmap.h>

struct RenderingContext;
struct MeshVTLStorage;
struct ClusteredRLStorage;
struct AdaptiveClusteredRLStorage;

/// \page PSFPTPageCode psfpt_impl.h
///
/// \include psfpt_impl.h

/// \page PSFPTPage The Path-Space Filtering Path Tracer
/// Top: \ref OvertureContentsPage
///
///\par
/// In order to highlight the flexibility of our \ref PTLib library, we will now describe an
/// implementation of the <a href="https://dl.acm.org/citation.cfm?id=3214806">Fast path space filtering by jittered spatial hashing</a>
/// algorithm by Binder et al. built on top of this framework.
///\par
/// The main idea behind this algorithm is that rather than utilizing the sampled paths directly and splatting them to their single originating pixel
/// only, they are cut somewhere in the middle (the original paper was doing this at the first bounce, we extend this to an arbitrary vertex), and the
/// contributions at the specified vertex are merged and averaged (or "filtered") into a discrete spatial hash.
/// This way, the filtered value averaged into a single cell will be virtually "splat" to all paths incident to that cell, and consequently to all the
/// corresponding pixels (dramatically decreasing variance at the expense of some bias, or error).
/// The whole process is illustrated for two paths in the following figure:
/// <img src="psfpt-scheme.jpg" style="position:relative; bottom:-10px; border:0px; width:500px;"/>
///\par
/// Now, taking a look at Fermat's implementation in \ref PSFPTPageCode, you'll notice that the overall structure is pretty similar to that of our \ref HelloRendererPage
/// "Hello World" prototype path-tracer, and even more to that of the \ref PTPage "PTLib-based Path Tracer".
/// Skipping some details, you'll notice there is again a context class definition:
///\n
///\code
/// template <typename TDirectLightingSampler>
/// struct PSFPTContext : PTContextBase<PSFPTOptions>, PTContextQueues
/// {
/// 	PSFRefQueue	ref_queue;		// a queue of PSF references
/// 
/// 	HashMap		psf_hashmap;	// the PSF hashmap
/// 	float4*		psf_values;		// the PSF values
/// 
/// 	TDirectLightingSampler dl;	// the direct-lighting sampler
/// };
///\endcode
///
///\par
/// The main news here is the fact we inherited it from \ref PTContextBase and \ref PTContextQueues, and added a few fields:
///\n
/// - an additional "references queue"
/// - a hashmap and the corresponding values
/// - and a \ref TDirectLightingSampler "direct lighting sampler".
///\par
/// The additional queue is needed to store references to the hashmap cells. These references represent forward paths sampled from the camera that land on some cell,
/// with their corresponding path weight (i.e. the throughput the path was carrying until it hit the cell, properly divided by its sampling pdf).
/// We need to keep these references around because we are going to employ a two pass algorithm: a first one in which full paths are sampled and cut at the specified vertex,
/// inserting their outgoing radiance into the corresponding hash cell, and a final one in which all the cell references created in the first pass are looked up and splat 
/// on screen.
/// This two-stage separation is needed to make sure that <i>all</i> samples are filtered together <i>before</i> we actually splat them.
///
///\par
/// The beginning of the PSFPT::render() method should also look fairly familiar:
///
///\snippet psfpt_impl.h PSFPT::render-1
///
///\par
/// In fact, the only news here should be the very last two lines:
///
///\snippet psfpt_impl.h PSFPT::instantiate_vertex_processor
///
/// i.e. the instantiation of a custom \ref TPTVertexProcessor - in this case the \ref PSFPTVertexProcessor.
/// After that, the body of the render method is almost trivial:
///
///\snippet psfpt_impl.h PSFPT::render-2
///
///\par
/// All path tracing and kernel dispatch complexity has been absorbed into \ref PTLibPage!
/// Particularly, it has been absorbed in the \ref path_trace_loop_anchor "path_trace_loop()" method.
///
///\par
/// All of it, except for some crucial details.
/// The details specified by our \ref PSFPTVertexProcessor policy class, the class saying what exactly needs to be done with the path vertices
/// generated by \ref PTLibPage itself.
///
///\par
/// The first method this class is responsible for implementing is the preprocess_vertex() method, which is called each time a new path
/// vertex is created, before anything is actually done at that very vertex.
/// Here, we use this to perform our jittered spatial hashing of the vertex position and normal coordinates, and retrieve the corresponding hash cell:
///
///\code
/// // preprocess a vertex and return some packed vertex info
/// //
/// template <typename TPTContext>
/// FERMAT_DEVICE
/// uint32 preprocess_vertex(
/// 		  TPTContext&			context,			// the current context
/// 	const RenderingContextView&	renderer,			// the current renderer
/// 	const PixelInfo				pixel_info,			// packed pixel info
/// 	const EyeVertex&			ev,					// the local vertex
/// 	const float					cone_radius,		// the current cone radius
/// 	const cugar::Bbox3f			scene_bbox,			// the scene bounding box
/// 	const uint32				prev_vertex_info,	// the packed vertex info at the previous vertex
/// 	const cugar::Vector3f		w,					// the current path weight
/// 	const float					p_prev)				// the scattering solid angle probability at the previous vertex
/// {
/// 	// access the vertex info we returned at the previous vertex along this path (sampled from the eye)
/// 	CacheInfo prev_cache_info(prev_vertex_info);
/// 
/// 	// determine the cache slot
/// 	uint32 new_cache_slot = prev_cache_info.pixel;
/// 	bool   new_cache_entry = false;
/// 	
/// 	// We should create a new cache entry if and only if:
/// 	//  1. none has been created so far along this path
/// 	//  2. the depth is sufficient
/// 	//  3. other conditions like the hit being at a minimum distance and the sampling probability being low enough (indicating a rough-enough interaction) hold
/// 	if (prev_cache_info.is_invalid() &&
/// 		context.in_bounce >= context.options.psf_depth &&
/// 		p_prev < context.options.psf_max_prob)
/// 	{
/// 		<< Compute per-pixel jittering coordinates >>
/// 		<< Compute a spatial hashkey >>
/// 		<< Insert key into the hashmap >>
/// 		<< Append references to the PSF queue >>
/// 		<< Finalize >>
/// 	}
/// 	return CacheInfo(new_cache_slot, 0, new_cache_entry);
/// }
///\endcode
///
///\par
/// The first step is computing some random numbers to jitter the spatial hashing itself:
/// 
///\anchor Compute_per-pixel_jittering_coordinates_anchor
///>   <i> << Compute per-pixel jittering coordinates >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::compute_jittering_coordinates
///
///\par
/// After that, computing the hash key is fairly straightforward:
/// 
///\anchor Compute_a_spatial_hashkey_anchor
///>   <i> << Compute a spatial hash key >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::compute_spatial_hash
///
///\par
/// and so is insertion into the hashmap:
/// 
///\anchor Insert_key_into_the_hashmap_anchor
///>   <i> << Insert key into the hashmap >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::insert_into_hashmap
///
///\par
/// Finally, we need to append a reference to this newly created cell into the PSF splatting queue,
/// and finalize, marking our cell as new:
/// 
///\anchor Append_references_to_the_PSF_queue_anchor
///>   <i> << Append references to the PSF queue >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::append_refs
/// 
///\anchor Finalize_anchor
///>   <i> << Finalize >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::finalize
///
///\par
/// Notice that the method returns an integer, which is later passed to all other methods as <i>vertex_info</i>.
/// Here we use this integer to pack all the information we'll need later on, that is to say: the hash cell index, a bit flag indicating whether this hash
/// cell has been newly created, i.e. this is <b>the first diffuse vertex</b> along the path where filtering is performed, or whether the cell was already
/// looked-up by some previous vertex, and some extra flags indicating the components this path is sampling (e.g. diffuse or glossy).
/// For convenience, we use a simple helper class to wrap this information in an easily accessible bit field:
///
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::CacheInfo
///
///\par
/// The next method specifies how to compute the Next-Event Estimation weights, separately for diffuse and glossy interactions.
/// Here we have to distinguish a few cases:
/// - the case where this path vertex comes <b>before</b> any hashing is done (as seen from the eye/camera): in this case we'll do business as usual, compute the weights as one would normally do and simply accumulate the resulting samples directly to the framebuffer (though this is done in a separate method)
/// - the case where this path vertex is <b>the vertex</b> where hashing/filtering is done: in this case, we want to accumulate its <i>demodulated</i> diffuse contribution to the hashmap (where by demodulated we mean that we'll remove any high frequency details introduced by the diffuse texture), while we'll accumulate its glossy contribution directly to the framebuffer (as glossy reflections are generally too high-frequency to be cached and filtered in path space)
/// - the case where this path vertex comes <b>after</b> the vertex where caching is done: in this case again we'll do business as usual in terms of weight calculation, except we'll add both contributions to the corresponding hash cell.
///\par
/// Again, this method only takes care of computing the weights, while the actual sample accumulations are done in a different method we'll see in a moment.
/// So in practice, as the first and last case result in the same exact weights and only the central case is different, we can group them into two cases only:
///
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::compute_nee_weights
///
///\par
/// The actual sample accumulation follows from the above mentioned logic. Conceptually, it would be even simpler than it ends up being, except this renderer also keeps track of
/// separate diffuse and glossy channels, which requires some extra special casing based on the path type.
///
///\code
/// template <typename TPTContext>
/// FERMAT_DEVICE
/// void accumulate_nee(
/// 	const TPTContext&			context,
/// 		  RenderingContextView&	renderer,
/// 	const PixelInfo				pixel_info,
/// 	const uint32				vertex_info,
/// 	const bool					shadow_hit,
/// 	const cugar::Vector3f&		w_d,
/// 	const cugar::Vector3f&		w_g)
/// {
/// 	FBufferView& fb = renderer.fb;
/// 	FBufferChannelView& composited_channel = fb(FBufferDesc::COMPOSITED_C);
/// 	FBufferChannelView& direct_channel     = fb(FBufferDesc::DIRECT_C);
/// 	FBufferChannelView& diffuse_channel    = fb(FBufferDesc::DIFFUSE_C);
/// 	FBufferChannelView& specular_channel   = fb(FBufferDesc::SPECULAR_C);
/// 
///		// unpack the pixel index & sampling component
/// 	const uint32 pixel_index = pixel_info.pixel;
/// 	const uint32 pixel_comp  = pixel_info.comp;
/// 	const float frame_weight = context.frame_weight;
/// 
/// 	// access the packed vertex info
/// 	const CacheInfo cache_info(vertex_info);
/// 
/// 	if (shadow_hit == false)
/// 	{		
/// 		// check if the cache cell is valid
/// 		if (cache_info.is_valid())
/// 		{
/// 			<< Accumulate selected components to the cache cell >>
/// 			<< Accumulate remainder to the framebuffer >>
/// 		}
/// 		else
/// 		{
/// 			<< Accumulate every component to the framebuffer >>
/// 		}
/// 	}
/// }
///\endcode
///
///\par
/// You can see that there are again basically two cases: the case there is a valid <i>cache_info</i>, specifying a hash cell to
/// accumulate the sample to, and the opposite case, in which the sample has to be accumulated directly to the framebuffer (which
/// might happen if no diffuse vertex has been found along the path, or the required number of bounces for path-space filtering
/// has not yet been reached).
/// In the first case, we need to first determine which components to add the cache - as specified by <i>cache_info.comp</i> -
/// and then issue an atomic for each of the sample value's components. The atomics are needed to make sure conflicting writes to the
/// same cell are appropriately resolved:
/// 
///\anchor Accumulate_selected_components_to_the_cache_cell_anchor
///>   <i> << Accumulate selected components to the cache cell >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::accumulate_to_cache
///
///\par
/// In case only the diffuse component was accumulated to the cache cell, we need to add the remaining glossy component
/// to the framebuffer, concluding the treatment of the first case:
///
///\anchor Accumulate_remainder_to_the_framebuffer_anchor
///>   <i> << Accumulate remainder to the framebuffer >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::accumulate_remainder_to_framebuffer
///
///\par
/// The second case is conceptually simpler, and more or less the same of what we did in the standard path tracer:
/// 
///\anchor Accumulate_every_component_to_the_framebuffer_anchor
///>   <i> << Accumulate every component to the framebuffer >> := </i> 
///>
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::accumulate_all_to_framebuffer
///
///\par
/// Similar logic applies to the calculation of scattering weights; basically, everything's done as usual in a path tracer, except if we are
/// filtering at the current vertex, in which case we demodulate the weight in order to filter the <i>demodulated</i> path contribution:
///
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::compute_scattering_weights
///
///\par
/// Finally, the last method prescribes what to do with emissive path vertices, and again we apply a logic similar to the above: if the hash cell computed at the <i>previous</i> path vertex is invalid,
/// we splat the sample directly to the framebuffer, otherwise we splat the sample into the hashmap.
/// The reason why we here look at the previous path vertex is that we are looking at emission at the current vertex towards the previous one, sampling what is basically direct lighting at the previous
/// vertex along the path:
///
///\snippet psfpt_vertex_processor.h PSFPTVertexProcessor::accumulate_emissive
///
///\par
/// Here we go, this is pretty much all the magic needed to perform path space filtering within our \ref PTLib framework. Hopefully, this speaks to its flexibility...
///\n
/// Of course, there's many important details we just skimmed over, like the actual implementation of the massively parallel hashmap, but thankfully this is all provided by the underlying
/// \ref cugar_page "CUGAR" library.
///
///\par
/// So here's a comparison of what you get with standard path tracing (top), and path-space filtering (bottom) at 32 samples per pixel.
/// Ideally, you should also think this coding excercise was worth the effort.
/// <img src="pt-32.jpg" style="position:relative; bottom:-10px; border:0px; width:760px;"/>
///\n
/// <img src="psfpt-32.jpg" style="position:relative; bottom:-10px; border:0px; width:760px;"/>
///\n
/// 
/// 
/// Next: \ref BPTLibPage


///@addtogroup Fermat
///@{

///@defgroup PSFPTModule
/// This module defines an implementation of a path tracer using a simplified version of Path Space Filtering.
/// Unlike the original implementation, this version is not progressive, and relies on efficient
/// spatial hashing for defining accumulation neighborhoods, following ideas first developed by
/// Sascha Fricke, Nikolaus Binder and Alex Keller:
///
///>   <a href="https://dl.acm.org/citation.cfm?id=3214806">Fast path space filtering by jittered spatial hashing</a>,
///>   Binder et al, ACM SIGGRAPH 2018 Talks.
///@{

/// PSFPT Options
///
struct PSFPTOptions : PTOptions
{
	uint32  psf_depth;
	float   psf_width;
	float   psf_min_dist;
	float   psf_max_prob;
	uint32  psf_temporal_reuse;
	float	firefly_filter;

	PSFPTOptions() :
		psf_depth(1),
		psf_width(3.0f),
		psf_min_dist(0.1f),
		psf_max_prob(32.0f),
		psf_temporal_reuse(64),
		firefly_filter(100.0f) {}

	void parse(const int argc, char** argv)
	{
		PTOptions::parse(argc, argv);

		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-filter-depth") == 0)
				psf_depth = atoi(argv[++i]);
			else if (strcmp(argv[i], "-filter-width") == 0)
				psf_width = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-filter-min-dist") == 0)
				psf_min_dist = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-filter-max-prob") == 0)
				psf_max_prob = (float)atof(argv[++i]);
			else if (strcmp(argv[i], "-temporal-reuse") == 0)
				psf_temporal_reuse = atoi(argv[++i]);
			else if (strcmp(argv[i], "-firefly-filter") == 0 ||
					 strcmp(argv[i], "-ff") == 0)
				firefly_filter = (float)atof(argv[++i]);
		}
	}
};

/// An implementation of a path tracer using a simplified version of Path Space Filtering.
/// Unlike the original implementation, this version is not progressive, and relies on efficient
/// spatial hashing for defining accumulation neighborhoods, following ideas first developed by
/// Sascha Fricke, Nikolaus Binder and Alex Keller.
///
/// see:
///>   <a href="https://dl.acm.org/citation.cfm?id=3214806">Fast path space filtering by jittered spatial hashing</a>,
///>   Binder et al, ACM SIGGRAPH 2018 Talks.
///
struct PSFPT : RendererInterface
{
	typedef AdaptiveClusteredRLStorage	VTLRLStorage;

	enum PassType {
		kPresamplePass	= 0,
		kFinalPass		= 1
	};

	PSFPT();

	void init(int argc, char** argv, RenderingContext& renderer);

	void render_pass(const uint32 instance, RenderingContext& renderer, const PassType pass_type);

	void render(const uint32 instance, RenderingContext& renderer);

	void keyboard(unsigned char character, int x, int y, bool& invalidate);

	void destroy() { delete this;  }

	static RendererInterface* factory() { return new PSFPT(); }

	// general path tracing members
	DomainBuffer<CUDA_BUFFER, uint8>	m_memory_pool;

	PSFPTOptions				m_options;
	TiledSequence				m_sequence;

	cugar::LFSRGeneratorMatrix  m_generator;
	cugar::LFSRRandomStream		m_random;

	MeshVTLStorage*				m_mesh_vtls;
	VTLRLStorage*				m_vtls_rl;
	cugar::Bbox3f				m_bbox;

	float						m_time;
	PTStats						m_stats;

	// PSFPT specific members
	DeviceHashTable						m_psf_hash;
	DomainBuffer<CUDA_BUFFER, float4>	m_psf_values;
	//DomainBuffer<CUDA_BUFFER, uint32>	m_psf_timestamps;
};

///@} PSFPTModule
///@} Fermat
