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

//#define BARYCENTRIC_HIT_POINT

#include <bpt.h>
#include <bpt_impl.h>

BPT::BPT() :
	m_generator(32, cugar::LFSRGeneratorMatrix::GOOD_PROJECTIONS),
	m_random(&m_generator, 1u, 1351u)
{
}

void BPT::init(int argc, char** argv, RenderingContext& renderer)
{
	const uint2 res = renderer.res();
	const uint32 n_pixels = res.x * res.y;

	const uint32 n_light_paths = n_pixels;

	fprintf(stderr, "  creating mesh lights... started\n");

	// initialize the mesh lights sampler
	renderer.get_mesh_lights().init( n_light_paths, renderer, 0u );

	fprintf(stderr, "  creating mesh lights... done\n");

	// parse the options
	m_options.parse(argc, argv);

	// if we perform a single connection, RR must be enabled
	if (m_options.single_connection)
		m_options.rr = true;

	fprintf(stderr, "  BPT settings:\n");
	fprintf(stderr, "    single-conn    : %u\n", m_options.single_connection);
	fprintf(stderr, "    RR             : %u\n", m_options.rr);
	fprintf(stderr, "    path-length    : %u\n", m_options.max_path_length);
	fprintf(stderr, "    direct-nee     : %u\n", m_options.direct_lighting_nee ? 1 : 0);
	fprintf(stderr, "    direct-bsdf    : %u\n", m_options.direct_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    indirect-nee   : %u\n", m_options.indirect_lighting_nee ? 1 : 0);
	fprintf(stderr, "    indirect-bsdf  : %u\n", m_options.indirect_lighting_bsdf ? 1 : 0);
	fprintf(stderr, "    visible-lights : %u\n", m_options.visible_lights ? 1 : 0);
	fprintf(stderr, "    light-tracing  : %f\n", m_options.light_tracing);

	const uint32 queue_size = cugar::max(cugar::max(n_pixels, n_light_paths) * 3, n_light_paths * m_options.max_path_length);

	fprintf(stderr, "  allocating ray queue storage... started (%u entries)\n", queue_size);

	// compensate for the amount of light vs eye subpaths, changing the screen sampling densities
	m_options.light_tracing *= float(n_light_paths) / float(n_pixels);

	// alloc ray queues
	m_queues.alloc(n_pixels, n_light_paths, m_options.max_path_length);
	
	fprintf(stderr, "  allocating ray queue storage...done\n");

	// build the set of shifts
	const uint32 n_dimensions = (m_options.max_path_length + 1) * 2 * 6;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	m_sequence.setup(n_dimensions, SHIFT_RES);

	fprintf(stderr, "  allocating light vertex storage... started (%u paths, %u vertices)\n", n_light_paths, n_light_paths * m_options.max_path_length);
	m_light_vertices.alloc(n_light_paths, n_light_paths * m_options.max_path_length);
	fprintf(stderr, "  allocating light vertex storage... done\n");

	if (m_options.use_vpls)
		cudaMemcpy(m_light_vertices.vertex.ptr(), renderer.get_mesh_lights().get_vpls(), sizeof(VPL)*n_light_paths, cudaMemcpyDeviceToDevice);

	m_n_light_subpaths = n_light_paths;
	m_n_eye_subpaths   = n_pixels;
}

void BPT::regenerate_primary_light_vertices(const uint32 instance, RenderingContext& renderer)
{
	// regenerate the VPLs and use them as primary light vertices
	renderer.get_mesh_lights().init(
		m_n_light_subpaths,
		renderer,
		instance);

	if (m_options.use_vpls)
		cudaMemcpy(m_light_vertices.vertex.ptr(), renderer.get_mesh_lights().get_vpls(), sizeof(VPL)*m_n_light_subpaths, cudaMemcpyDeviceToDevice);
}
