/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <nih/bvh/cuda/binned_sah_builder.h>
#include <nih/sampling/random.h>
#include <nih/time/timer.h>
#include <nih/basic/cuda_domains.h>

struct min_functor
{
    __device__ float3 operator() (const float3 a, const float3 b) const
    {
        return make_float3(
            nih::min(a.x,b.x),
            nih::min(a.y,b.y),
            nih::min(a.z,b.z) );
    }
};

namespace nih {

void binned_sah_bvh_test()
{
    fprintf(stderr, "sah bvh test... started\n");

    const uint32 n_objs  = 1024*1024;
    const uint32 n_tests = 10;

    thrust::host_vector<Bbox4f> h_bboxes( n_objs );

    Random random;
    for (uint32 i = 0; i < n_objs; ++i)
        h_bboxes[i] = Bbox4f( Vector4f( random.next(), random.next(), random.next(), 1.0f ) );

    thrust::device_vector<Bbox4f> d_bboxes( h_bboxes );

    thrust::device_vector<Bvh_node> bvh_nodes;
    thrust::device_vector<uint2>    bvh_leaves;
    thrust::device_vector<uint32>   bvh_index;

    cuda::Binned_sah_builder builder( bvh_nodes, bvh_leaves, bvh_index );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    float time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        builder.build(
            16u,
            Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),
            thrust::raw_pointer_cast( &d_bboxes.front() ),
            thrust::raw_pointer_cast( &d_bboxes.front() ) + n_objs,
            thrust::raw_pointer_cast( &h_bboxes.front() ),
            8u,
            1.8f );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    fprintf(stderr, "sah bvh test... done\n");
    fprintf(stderr, "  time     : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  objs/sec : %f M\n", (n_objs / time) / 1.0e6f );
    fprintf(stderr, "  nodes    : %u\n", builder.m_node_count );
    fprintf(stderr, "  leaves   : %u\n", builder.m_leaf_count );
    fprintf(stderr, "  levels   : %u\n", builder.m_level_count );
    fprintf(stderr, "    init bins          : %f ms\n", builder.m_init_bins_time / float(n_tests) );
    fprintf(stderr, "    update bins        : %f ms\n", builder.m_update_bins_time / float(n_tests) );
    fprintf(stderr, "    sah split          : %f ms\n", builder.m_sah_split_time / float(n_tests) );
    fprintf(stderr, "    distribute objects : %f ms\n", builder.m_distribute_objects_time / float(n_tests) );
}

} // namespace nih
