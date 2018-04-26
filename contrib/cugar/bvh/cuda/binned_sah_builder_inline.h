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

#pragma once

#include <nih/basic/utils.h>
#include <nih/basic/cuda/scan.h>
#include <nih/basic/cuda_config.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace nih {
namespace cuda {

#define SAH_SINGLE_WARP 0
#define SAH_MAX_BINS    128
//#define SAH_CHECKS

namespace binned_sah {

void init_bins(const uint32 BINS, const uint32 n_nodes, Bins bins);

void sah_split(
    const uint32    BINS,
    Bins            bins,
    Queue           qin,
    const int       input_node_offset,
    Queue           qout,
    uint32*         n_output,
    int             output_node_offset,
    Bvh_node*       nodes,
    uint32*         n_leaves,
    const uint32    max_leaf_size,
    const float     max_cost);

void distribute_objects(
    const uint32    BINS,
    Objects         objects,
    const int       n_objects,
    Queue           queue,
    const int       input_node_offset,
    Bins            bins);

void setup_leaves(
    const int       n_objects,
    const int32*    leaf_ids,
    uint2*          leaves);


FORCE_INLINE NIH_DEVICE void update_bin(float3* bin_bmin, float3* bin_bmax, int32* bin_counter, const Bbox4f bbox)
{
    const float3 bmin = *bin_bmin;
    if (bbox[0][0] < bmin.x) atomicMin( (int32*)&(bin_bmin->x), __float_as_int(bbox[0][0]) );
    if (bbox[0][1] < bmin.y) atomicMin( (int32*)&(bin_bmin->y), __float_as_int(bbox[0][1]) );
    if (bbox[0][2] < bmin.z) atomicMin( (int32*)&(bin_bmin->z), __float_as_int(bbox[0][2]) );

    const float3 bmax = *bin_bmax;
    if (bbox[1][0] > bmax.x) atomicMax( (int32*)&(bin_bmax->x), __float_as_int(bbox[1][0]) );
    if (bbox[1][1] > bmax.y) atomicMax( (int32*)&(bin_bmax->y), __float_as_int(bbox[1][1]) );
    if (bbox[1][2] > bmax.z) atomicMax( (int32*)&(bin_bmax->z), __float_as_int(bbox[1][2]) );

    atomicAdd( bin_counter, 1 );
}

///
/// CUDA kernel: for each object, update the bin it belongs to among the ones of its parent
/// task/node.
///
template <typename Iterator>
__global__ void update_bins_kernel(
    const uint32    BINS,
    const uint32    n_objects,
    const Iterator  bboxes,
    const Vector4f  origin,
    const Objects   objects,
    const Queue     queue,
    Bins            bins)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_objects;
                base_idx += grid_size)
    {
        const uint32 idx = threadIdx.x + base_idx;

        if (idx >= n_objects)
            return;

        const uint32 id = idx;

        // check if the object has already been assigned to a node
        const int32 node_id = objects.node_ids[id];
        if (node_id > -1)
            continue;

        const int32 split_id = objects.split_ids[id];

	    const Bin node_bbox = queue.bins[split_id];
        const float3 node_size = make_float3(
            node_bbox.bmax.x - node_bbox.bmin.x,
            node_bbox.bmax.y - node_bbox.bmin.y,
            node_bbox.bmax.z - node_bbox.bmin.z );

        Bbox4f bbox = bboxes[id];
        bbox[0] += origin;
        bbox[1] += origin;
        const Vector3f center = (xyz(bbox[0]) + xyz(bbox[1]))*0.5f;

        const int4 bin_id = make_int4(
            (node_size.x < 1.0e-8f ? 0 : quantize( (center[0] - node_bbox.bmin.x) / node_size.x, BINS )),
            (node_size.y < 1.0e-8f ? 0 : quantize( (center[1] - node_bbox.bmin.y) / node_size.y, BINS )),
            (node_size.z < 1.0e-8f ? 0 : quantize( (center[2] - node_bbox.bmin.z) / node_size.z, BINS )),
            0 );

        objects.bin_ids[idx] = bin_id;

        const uint32 binX = split_id + (BINS * 0 + bin_id.x)*queue.size;
        const uint32 binY = split_id + (BINS * 1 + bin_id.y)*queue.size;
        const uint32 binZ = split_id + (BINS * 2 + bin_id.z)*queue.size;

        update_bin( bins.bmin + binX, bins.bmax + binX, (int32*)bins.size + binX, bbox );
        update_bin( bins.bmin + binY, bins.bmax + binY, (int32*)bins.size + binY, bbox );
        update_bin( bins.bmin + binZ, bins.bmax + binZ, (int32*)bins.size + binZ, bbox );
    }
}

///
/// For each object, update the bin it belongs to among the ones of its parent task/node.
///
template <typename Iterator>
inline void update_bins(
    const uint32    BINS,
    const uint32    n_objects,
    const Iterator  bboxes,
    const Vector4f  origin,
    const Objects   objects,
    const Queue     queue,
    Bins            bins)
{
    const uint32 BLOCK_SIZE = SAH_SINGLE_WARP ? 32 : 128;
    const size_t max_blocks = SAH_SINGLE_WARP ? 1 : thrust::detail::backend::cuda::arch::max_active_blocks(update_bins_kernel<Iterator>, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_objects + BLOCK_SIZE-1) / BLOCK_SIZE );

    update_bins_kernel<<<n_blocks,BLOCK_SIZE>>> (
        BINS,
        n_objects,
        bboxes,
        origin,
        objects,
        queue,
        bins );

    cudaThreadSynchronize();
}

inline void start_timer(const cudaEvent_t start, const cudaEvent_t stop)
{
    cudaEventRecord( start, 0 );
}
inline float stop_timer(const cudaEvent_t start, const cudaEvent_t stop)
{
    float dtime;
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &dtime, start, stop );
    return dtime;
}

struct Bin_counter
{
    typedef Bin         argument_type;
    typedef uint32      result_type;

    NIH_HOST_DEVICE uint32 operator() (const Bin bin) const
    {
        return binary_cast<int32>(bin.bmin.w);
    }
};

} // namespace binned_sah

// build a bvh given a set of bboxes
template <typename Iterator>
void Binned_sah_builder::build(
    const uint32    BINS,
    const Bbox3f    bbox,
    const Iterator  bbox_begin,
    const Iterator  bbox_end,
const Iterator  h_bbox_begin,
    const uint32    max_leaf_size,
    const float     max_cost)
{
    const uint32 n_objects = uint32( bbox_end - bbox_begin );

    need_space( *m_nodes,  n_objects*2 );
    need_space( *m_leaves, n_objects );
    need_space( *m_index,  n_objects );

    need_space( m_bin_bmin, (n_objects / max_leaf_size) * BINS * 3 ); // might need more later on...
    need_space( m_bin_bmax, (n_objects / max_leaf_size) * BINS * 3 ); // might need more later on...
    need_space( m_bin_size, (n_objects / max_leaf_size) * BINS * 3 ); // might need more later on...
    need_space( m_queue_bins,    n_objects * 2 );
    need_space( m_queue_splits,  n_objects * 2 );

    need_space( m_bin_ids,   n_objects );
    need_space( m_split_ids, n_objects );
    need_space( m_node_ids,  n_objects );

    Queue queue[2];

    queue[0].bins   = thrust::raw_pointer_cast( &m_queue_bins.front() );
    queue[0].splits = thrust::raw_pointer_cast( &m_queue_splits.front() );
    queue[1].bins   = thrust::raw_pointer_cast( &m_queue_bins.front() )   + n_objects;
    queue[1].splits = thrust::raw_pointer_cast( &m_queue_splits.front() ) + n_objects;

    Objects objects;
    objects.bin_ids   = thrust::raw_pointer_cast( &m_bin_ids.front() );
    objects.node_ids  = thrust::raw_pointer_cast( &m_node_ids.front() );
    objects.split_ids = thrust::raw_pointer_cast( &m_split_ids.front() );

    // assign all objects to split task 0 and node -1
    thrust::fill( m_split_ids.begin(), m_split_ids.begin() + n_objects, 0 );
    thrust::fill( m_node_ids.begin(),  m_node_ids.begin() + n_objects, -1 );

    // initialize root bounding box
    {
        Bin bin;
        bin.bmin = make_float4( 0.0f,0.0f,0.0f,binary_cast<float>(n_objects) );
        bin.bmax = make_float4( bbox[1][0]-bbox[0][0],bbox[1][1]-bbox[0][1],bbox[1][2]-bbox[0][2],binary_cast<float>(Bvh_node::kInvalid) );
        m_queue_bins[0] = bin;
    }

    int input_node_offset  = 0;
    int output_node_offset = 1;

    m_counters.resize(2);
    m_counters[0] = 0;
    m_counters[1] = 0;

    int in_queue      = 0;
    int out_queue     = 1;
    int n_input_tasks = 1;

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    Bins bins;

    m_levels[0] = 0;
    int32 level = 0;

    // keep processing nodes in the input task queue until there's no more output
    while (n_input_tasks)
    {
        // mark the beginning of the new level
        m_levels[ level++ ] = input_node_offset;

        need_space( m_bin_bmin, n_input_tasks * BINS * 3 );
        need_space( m_bin_bmax, n_input_tasks * BINS * 3 );
        need_space( m_bin_size, n_input_tasks * BINS * 3 );

        bins.bmin   = thrust::raw_pointer_cast( &m_bin_bmin.front() );
        bins.bmax   = thrust::raw_pointer_cast( &m_bin_bmax.front() );
        bins.size   = thrust::raw_pointer_cast( &m_bin_size.front() );

        // reset the output task counter
        m_counters[0] = 0;

        // set queue size
        queue[ in_queue ].size = n_input_tasks;

        binned_sah::start_timer( start, stop );
        //binned_sah::init_bins( BINS, n_input_tasks, bins );
        {
            const float HUGE = 1.0e8f;
            thrust::fill( m_bin_bmin.begin(), m_bin_bmin.begin() + n_input_tasks * BINS * 3, make_float3(  HUGE,  HUGE,  HUGE ) );
            thrust::fill( m_bin_bmax.begin(), m_bin_bmax.begin() + n_input_tasks * BINS * 3, make_float3( -HUGE, -HUGE, -HUGE ) );
            thrust::fill( m_bin_size.begin(), m_bin_size.begin() + n_input_tasks * BINS * 3, 0 );
        }

        m_init_bins_time += binned_sah::stop_timer( start, stop );

        binned_sah::start_timer( start, stop );
        binned_sah::update_bins( BINS, n_objects, bbox_begin, Vector4f(-bbox[0][0],-bbox[0][1],-bbox[0][2],0.0f), objects, queue[ in_queue ], bins);
        m_update_bins_time += binned_sah::stop_timer( start, stop );

        binned_sah::start_timer( start, stop );
        binned_sah::sah_split(
            BINS,
            bins,
            queue[ in_queue ],
            input_node_offset,
            queue[ out_queue ],
            thrust::raw_pointer_cast( &m_counters.front() ),
            output_node_offset,
            thrust::raw_pointer_cast( &m_nodes->front() ),
            thrust::raw_pointer_cast( &m_counters.front() ) + 1,
            max_leaf_size,
            max_cost );
        m_sah_split_time += binned_sah::stop_timer( start, stop );

        binned_sah::start_timer( start, stop );

        binned_sah::distribute_objects(
            BINS,
            objects,
            n_objects,
            queue[ in_queue ],
            input_node_offset,
            bins );

        m_distribute_objects_time += binned_sah::stop_timer( start, stop );

        // get the new number of generated tasks
        const uint32 n_output_tasks = m_counters[0];

        // update input & output counters
        input_node_offset   = output_node_offset;
        output_node_offset += n_output_tasks;
        n_input_tasks       = n_output_tasks;

        // swap the input & output queues
        std::swap( in_queue, out_queue );
    }

    m_level_count = level;
    for (; level < 128; ++level)
        m_levels[ level ] = output_node_offset;

    // sort the objects by their leaf id
    thrust::copy( thrust::make_counting_iterator(0u), thrust::make_counting_iterator(0u) + n_objects, m_index->begin() );
    thrust::sort_by_key( m_node_ids.begin(), m_node_ids.begin() + n_objects, m_index->begin() );

    // setup leaf ranges
    binned_sah::setup_leaves( n_objects, objects.node_ids, thrust::raw_pointer_cast( &m_leaves->front() ) );

    m_leaf_count = m_counters[1];
    m_node_count = output_node_offset;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}

} // namespace cuda
} // namespace nih
