/* Copyright 2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <limits>
#include <iostream>
#include <type_traits>
#include <cstdio>
#include <cassert>
#include <cmath>

//#define HAVE_CURAND
#ifdef HAVE_CURAND
#include <curand_kernel.h>
#endif

#include <thrust/gather.h>

#include <cub/block/block_reduce.cuh>

#include <vector_types.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/bvh/cuda/lbvh_builder.h>
#include <cugar/tree/cuda/reduce.h>
#include <cugar/bintree/bintree_visitor.h>
#include <thrust/iterator/transform_iterator.h>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}
#endif

namespace {

#if defined(HAVE_CURAND)
__global__
void init_curand(curandState * state, const int n, const unsigned long long seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n ) {
        curand_init( seed, i, 0, state+i );
    }
}

__global__ void generate_points_kernel(
    float4* __restrict__ const pts, const int num_pts,
    curandState * state, const int num_states )
{
    const int start_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride    = blockDim.x*gridDim.x;
    assert( start_idx < num_states );
    
    curandState localState = state[start_idx];
    
    for ( int idx = start_idx; idx < num_pts; idx += stride )
    {
        float4 point;
        point.x = curand_uniform_double(&localState);
        point.y = curand_uniform_double(&localState);
		point.z = curand_uniform_double(&localState);
		point.w = 0.0f;
        pts[idx] = point;
    }
    state[start_idx] = localState;
}

void generate_points( float4* __restrict__ const pts, const int num_pts )
{
    const unsigned long long seed = 1234ULL;
    const int block_size = 128;
    //maximize exposed parallelism while minimizing storage for curand state
    int num_blocks_generate_points_kernel = 1;
    CUDA_RT_CALL( cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &num_blocks_generate_points_kernel, generate_points_kernel , block_size, 0 ) );
    
    int dev_id = 0;
    CUDA_RT_CALL( cudaGetDevice( &dev_id ) );
    int num_sms = 0;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &num_sms, cudaDevAttrMultiProcessorCount, dev_id ) );

    const int num_states = num_sms*num_blocks_generate_points_kernel*block_size;
    curandState * devStates;
    CUDA_RT_CALL( cudaMalloc( &devStates, num_states*sizeof(curandState) ) );
    init_curand<<<((num_states-1)/block_size)+1,block_size>>>( devStates, num_states, seed );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );

    generate_points_kernel<<<num_sms*num_blocks_generate_points_kernel,block_size>>>( pts, num_pts, devStates, num_states );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    CUDA_RT_CALL( cudaFree( devStates ) );
}
#else
__global__ void generate_points_kernel(
    float4* __restrict__ const pts, const int num_pts )
{
    const int start_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride    = blockDim.x*gridDim.x;
    
    for ( int idx = start_idx; idx < num_pts; idx += stride )
    {
        float4 point;
        point.x = cugar::randfloat( idx, 0 );
        point.y = cugar::randfloat( idx, 1 );
		point.z = cugar::randfloat( idx, 2 );
		point.w = 0.0f;
        pts[idx] = point;
    }
}
void generate_points( float4* __restrict__ const pts, const int num_pts )
{
    const int block_size = 128;
    //maximize exposed parallelism while minimizing storage for curand state
    int num_blocks_generate_points_kernel = 1;
    CUDA_RT_CALL( cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &num_blocks_generate_points_kernel, generate_points_kernel , block_size, 0 ) );
    
    int dev_id = 0;
    CUDA_RT_CALL( cudaGetDevice( &dev_id ) );
    int num_sms = 0;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &num_sms, cudaDevAttrMultiProcessorCount, dev_id ) );

    generate_points_kernel<<<num_sms*num_blocks_generate_points_kernel,block_size>>>( pts, num_pts );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
}
#endif

__host__ __device__  __forceinline
bool point_in_bbox(const float4 a, const float4 b, const float eps)
{
  return 
	  fabsf( a.x - b.x ) <= eps &&
	  fabsf( a.y - b.y ) <= eps &&
	  fabsf( a.z - b.z ) <= eps;
}

template<int BLOCK_SIZE>
__global__
void count_neighbors(
    int* __restrict__ const num_neighbors,
    const float4* __restrict__ const pts, const int num_pts,
    const float eps )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float4 pts_shared[2][BLOCK_SIZE];

    int n_neighbors = 0;
    float4 point;
    if ( i < num_pts )
        point = pts[i];
    
    int curr_buff = 0;
    if ( threadIdx.x < num_pts )
        pts_shared[curr_buff][threadIdx.x] = pts[threadIdx.x];
    
    for ( int j = threadIdx.x; j < (num_pts/BLOCK_SIZE)*BLOCK_SIZE; j+=BLOCK_SIZE ) {
        __syncthreads();
        if ( BLOCK_SIZE+j < (num_pts/BLOCK_SIZE)*BLOCK_SIZE )
            pts_shared[(curr_buff+1)%2][threadIdx.x] = pts[BLOCK_SIZE+j];
        
        #pragma unroll 16
        for ( int k = 0; k < BLOCK_SIZE; ++k ) {
            if ( i < num_pts && i != ((j-threadIdx.x)+k) && point_in_bbox( pts_shared[curr_buff][k], point, eps ) )
                ++n_neighbors;
        }
        curr_buff = (curr_buff+1)%2;
    }
    
    if ( i < num_pts ) {
        for ( int j = (num_pts/BLOCK_SIZE)*BLOCK_SIZE; j < num_pts; ++j ) {
            if ( i!=j && point_in_bbox( pts[j], point, eps ) )
                ++n_neighbors;
        }
    }
    
    num_neighbors[i] = n_neighbors;
}

template<int BLOCK_SIZE>
__global__
void calc_bounding_box(
    const float4* __restrict__ const pts, const int num_pts,
    float4* __restrict__ const min, float4* __restrict__ const max)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int start_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride    = blockDim.x*gridDim.x;
    
    float4 local_min = cugar::Vector4f(  1.0e8f );
    float4 local_max = cugar::Vector4f( -1.0e8f );
    
    for ( int idx = start_idx; idx < num_pts; idx += stride )
    {
        const float4 point = pts[idx];
        
        local_min.x = point.x < local_min.x ? point.x : local_min.x;
        local_min.y = point.y < local_min.y ? point.y : local_min.y;
        local_min.z = point.z < local_min.z ? point.z : local_min.z;
        
        local_max.x = point.x > local_max.x ? point.x : local_max.x;
        local_max.y = point.y > local_max.y ? point.y : local_max.y;
		local_max.z = point.z > local_max.z ? point.z : local_max.z;
    }
    
    local_min.x = BlockReduce(temp_storage).Reduce(local_min.x, cub::Min());
    __syncthreads();
    local_min.y = BlockReduce(temp_storage).Reduce(local_min.y, cub::Min());
    __syncthreads();
    local_min.z = BlockReduce(temp_storage).Reduce(local_min.z, cub::Min());
    __syncthreads();

    local_max.x = BlockReduce(temp_storage).Reduce(local_max.x, cub::Max());
    __syncthreads();
    local_max.y = BlockReduce(temp_storage).Reduce(local_max.y, cub::Max());
    __syncthreads();
    local_max.z = BlockReduce(temp_storage).Reduce(local_max.z, cub::Max());
    __syncthreads();
    
    if ( 0 == threadIdx.x ) {
        atomicMin( (int*)&(min->x), __float_as_int(local_min.x) );
        atomicMin( (int*)&(min->y), __float_as_int(local_min.y) );
		atomicMin( (int*)&(min->z), __float_as_int(local_min.z) );
        atomicMax( (int*)&(max->x), __float_as_int(local_max.x) );
        atomicMax( (int*)&(max->y), __float_as_int(local_max.y) );
		atomicMax( (int*)&(max->z), __float_as_int(local_max.z) );
    }
}

template<int BLOCK_SIZE>
__global__
void check_results(
    int* __restrict__ const num_errors_d,
    const int* __restrict__ const num_neighbors,
    const int* __restrict__ const num_neighbors_simple,
    const int num_pts )
{
    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int start_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride    = blockDim.x*gridDim.x;
    
    int num_errors = 0;
    
    for ( int idx = start_idx; idx < num_pts; idx += stride )
    {
        if ( num_neighbors[idx] != num_neighbors_simple[idx] )
            ++num_errors;
    }
    
    num_errors = BlockReduce(temp_storage).Sum(num_errors);
    __syncthreads();
    if ( 0 == threadIdx.x )
        atomicAdd(num_errors_d, num_errors);
}

__global__
void calc_stats(
    int2* __restrict__ const min_max_neighbors,
    unsigned long long int* __restrict__ const total_neighbors,
    const int* __restrict__ const num_neighbors, const int num_pts)
{
    typedef cub::BlockReduce<int, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int start_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride    = blockDim.x*gridDim.x;

    int my_total_neighbors = 0;
    int my_min_neighbors = 2147483647;
    int my_max_neighbors = 0;

    for ( int idx = start_idx; idx < num_pts; idx += stride )
    {
        int n_neighbors = num_neighbors[idx];
        my_min_neighbors = n_neighbors < my_min_neighbors? n_neighbors : my_min_neighbors;
        my_max_neighbors = n_neighbors > my_max_neighbors? n_neighbors : my_max_neighbors;
        my_total_neighbors += n_neighbors;
    }
    
    my_min_neighbors = BlockReduce(temp_storage).Reduce(my_min_neighbors, cub::Min());
    __syncthreads();
    if ( 0 == threadIdx.x )
        atomicMin(&(min_max_neighbors->x), my_min_neighbors);
    
    my_max_neighbors = BlockReduce(temp_storage).Reduce(my_max_neighbors, cub::Max());
    __syncthreads();
    if ( 0 == threadIdx.x )
        atomicMax(&(min_max_neighbors->y), my_max_neighbors);
    
    my_total_neighbors = BlockReduce(temp_storage).Sum(my_total_neighbors);
    __syncthreads();
    if ( 0 == threadIdx.x )
        atomicAdd(total_neighbors, my_total_neighbors);
}

using cugar::uint32;

struct floar4tovec3
{
	typedef cugar::Vector3f result_type;
	typedef float4			argument_type;

	CUGAR_HOST_DEVICE
	cugar::Vector3f operator() (const float4 v) const { return cugar::Vector3f(v.x,v.y,v.z); }
};

struct leaf_index_functor
{
	typedef uint32					result_type;
	typedef cugar::Bvh_node_3d		argument_type;

	CUGAR_HOST_DEVICE
	result_type operator() (const argument_type node) const { return node.get_child_index(); }
};

struct is_leaf_functor
{
	typedef bool					result_type;
	typedef cugar::Bvh_node_3d		argument_type;

	CUGAR_HOST_DEVICE
	result_type operator() (const argument_type node) const { return node.is_leaf() ? true : false; }
};

struct Bbox_merge_functor
{
    // merge two points
    CUGAR_HOST_DEVICE
	cugar::Bbox3f operator() (
        const float4 pnt1,
        const float4 pnt2) const
    {
        // Build a bbox for each of the two points
        cugar::Bbox3f bbox1(
            cugar::Vector3f( pnt1.x - pnt1.w, pnt1.y - pnt1.w, pnt1.z - pnt1.w ),
            cugar::Vector3f( pnt1.x + pnt1.w, pnt1.y + pnt1.w, pnt1.z + pnt1.w ) );
        cugar::Bbox3f bbox2(
            cugar::Vector3f( pnt2.x - pnt2.w, pnt2.y - pnt2.w, pnt2.z - pnt2.w ),
            cugar::Vector3f( pnt2.x + pnt2.w, pnt2.y + pnt2.w, pnt2.z + pnt2.w ) );

        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }

    // merge a bbox and a point
    CUGAR_HOST_DEVICE
	cugar::Bbox3f operator() (
        const cugar::Bbox3f	bbox1,
        const float4		pnt2) const
    {
        cugar::Bbox3f bbox2(
            cugar::Vector3f( pnt2.x - pnt2.w, pnt2.y - pnt2.w, pnt2.z - pnt2.w ),
            cugar::Vector3f( pnt2.x + pnt2.w, pnt2.y + pnt2.w, pnt2.z + pnt2.w ) );

        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }

    // merge two bboxes
    CUGAR_HOST_DEVICE cugar::Bbox3f operator() (
        const cugar::Bbox3f bbox1,
        const cugar::Bbox3f	bbox2) const
    {
        cugar::Bbox3f result;
        result.insert( bbox1 );
        result.insert( bbox2 );
        return result;
    }
};

CUGAR_HOST_DEVICE
inline uint32 count(const cugar::Bvh_node_3d* nodes, const uint32* skip_nodes, const float4* points, const cugar::Bbox3f query, const float4 query_pt, const float eps, const uint32 query_idx)
{
	uint32 counter = 0;

	// start from the corresponding root node
    uint32_t node_index = 0u;

    // traverse until we land on the invalid node
    while (node_index != uint32(-1))
    {
        // fetch the current node
        const cugar::Bvh_node_3d node = cugar::Bvh_node_3d::load_ldg( nodes + node_index );

    //#define COMBINED_CHILDREN_TEST
    #if defined(COMBINED_CHILDREN_TEST)
        if (!node.is_leaf())
        {
            const uint32_t child_index = node.get_child_index();

			cugar::Bvh_node_3d child1 = cugar::Bvh_node_3d::load_ldg( nodes + child_index );
			cugar::Bvh_node_3d child2 = cugar::Bvh_node_3d::load_ldg( nodes + child_index + 1 );

            if (query[1].x >= child1.bbox[0].x &&
				query[1].y >= child1.bbox[0].y &&
				query[1].z >= child1.bbox[0].z &&
				query[0].x <= child1.bbox[1].x &&
				query[0].y <= child1.bbox[1].y &&
				query[0].z <= child1.bbox[1].z)
            {
                node_index = child_index;
                continue;
            }

            if (query[1].x >= child2.bbox[0].x &&
				query[1].y >= child2.bbox[0].y &&
				query[1].z >= child2.bbox[0].z &&
				query[0].x <= child2.bbox[1].x &&
				query[0].y <= child2.bbox[1].y &&
				query[0].z <= child2.bbox[1].z)
            {
                node_index = child_index + 1;
                continue;
            }
        }
    #else
        // test the bbox independently of whether this node is internal or a leaf
        if (query[0].x > node.bbox[1].x ||
			query[0].y > node.bbox[1].y ||
			query[0].z > node.bbox[1].z ||
			query[1].x < node.bbox[0].x ||
			query[1].y < node.bbox[0].y ||
			query[1].z < node.bbox[0].z)
        {
            // jump to the skip node
            node_index = skip_nodes[ node_index ];
            continue;
        }

        if (!node.is_leaf())
        {
            // step directly into the child, without any test
            node_index = node.get_child_index();
            continue;
        }
    #endif
        else
        {
		#if 1
			// perform all point-in-bbox tests for all points in the leaf
			const uint2 leaf = node.get_leaf_range();
            for (unsigned int j = leaf.x; j < leaf.y; ++j)
            {
				//if (query_idx != j && cugar::contains( query, cugar::Vector4f( points[j] ).xyz() ))
				if (query_idx != j && point_in_bbox( points[j], query_pt, eps ))
                    ++counter;
            }
		#else
			counter += node.get_range_size();
		#endif
        }

        // jump to the skip node
        node_index = skip_nodes[ node_index ];
    }

    return counter;
}

__global__
void count_neighbors_indexed(
    int* __restrict__ const num_neighbors,
    const float4* __restrict__ const pts, const int num_pts,
    const float eps,
    const cugar::Bvh_node_3d* __restrict__ const bvh_nodes,
	const uint32* __restrict__ const bvh_skip_nodes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < num_pts ) {
        const float4 point = pts[i];
        
		cugar::Bbox3f query_bbox(
			cugar::Vector3f( point.x - eps, point.y - eps, point.z - eps ),
			cugar::Vector3f( point.x + eps, point.y + eps, point.z + eps )
		);

        num_neighbors[i] = (int)count( bvh_nodes, bvh_skip_nodes, pts, query_bbox, point, eps, i );
    }
}

} // anonymous namespace

int bvh_range_query_test(int argc, char * argv[])
{
    int dev_id  = 0;
    int num_pts = 1 << 20;
    float eps   = 0.05f;
    bool csv    = false;

	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-dev_id") == 0)
			dev_id = atoi(argv[++i]);
		else if (strcmp(argv[i], "-num_pts") == 0)
			num_pts = atoi(argv[++i]);
		else if (strcmp(argv[i], "-eps") == 0)
			eps = (float)atof(argv[++i]);
		else if (strcmp(argv[i], "-csv") == 0)
			csv = atoi(argv[++i]) ? true : false;
	}
    
    CUDA_RT_CALL( cudaSetDevice(dev_id) );
    
    int num_sms = 0;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &num_sms, cudaDevAttrMultiProcessorCount, dev_id ) );
    int threads_per_sm = 0;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev_id ) );
    
	try
    {
		cugar::device_vector<float4> pts( num_pts );
		cugar::device_vector<float4> pts_sorted( num_pts );
        generate_points( cugar::raw_pointer(pts), num_pts );
        
		cugar::device_vector<int> num_neighbors_simple( num_pts, 0 );
		cugar::device_vector<int> num_neighbors_simple_sorted( num_pts, 0 );
		cugar::device_vector<int> num_neighbors( num_pts, 0 );
        
        CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaDeviceSynchronize() );

		constexpr int block_size = 128;

		cugar::cuda::Timer timer;
		timer.start();

		count_neighbors<block_size><<<(num_pts+block_size-1)/block_size,block_size>>>( cugar::raw_pointer(num_neighbors_simple), cugar::raw_pointer(pts), num_pts, eps );

		CUDA_RT_CALL( cudaGetLastError() );
        CUDA_RT_CALL( cudaDeviceSynchronize() );
		timer.stop();
        const float runtime_simple = timer.seconds();

		cugar::device_var<float4> min_coord( cugar::Vector4f(  1.0e8f ) );
		cugar::device_var<float4> max_coord( cugar::Vector4f( -1.0e8f ) );
        
		thrust::device_ptr<cugar::Vector4f>						bvh_points( static_cast<cugar::Vector4f*>(cugar::raw_pointer(pts)) );
		thrust::device_ptr<cugar::Vector4f>						bvh_points_sorted( static_cast<cugar::Vector4f*>(cugar::raw_pointer(pts_sorted)) );
		cugar::vector<cugar::device_tag, cugar::Bvh_node_3d>	bvh_nodes;
		cugar::vector<cugar::device_tag, cugar::uint32>			bvh_index;
		cugar::vector<cugar::device_tag, cugar::uint32>			bvh_parents;
        cugar::vector<cugar::device_tag, cugar::uint32>			bvh_skip_nodes;
        cugar::vector<cugar::device_tag, uint2>					bvh_ranges;

		cugar::cuda::LBVH_builder<cugar::uint64,cugar::Bvh_node_3d> bvh_builder( &bvh_nodes, &bvh_index, NULL, NULL, &bvh_parents, &bvh_skip_nodes, &bvh_ranges );

		float runtime_index_build = 0.0f;

		for (uint32 i = 0; i < 2; ++i)
		{
			timer.start();

			constexpr int cbbsize = 1024;
			calc_bounding_box<cbbsize><<<(threads_per_sm/cbbsize)*num_sms,cbbsize>>>(cugar::raw_pointer(pts), num_pts, cugar::raw_pointer(min_coord), cugar::raw_pointer(max_coord) );

			cugar::Bbox3f bbox = cugar::Bbox3f(
				cugar::Vector4f(min_coord).xyz(),
				cugar::Vector4f(max_coord).xyz() );

			bvh_builder.build( bbox, thrust::make_transform_iterator(bvh_points, floar4tovec3()), thrust::make_transform_iterator(bvh_points + num_pts, floar4tovec3()), 16u );

			// sort the points
			thrust::gather(
				bvh_index.begin(),
				bvh_index.end(),
				bvh_points,
				bvh_points_sorted );

			// build its bounding boxes
			cugar::Bintree_visitor<cugar::Bvh_node_3d> bvh;
			bvh.set_node_count( bvh_builder.m_node_count );
			bvh.set_leaf_count( bvh_builder.m_leaf_count );
			bvh.set_nodes( cugar::raw_pointer( bvh_nodes ) );
			bvh.set_parents( cugar::raw_pointer( bvh_parents ) );

			cugar::Bvh_node_3d_bbox_iterator bbox_iterator( cugar::raw_pointer( bvh_nodes ) );

			cugar::cuda::tree_reduce(
				bvh,
				bvh_points_sorted,
				bbox_iterator,
				Bbox_merge_functor(),
				cugar::Bbox3f() );
		
			timer.stop();

			if (i)
				runtime_index_build = timer.seconds();

			CUDA_RT_CALL( cudaDeviceSynchronize() );
		}

		float runtime = 0.0f;
		for (unsigned i = 0; i <= 10; ++i)
		{
			timer.start();
        
			count_neighbors_indexed<<<(num_pts+block_size-1)/block_size,block_size>>>(
				cugar::raw_pointer(num_neighbors),
				cugar::raw_pointer(pts_sorted),
				num_pts,
				eps,
				cugar::raw_pointer(bvh_nodes),
				cugar::raw_pointer(bvh_skip_nodes) );

			timer.stop();
			if (i) // skip a warmup call
				runtime += timer.seconds() / 10;

			CUDA_RT_CALL( cudaGetLastError() );
			CUDA_RT_CALL( cudaDeviceSynchronize() );
		}

		cugar::device_var<int> num_errors = 0;
		if (0)
        {
            thrust::gather(
                bvh_index.begin(),
                bvh_index.end(),
                thrust::device_ptr<int>(cugar::raw_pointer(num_neighbors_simple)),
                thrust::device_ptr<int>(cugar::raw_pointer(num_neighbors_simple_sorted)) );
            
            check_results<block_size><<<(num_pts+block_size-1)/block_size,block_size>>>(
				cugar::raw_pointer( num_errors ),
				cugar::raw_pointer( num_neighbors ),
				cugar::raw_pointer( num_neighbors_simple_sorted ),
				num_pts );

			CUDA_RT_CALL( cudaGetLastError() );
            CUDA_RT_CALL( cudaDeviceSynchronize() );
        }

		if ( 0 == num_errors )
        {
			cugar::device_var<int2> min_max_neighbors = make_int2( std::numeric_limits<int>::max(), 0 );
            cugar::device_var<cugar::uint64> total_neighbors = 0;
            
            calc_stats<<<std::min( std::max(num_pts/block_size, 1), num_sms*(threads_per_sm/block_size) ),block_size>>>(
				cugar::raw_pointer( min_max_neighbors ),
				cugar::raw_pointer( total_neighbors ),
				cugar::raw_pointer( num_neighbors ),
				num_pts );

			CUDA_RT_CALL( cudaGetLastError() );
            CUDA_RT_CALL( cudaDeviceSynchronize() );
                        
            if (csv) {
                std::cout<<num_pts<<","<<runtime_simple<<","<<runtime<<","<<runtime_index_build<<std::endl;
            } else {
                std::cout<<"num_pts = "<<num_pts<<std::endl;
                std::cout<<"eps     = "<<eps<<std::endl;
                std::cout<<"Min     = "<< int2(min_max_neighbors).x<<std::endl;
                std::cout<<"Max     = "<< int2(min_max_neighbors).y<<std::endl;
                std::cout<<"Avg     = "<<(1.0*cugar::uint64(total_neighbors))/num_pts<<std::endl;
                
                std::cout<<"Simple:"<<std::endl;
                std::cout<<"  Performance [Mpts/s]: "<< (0.000001*num_pts)/runtime_simple<<std::endl;
                std::cout<<"  Runtime          [s]: "<< runtime_simple << std::endl;
                
                std::cout<<"Indexed:"<<std::endl;
                std::cout<<"  Performance [Mpts/s]: "<< (0.000001*num_pts)/runtime << std::endl;
                std::cout<<"  Runtime          [s]: "<< runtime + runtime_index_build << " = " << runtime_index_build << " + " << runtime << std::endl;
            }
        }
        else
        {
            std::cerr<<"There have been "<< int(num_errors) <<" errors!"<<std::endl;
			return num_errors;
        }
    }
	catch (cugar::bad_alloc& error)
	{	
		fprintf(stderr, "cugar::bad_alloc: %s!\n", error.what());
		exit(0);
	}
	catch (cugar::runtime_error& error)
	{	
		fprintf(stderr, "cugar::runtime_error: %s!\n", error.what());
		exit(0);
	}
	catch (cugar::cuda_error& error)
	{	
		fprintf(stderr, "cugar::cuda_error: %s!\n", error.what());
		exit(0);
	}
	catch (thrust::system_error& error)
	{	
		fprintf(stderr, "thrust system_error: %s!\n", error.what());
		exit(0);
	}
	catch (...)
	{	
		fprintf(stderr, "uncaught exception!\n");
		exit(0);
	}
	//CUDA_RT_CALL( cudaDeviceReset() );
    return 0;
}
