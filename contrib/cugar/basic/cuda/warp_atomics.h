/*
 * CUGAR : Cuda Graphics Accelerator
 *
 * Copyright (c) 2010-2018, NVIDIA Corporation
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

/*! \file warp_atomics.h
 *   \brief Define CUDA based warp adders.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/bits/popcount.h>
#include <cub/cub.cuh>

namespace cugar {
namespace cuda {

///@addtogroup Basic
///@{

///@addtogroup CUDAModule
///@{

///@defgroup CUDAAtomicsModule CUDA Atomics
/// This module various types of CUDA atomics
///@{

/// An efficient warp-synchronous atomic adder, to add or subtract compile-time constants to a shared integer.
///
/// Given a pointer to an integer (e.g. representing a "pool"), this class allows the threads
/// in a warp to add (allocate) or subtract (deallocate) a compile-time constant to that integer
/// in a predicated fashion.
///
struct warp_static_atomic
{
    /// stateful constructor
    ///
    __device__ __forceinline__
    warp_static_atomic(uint32* pool)
        : m_dest(pool) {}

    /// add zero or exactly N per thread to a shared value without waiting for the result: useful to alloc N entries from a common pool
    ///
    /// \param n                number of elements to alloc
    template <uint32 N>
    __device__ __forceinline__
    void add(bool p)
    {
        warp_static_atomic::static_add<N>( p, m_dest );
    }

    /// subtract zero or exactly N per thread to a shared value without waiting for the result: useful to alloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    template <uint32 N>
    __device__ __forceinline__
    void sub(bool p)
    {
        warp_static_atomic::static_sub<N>( p, m_dest );
    }

    /// add zero or exactly N per thread to a shared value: useful to alloc N entries from a common pool
    ///
    /// \param n                number of elements to alloc
    template <uint32 N>
    __device__ __forceinline__
    void add(bool p, uint32* result)
    {
        warp_static_atomic::static_add<N>( p, m_dest, result );
    }

    /// subtract zero or exactly N per thread to a shared value: useful to alloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    template <uint32 N>
    __device__ __forceinline__
    void sub(bool p, uint32* result)
    {
        warp_static_atomic::static_sub<N>( p, m_dest, result );
    }

    // --- stateless methods --- //

    /// add zero or exactly N per thread to a shared value without waiting for the result: useful to alloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    template <uint32 N>
    __device__ __forceinline__
    static void static_add(bool p, uint32* dest)
    {
        const uint32 warp_tid   = threadIdx.x & 31;
        const uint32 warp_mask  = __ballot( p );
        const uint32 warp_count = __popc( warp_mask );
        const uint32 warp_scan  = __popc( warp_mask << (warpSize - warp_tid) );

        // perform the atomic
        if (warp_scan == 0 && p)
            atomicAdd( dest, warp_count * N );
    }

    /// subtract zero or exactly N per thread to a shared value without waiting for the result: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    template <uint32 N>
    __device__ __forceinline__
    static void static_sub(bool p, uint32* dest)
    {
        const uint32 warp_tid   = threadIdx.x & 31;
        const uint32 warp_mask  = __ballot( p );
        const uint32 warp_count = __popc( warp_mask );
        const uint32 warp_scan  = __popc( warp_mask << (warpSize - warp_tid) );

        // perform the atomic
        if (warp_scan == 0 && p)
            atomicSub( dest, warp_count * N );
    }

    /// add zero or exactly N per thread to a shared value: useful to alloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    static void static_add(bool p, uint32* dest, uint32* result)
    {
        const uint32 warp_tid   = threadIdx.x & 31;
        const uint32 warp_mask  = __ballot( p );
        const uint32 warp_count = __popc( warp_mask );
        const uint32 warp_scan  = __popc( warp_mask << (warpSize - warp_tid) );

		const uint32 first_tid = ffs(warp_mask) - 1;

		uint32 broadcast_offset;

        // acquire an offset for this warp
		if (warp_tid == first_tid)
            broadcast_offset = atomicAdd( dest, warp_count * N );

		// obtain the offset from the first participating thread
		const uint32 offset = cub::ShuffleIndex(broadcast_offset, first_tid);

        // compute the per-thread offset
        *result = offset + warp_scan * N;
    }

    /// subtract zero or exactly N per thread to a shared value: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    static void static_sub(bool p, uint32* dest, uint32* result)
    {
        const uint32 warp_tid   = threadIdx.x & 31;
        const uint32 warp_mask  = __ballot( p );
        const uint32 warp_count = __popc( warp_mask );
        const uint32 warp_scan  = __popc( warp_mask << (warpSize - warp_tid) );

		const uint32 first_tid = ffs(warp_mask) - 1;

		uint32 broadcast_offset;

        // acquire an offset for this warp
		if (warp_tid == first_tid)
            broadcast_offset = atomicSub( dest, warp_count * N );

		// obtain the offset from the first participating thread
		const uint32 offset = cub::ShuffleIndex(broadcast_offset, first_tid);

        // compute the per-thread offset
        *result = offset - warp_scan * N;
    }

private:
    uint32* m_dest;
};

/// An efficient warp-synchronous atomic adder, to add or subtract to a shared integer.
///
/// Given a pointer to an integer (e.g. representing a "pool"), this class allows the threads
/// in a warp to add (allocate) or subtract (deallocate) a per-thread integer to it.
///
struct warp_atomic
{
    struct temp_storage_type
    {
        union {
            typename cub::WarpScan<uint32>::TempStorage     scan_storage;
            typename cub::WarpReduce<uint32>::TempStorage   reduce_storage;
        };
    };

    /// stateful object constructor
    ///
    __device__ __forceinline__
    warp_atomic(uint32* dest, temp_storage_type& temp_storage)
        : m_dest(dest), m_temp_storage(temp_storage) {}


    /// add a per-thread value to the shared integer without waiting for the result
    ///
    /// \param n                number of elements to alloc
    __device__ __forceinline__
    void add(uint32 n)
    {
        warp_atomic::add( n, m_dest, m_temp_storage );
    }

    /// add a per-thread value to the shared integer without waiting for the result
    ///
    /// \param n                number of elements to alloc
    __device__ __forceinline__
    void sub(uint32 n)
    {
        warp_atomic::sub( n, m_dest, m_temp_storage );
    }

    /// add zero or exactly N per thread to a shared value without waiting for the result: useful to alloc N entries from a common pool
    /// NOTE that this class internally uses a synchronous warp reduction, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param p                allocation predicate
    template <uint32 N>
    __device__ __forceinline__
    void add(bool p)
    {
        return warp_atomic::static_add<N>( p, m_dest );
    }

    /// subtract zero or exactly N per thread to a shared value without waiting for the result: useful to dealloc N entries from a common pool
    /// NOTE that this class internally uses a synchronous warp reduction, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param p                allocation predicate
    template <uint32 N>
    __device__ __forceinline__
    void sub(bool p)
    {
        warp_atomic::static_sub<N>( p, m_dest );
    }

    /// add a per-thread value to the shared integer: useful to alloc entries from a common pool
    /// NOTE that this class internally uses a synchronous warp scan, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param result           output result
    __device__ __forceinline__
    void add(uint32 n, uint32* result)
    {
        warp_atomic::add( n, m_dest, result, m_temp_storage );
    }

    /// subtract a per-thread value to the shared integer: useful to dealloc entries from a common pool
    /// NOTE that this class internally uses a synchronous warp scan, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param result           output result
    __device__ __forceinline__
    void sub(uint32 n, uint32* result)
    {
        warp_atomic::sub( n, m_dest, result, m_temp_storage );
    }

    /// add zero or exactly N per thread to a shared value: useful to alloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    void add(bool p, uint32* result)
    {
        return warp_atomic::static_add<N>( p, m_dest, result );
    }

    /// subtract zero or exactly N per thread to a shared value: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    void sub(bool p, uint32* result)
    {
        return warp_atomic::static_sub<N>( p, m_dest, result );
    }

    // --- stateless methods --- //

    /// add a per-thread value to the shared integer without waiting for the result
    /// NOTE that this class internally uses a synchronous warp reduction, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param dest             the destination of the atomic
    __device__ __forceinline__
    static void add(uint32 n, uint32* dest, temp_storage_type& temp_storage)
    {
        // issue a warp-reduction
        const uint32 warp_count = cub::WarpReduce<uint32>(temp_storage.reduce_storage).Sum(n);

        // issue a per-warp atomic
        const uint32 warp_tid = threadIdx.x & 31;
        if (warp_tid == 0)
            atomicAdd( dest, warp_count );
    }

    /// subtract a per-thread value to the shared integer without waiting for the result
    /// NOTE that this class internally uses a synchronous warp reduction, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param dest             the destination of the atomic
    __device__ __forceinline__
    static void sub(uint32 n, uint32* dest, temp_storage_type& temp_storage)
    {
        // issue a warp-reduction
        const uint32 warp_count = cub::WarpReduce<uint32>(temp_storage.reduce_storage).Sum(n);

        // issue a per-warp atomic
        const uint32 warp_tid = threadIdx.x & 31;
        if (warp_tid == 0)
            atomicSub( dest, warp_count );
    }

    /// add a per-thread value to the shared integer: useful to alloc entries from a common pool
    /// NOTE that this class internally uses a synchronous warp scan, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param dest             the destination of the atomic
    /// \param result           output result
    __device__ __forceinline__
    static void add(uint32 n, uint32* dest, uint32* result, temp_storage_type& temp_storage)
    {
        uint32 warp_scan, warp_count;

        // issue a warp-scan
        cub::WarpScan<uint32>(temp_storage.scan_storage).ExclusiveSum(n, warp_scan, warp_count);

        const uint32 warp_tid = threadIdx.x & 31;

        // issue a per-warp atomic
        uint32 base_index;
        if (warp_tid == 0)
            base_index = atomicAdd( dest, warp_count );

        // compute the per-thread offset
        *result = cub::ShuffleIndex( base_index, 0 ) + warp_scan;
    }

    /// subtract a per-thread value to the shared integer: useful to dealloc entries from a common pool
    /// NOTE that this class internally uses a synchronous warp scan, and as such it requires all
    /// threads to participate in the operation.
    ///
    /// \param n                number of elements to alloc
    /// \param dest             the destination of the atomic
    /// \param result           output result
    __device__ __forceinline__
    static void sub(uint32 n, uint32* dest, uint32* result, temp_storage_type& temp_storage)
    {
        uint32 warp_scan, warp_count;

        // issue a warp-scan
        cub::WarpScan<uint32>(temp_storage.scan_storage).ExclusiveSum(n, warp_scan, warp_count);

        const uint32 warp_tid = threadIdx.x & 31;

        // issue a per-warp atomic
        uint32 base_index;
        if (warp_tid == 0)
            base_index = atomicSub( dest, warp_count );

        // compute the per-thread offset
        *result = cub::ShuffleIndex( base_index, 0 ) - warp_scan;
    }

    /// add zero or exactly N per thread to a shared value without waiting for the result: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    template <uint32 N>
    __device__ __forceinline__
    static void static_add(bool p, uint32* dest)
    {
        return warp_static_atomic::static_add<N>( p, dest );
    }

    /// subtract zero or exactly N per thread to a shared value without waiting for the result: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    template <uint32 N>
    __device__ __forceinline__
    static void static_sub(bool p, uint32* dest)
    {
        warp_static_atomic::static_sub<N>( p, dest );
    }

    /// add zero or exactly N per thread to a shared value: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    static void static_add(bool p, uint32* dest, uint32* result)
    {
        return warp_static_atomic::static_add<N>( p, dest, result );
    }

    /// subtract zero or exactly N per thread to a shared value: useful to dealloc N entries from a common pool
    ///
    /// \param p                allocation predicate
    /// \param dest             the destination of the atomic
    /// \param result           output result
    template <uint32 N>
    __device__ __forceinline__
    static void static_sub(bool p, uint32* dest, uint32* result)
    {
        warp_static_atomic::static_sub<N>( p, dest, result );
    }

private:
    uint32*             m_dest;
    temp_storage_type&  m_temp_storage;
};

///@} CUDAAtomicsModule
///@} CUDAModule
///@} Basic

} // namespace cuda
} // namespace cugar
