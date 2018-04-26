/*
 * cugar
 * Copyright (c) 2011-2014, NVIDIA CORPORATION. All rights reserved.
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

/*! \file vector.h
 *   \brief Define host / device vectors
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/threads.h>
#include <cugar/basic/cuda/arch.h>
#include <cub/util_allocator.cuh>
#include <thrust/device_ptr.h>

namespace cugar {

///@addtogroup Basic
///@{

///@addtogroup CUDAModule
///@{

///@defgroup CUDAAllocatorsModules CUDA Allocators
/// This module defines custom CUDA allocators
///@{

/// Implements a caching device allocator based on CUB's
///
struct byte_caching_device_allocator
{
  public:
    typedef char              value_type;
    typedef char*             pointer;
    typedef const char*       const_pointer;
    typedef char&             reference;
    typedef const char&       const_reference;
    typedef size_t            size_type;
    typedef int64             difference_type;

    /// allocate a new chunk
    ///
    CUGAR_HOST_DEVICE
    char* allocate(size_type num_bytes)
    {
      #if defined(CUGAR_DEVICE_COMPILATION)
        return NULL;
      #else
        if (s_caching_device_allocator == NULL)
            init();

        void* ptr;
        cuda::check_error( s_caching_device_allocator->DeviceAllocate( &ptr, num_bytes ), "cugar::caching_device_allocator::allocate()" );
        return (char*)ptr;
      #endif
    }

    /// deallocate a previously allocated chunk
    ///
    CUGAR_HOST_DEVICE
    void deallocate(char *ptr, size_type n)
    {
      #if !defined(CUGAR_DEVICE_COMPILATION)
        cuda::check_error( s_caching_device_allocator->DeviceFree( ptr ), "cugar::caching_device_allocator::deallocate()" );
      #endif
    }

    /// initialize the global pooled allocator
    ///
    static void init(const size_t max_cached_bytes = 128u*1024u*1024u)
    {
        ScopedLock scoped_lock( &s_mutex );
        if (s_caching_device_allocator == NULL)
            s_caching_device_allocator = new cub::CachingDeviceAllocator( 8, 2u, 10u, max_cached_bytes );
    }

private:
	CUGAR_API static cub::CachingDeviceAllocator* volatile s_caching_device_allocator;
    static Mutex                                 s_mutex;
};

/// Implements a caching device allocator based on CUB's
///
template <typename T>
struct caching_device_allocator
{
    /*! Type of element allocated, \c T. */
    typedef T                                       value_type;

    /*! Pointer to allocation, \c device_ptr<T>. */
    typedef thrust::device_ptr<T>                   pointer;

    /*! \c const pointer to allocation, \c device_ptr<const T>. */
    typedef thrust::device_ptr<const T>             const_pointer;

    /*! Reference to allocated element, \c device_reference<T>. */
    typedef thrust::device_reference<T>             reference;

    /*! \c const reference to allocated element, \c device_reference<const T>. */
    typedef thrust::device_reference<const T>       const_reference;

    /*! Type of allocation size, \c std::size_t. */
    typedef std::size_t                             size_type;

    /*! Type of allocation difference, \c pointer::difference_type. */
    typedef typename pointer::difference_type       difference_type;

    /*! Returns the address of an allocated object.
     *  \return <tt>&r</tt>.
     */
    __host__ __device__
    inline pointer address(reference r) { return &r; }
    
    /*! Returns the address an allocated object.
     *  \return <tt>&r</tt>.
     */
    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    /*! Allocates storage for \p cnt objects.
     *  \param cnt The number of objects to allocate.
     *  \return A \p pointer to uninitialized storage for \p cnt objects.
     *  \note Memory allocated by this function must be deallocated with \p deallocate.
     */
    CUGAR_HOST
    inline pointer allocate(size_type cnt,
                            const_pointer = const_pointer(static_cast<T*>(0)))
    {
        return pointer( (T*)base_allocator.allocate( cnt * sizeof(T) ) );
    } // end allocate()

    /*! Deallocates storage for objects allocated with \p allocate.
     *  \param p A \p pointer to the storage to deallocate.
     *  \param cnt The size of the previous allocation.
     *  \note Memory deallocated by this function must previously have been
     *        allocated with \p allocate.
     */
    CUGAR_HOST
    inline void deallocate(pointer p, size_type cnt)
    {
        base_allocator.deallocate((char*)p.get(), cnt * sizeof(T));
    } // end deallocate()

private:
    byte_caching_device_allocator base_allocator;
};

///@} CUDAAllocatorsModule
///@} CUDAModule
///@} Basic

} // namespace cugar
