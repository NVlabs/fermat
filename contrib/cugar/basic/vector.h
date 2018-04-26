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
#include <cugar/basic/iterator.h>
#include <cugar/basic/thrust_view.h>
#include <cugar/basic/vector_view.h>
#include <cugar/basic/cuda/allocator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace cugar {
namespace cuda {

	/// \page vectors_page Vectors
	///
	/// This module implements host & device vectors
	///
	/// - vector
	/// - host_vector
	/// - device_vector
	/// - caching_device_vector
	///
	/// \section VectorsExampleSection Example
	///
	///\code
	/// // build a host vector and copy it to the device
	/// cugar::vector<host_tag,uint32> h_vector;
	/// cugar::vector<host_tag,uint32> d_vector;
	///
	/// h_vector.push_back(3u);
	/// d_vector = h_vector;
	///\endcode
	///

///@addtogroup Basic
///@{

// utility function to copy a thrust device vector to a thrust host vector
// the sole reason for this is to eliminate warnings from thrust when using the assignment operator
template<typename TTargetVector, typename TSourceVector>
static CUGAR_FORCEINLINE void thrust_copy_vector(TTargetVector& target, TSourceVector& source)
{
    if (target.size() != source.size())
    {
        target.clear();
        target.resize(source.size());
    }

    thrust::copy(source.begin(), source.end(), target.begin());
}

template<typename TTargetVector, typename TSourceVector>
static CUGAR_FORCEINLINE void thrust_copy_vector(TTargetVector& target, TSourceVector& source, uint32 count)
{
    if (target.size() != count)
    {
        target.clear();
        target.resize(count);
    }

    thrust::copy(source.begin(), source.begin() + count, target.begin());
}

} // namespace cuda

/// a dynamic host/device vector class
///
template <typename system_tag, typename T>
struct default_vector_allocator {};

template <typename T>
struct default_vector_allocator<host_tag,T>
{
    typedef std::allocator<T>  type;
};

template <typename T>
struct default_vector_allocator<device_tag,T>
{
    typedef thrust::device_malloc_allocator<T>  type;
};

/// a dynamic host/device vector class
///
template <typename system_tag, typename T, typename Alloc = typename default_vector_allocator<system_tag,T>::type>
struct vector {};

/// a dynamic host vector class
///
template <typename T, typename Alloc>
struct vector<host_tag,T,Alloc> : public thrust::host_vector<T,Alloc>
{
    typedef host_tag                            system_tag;

    typedef thrust::host_vector<T,Alloc>        base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    typedef cugar::vector_view<T*,uint64>              plain_view_type;
    typedef cugar::vector_view<const T*,uint64>  const_plain_view_type;

    /// constructor
    ///
    vector(const size_t size = 0, const T val = T()) : base_type( size, val ) {}

    template <typename OtherAlloc>
    vector(const thrust::host_vector<T,OtherAlloc>&   v) : base_type( v ) {}

    template <typename OtherAlloc>
    vector(const thrust::device_vector<T,OtherAlloc>& v) : base_type( v ) {}

    template <typename OtherAlloc>
    vector<host_tag,T,Alloc>& operator= (const thrust::host_vector<T,OtherAlloc>& v)   { cuda::thrust_copy_vector( *this, v ); return *this; }

    template <typename OtherAlloc>
    vector<host_tag,T,Alloc>& operator= (const thrust::device_vector<T,OtherAlloc>& v) { cuda::thrust_copy_vector( *this, v ); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// resize if larger
    ///
    void expand(const size_t sz)
    {
        if (size() < sz)
            resize( sz );
    }

	T&		 last()		  { return (*this)[size() - 1]; }
	const T& last() const { return (*this)[size() - 1]; }
};

/// a dynamic device vector class
///
template <typename T, typename Alloc>
struct vector<device_tag,T,Alloc> : public thrust::device_vector<T,Alloc>
{
    typedef device_tag                          system_tag;

    typedef thrust::device_vector<T,Alloc>      base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    typedef cugar::vector_view<T*,uint64>              plain_view_type;
    typedef cugar::vector_view<const T*,uint64>  const_plain_view_type;

    /// constructor
    ///
    vector(const size_t size = 0, const T val = T()) : base_type( size, val ) {}

    template <typename OtherAlloc>
    vector(const thrust::host_vector<T,OtherAlloc>&   v) : base_type( v ) {}

    template <typename OtherAlloc>
    vector(const thrust::device_vector<T,OtherAlloc>& v) : base_type( v ) {}

    template <typename OtherAlloc>
    vector<device_tag,T,Alloc>& operator= (const thrust::host_vector<T,OtherAlloc>& v)   { cuda::thrust_copy_vector( *this, v ); return *this; }

    template <typename OtherAlloc>
    vector<device_tag,T,Alloc>& operator= (const thrust::device_vector<T,OtherAlloc>& v) { cuda::thrust_copy_vector( *this, v ); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// resize if larger
    ///
    void expand(const size_t sz)
    {
        if (size() < sz)
            resize( sz );
    }

	T&		 last()		  { return (*this)[size() - 1]; }
	const T& last() const { return (*this)[size() - 1]; }
};

/// a dynamic host vector class (with C++11 it would be:  template <T> typedef vector<host_tag,T> host_vector;)
///
template <typename T>
struct host_vector : public vector<host_tag,T>
{
    typedef host_tag                                            system_tag;

    typedef vector<host_tag,T>                                  base_type;
    typedef typename base_type::const_iterator                  const_iterator;
    typedef typename base_type::iterator                        iterator;
    typedef typename base_type::value_type                      value_type;

    typedef typename base_type::plain_view_type                 plain_view_type;
    typedef typename base_type::const_plain_view_type     const_plain_view_type;

    /// constructor
    ///
    host_vector(const size_t size = 0, const T val = T()) : base_type( size, val ) {}

    template <typename OtherAlloc>
    host_vector(const thrust::host_vector<T,OtherAlloc>&   v) : base_type( v ) {}

    template <typename OtherAlloc>
    host_vector(const thrust::device_vector<T,OtherAlloc>& v) : base_type( v ) {}

    template <typename OtherAlloc>
    host_vector<T>& operator= (const thrust::host_vector<T,OtherAlloc>& v)   { cuda::thrust_copy_vector( *this, v ); return *this; }

    template <typename OtherAlloc>
    host_vector<T>& operator= (const thrust::device_vector<T,OtherAlloc>& v) { cuda::thrust_copy_vector( *this, v ); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }
};

/// a dynamic device vector class
///
template <typename T>
struct device_vector : public vector<device_tag,T>
{
    typedef device_tag                                          system_tag;

    typedef vector<device_tag,T>                                base_type;
    typedef typename base_type::const_iterator                  const_iterator;
    typedef typename base_type::iterator                        iterator;
    typedef typename base_type::value_type                      value_type;

    typedef typename base_type::plain_view_type                 plain_view_type;
    typedef typename base_type::const_plain_view_type     const_plain_view_type;

    /// constructor
    ///
    device_vector(const size_t size = 0, const T val = T()) : base_type( size, val ) {}

    template <typename OtherAlloc>
    device_vector(const thrust::host_vector<T,OtherAlloc>&   v) : base_type( v ) {}

    template <typename OtherAlloc>
    device_vector(const thrust::device_vector<T,OtherAlloc>& v) : base_type( v ) {}

    template <typename OtherAlloc>
    device_vector<T>& operator= (const thrust::host_vector<T,OtherAlloc>& v)   { cuda::thrust_copy_vector( *this, v ); return *this; }

    template <typename OtherAlloc>
    device_vector<T>& operator= (const thrust::device_vector<T,OtherAlloc>& v) { cuda::thrust_copy_vector( *this, v ); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }
};

/// a dynamic device vector class with a caching allocator
///
template <typename T>
struct caching_device_vector : public vector<device_tag,T,caching_device_allocator<T> >
{
    typedef device_tag                                          system_tag;

    typedef vector<device_tag,T,caching_device_allocator<T> >   base_type;
    typedef typename base_type::const_iterator                  const_iterator;
    typedef typename base_type::iterator                        iterator;
    typedef typename base_type::value_type                      value_type;

    typedef typename base_type::plain_view_type                 plain_view_type;
    typedef typename base_type::const_plain_view_type     const_plain_view_type;

    /// constructor
    ///
    caching_device_vector(const size_t size = 0, const T val = T()) : base_type( size, val ) {}

    template <typename OtherAlloc>
    caching_device_vector(const thrust::host_vector<T,OtherAlloc>&   v) : base_type( v ) {}

    template <typename OtherAlloc>
    caching_device_vector(const thrust::device_vector<T,OtherAlloc>& v) : base_type( v ) {}

    template <typename OtherAlloc>
    caching_device_vector<T>& operator= (const thrust::host_vector<T,OtherAlloc>& v)   { cuda::thrust_copy_vector( *this, v ); return *this; }

    template <typename OtherAlloc>
    caching_device_vector<T>& operator= (const thrust::device_vector<T,OtherAlloc>& v) { cuda::thrust_copy_vector( *this, v ); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), cugar::raw_pointer( *this ) ); }
};

/// a utility meta-type to wrap naked device pointers as thrust::device_ptr
///
template <typename T>   struct device_iterator_type             { typedef T type; };
template <typename T>   struct device_iterator_type<T*>         { typedef thrust::device_ptr<T> type; };
template <typename T>   struct device_iterator_type<const T*>   { typedef thrust::device_ptr<const T> type; };

/// a convenience function to wrap naked device pointers as thrust::device_ptr
///
template <typename T>
typename device_iterator_type<T>::type device_iterator(const T it)
{
    // wrap the plain iterator
    return typename device_iterator_type<T>::type( it );
}

///@} Basic

} // namespace cugar
