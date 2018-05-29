/*
 * CUGAR : Cuda Graphics Accelerator
 *
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
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

#if defined(__CUDACC__)
#include <cub/cub.cuh>
#endif

#include <cugar/basic/types.h>
#include <cugar/basic/iterator.h>

namespace cugar {
namespace cuda {

///@addtogroup Basic
///@{

///@addtogroup CUDAModule
///@{

/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier
{
    LOAD_DEFAULT,       ///< Default (no modifier)
    LOAD_CA,            ///< Cache at all levels
    LOAD_CG,            ///< Cache at global level
    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
    LOAD_CV,            ///< Cache as volatile (including cached system lines)
    LOAD_LDG,           ///< Cache as texture
    LOAD_VOLATILE,      ///< Volatile (any memory space)
};

/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheStoreModifier
{
    STORE_DEFAULT,              ///< Default (no modifier)
    STORE_WB,                   ///< Cache write-back all coherent levels
    STORE_CG,                   ///< Cache at global level
    STORE_CS,                   ///< Cache streaming (likely to be accessed once)
    STORE_WT,                   ///< Cache write-through (to system memory)
    STORE_VOLATILE,             ///< Volatile shared (any memory space)
};

#if defined(__CUDACC__)
template <CacheLoadModifier MOD> struct cub_load_mod {};

template <> struct cub_load_mod<LOAD_DEFAULT>   { static const cub::CacheLoadModifier MOD = cub::LOAD_DEFAULT; };
template <> struct cub_load_mod<LOAD_CA>        { static const cub::CacheLoadModifier MOD = cub::LOAD_CA;      };
template <> struct cub_load_mod<LOAD_CG>        { static const cub::CacheLoadModifier MOD = cub::LOAD_CG;      };
template <> struct cub_load_mod<LOAD_CS>        { static const cub::CacheLoadModifier MOD = cub::LOAD_CS;      };
template <> struct cub_load_mod<LOAD_CV>        { static const cub::CacheLoadModifier MOD = cub::LOAD_CV;      };
template <> struct cub_load_mod<LOAD_LDG>       { static const cub::CacheLoadModifier MOD = cub::LOAD_LDG;     };
template <> struct cub_load_mod<LOAD_VOLATILE>  { static const cub::CacheLoadModifier MOD = cub::LOAD_VOLATILE;};

template <CacheStoreModifier MOD> struct cub_store_mod {};

template <> struct cub_store_mod<STORE_DEFAULT>     { static const cub::CacheStoreModifier MOD = cub::STORE_DEFAULT; };
template <> struct cub_store_mod<STORE_WB>          { static const cub::CacheStoreModifier MOD = cub::STORE_WB;      };
template <> struct cub_store_mod<STORE_CG>          { static const cub::CacheStoreModifier MOD = cub::STORE_CG;      };
template <> struct cub_store_mod<STORE_CS>          { static const cub::CacheStoreModifier MOD = cub::STORE_CS;      };
template <> struct cub_store_mod<STORE_WT>          { static const cub::CacheStoreModifier MOD = cub::STORE_WT;      };
template <> struct cub_store_mod<STORE_VOLATILE>    { static const cub::CacheStoreModifier MOD = cub::STORE_VOLATILE;};
#endif

/// issue a load
///
template <CacheLoadModifier LOAD_MOD, typename T>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T load(const T* ptr)
{
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return cub::ThreadLoad<cub_load_mod<LOAD_MOD>::MOD>( const_cast<T*>(ptr) );
  #else
    return *ptr;
  #endif
}

/// issue a store
///
template <CacheStoreModifier STORE_MOD, typename T>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
void store(T* ptr, const T& value)
{
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    cub::ThreadStore<cub_store_mod<STORE_MOD>::MOD>( ptr, value );
  #else
    *ptr = value;
  #endif
}

///
/// Wrapper class to create a cub::ThreadLoad iterator out of a raw pointer
///
template <typename T, CacheLoadModifier MOD>
struct load_pointer
{
    typedef T                                                           value_type;
    typedef value_type                                                  reference;
    typedef value_type                                                  const_reference;
    typedef value_type*                                                 pointer;
    typedef typename std::iterator_traits<const T*>::difference_type    difference_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer() {}

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer(const T* base) : m_base( base ) {}

    /// copy constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer(const load_pointer& it) : m_base( it.m_base ) {}

    /// const indexing operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    value_type operator[](const uint32 i) const
    {
        return load<MOD>( m_base + i );
    }

    /// dereference operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    value_type operator*() const
    {
        return load<MOD>( m_base );
    }

    /// pre-increment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD> operator++(int i)
    {
        load_pointer<T,MOD> r( m_base );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD> operator--(int i)
    {
        load_pointer<T,MOD> r( m_base );
        --m_base;
        return r;
    }

    /// addition
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD> operator+(const difference_type i) const
    {
        return load_pointer( m_base + i );
    }

    /// subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD> operator-(const difference_type i) const
    {
        return load_pointer( m_base - i );
    }

    /// addition
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer<T,MOD>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    difference_type operator-(const load_pointer<T,MOD> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    load_pointer& operator=(const load_pointer<T,MOD>& it)
    {
        m_base = it.m_base;
        return *this;
    }

    const T* m_base;
};

/// make a load_pointer
///
template <CacheLoadModifier MOD, typename T>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
load_pointer<T,MOD> make_load_pointer(const T* it)
{
    return load_pointer<T,MOD>( it );
}

///
/// Wrapper class to create a cub::ThreadStore reference out of a raw pointer
///
template <typename T, CacheStoreModifier STORE_MOD, CacheLoadModifier LOAD_MOD = LOAD_DEFAULT>
struct store_reference
{
    typedef T           value_type;

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference() {}

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference(T* base) : m_base( base ) {}

    /// copy constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference(const store_reference& it) : m_base( it.m_base ) {}

    /// assignment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference& operator=(const value_type value)
    {
        store<STORE_MOD>( m_base, value );
        return *this;
    }

    /// conversion to value_type
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    operator value_type()
    {
        return load<LOAD_MOD>( m_base );
    }

    /// += operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference& operator+=(const value_type value)
    {
        const value_type old = load<LOAD_MOD>( m_base );
        store<STORE_MOD>( m_base, old + value );
        return *this;
    }

    /// -= operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_reference& operator-=(const value_type value)
    {
        const value_type old = load<LOAD_MOD>( m_base );
        store<STORE_MOD>( m_base, old - value );
        return *this;
    }

    T* m_base;
};

///
/// Wrapper class to create a cub::ThreadStore iterator out of a raw pointer
///
template <typename T, CacheStoreModifier STORE_MOD, CacheLoadModifier LOAD_MOD = LOAD_DEFAULT>
struct store_pointer
{
    typedef T                                                           value_type;
    typedef store_reference<value_type,STORE_MOD,LOAD_MOD>              reference;
    typedef value_type                                                  const_reference;
    typedef value_type*                                                 pointer;
    typedef typename std::iterator_traits<T*>::difference_type          difference_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer() {}

    /// constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer(T* base) : m_base( base ) {}

    /// copy constructor
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer(const store_pointer& it) : m_base( it.m_base ) {}

    /// const indexing operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    reference operator[](const uint32 i)
    {
        return reference( m_base + i );
    }

    /// dereference operator
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    reference operator*()
    {
        return reference( m_base );
    }

    /// pre-increment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD> operator++(int i)
    {
        store_pointer<T,STORE_MOD,LOAD_MOD> r( m_base );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD> operator--(int i)
    {
        store_pointer<T,STORE_MOD,LOAD_MOD> r( m_base );
        --m_base;
        return r;
    }

    /// addition
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD> operator+(const difference_type i) const
    {
        return store_pointer( m_base + i );
    }

    /// subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD> operator-(const difference_type i) const
    {
        return store_pointer( m_base - i );
    }

    /// addition
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer<T,STORE_MOD,LOAD_MOD>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    difference_type operator-(const store_pointer<T,STORE_MOD,LOAD_MOD> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    store_pointer& operator=(const store_pointer<T,STORE_MOD,LOAD_MOD>& it)
    {
        m_base = it.m_base;
        return *this;
    }

    T* m_base;
};

/// make a store_pointer
///
template <CacheStoreModifier STORE_MOD, CacheLoadModifier LOAD_MOD, typename T>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
store_pointer<T,STORE_MOD,LOAD_MOD> make_store_pointer(const T* it)
{
    return store_pointer<T,STORE_MOD,LOAD_MOD>( it );
}

///@} CUDAModule
///@} Basic

} // namespace cuda
} // namespace cugar
