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

#pragma once

#include <cugar/basic/types.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#ifndef WIN32
#ifdef __INTEL_COMPILER
#include <ia32intrin.h> // ia32intrin.h
#else
//#warning atomics.h BROKEN on GCC!
// Intrinsics docs at http://gcc.gnu.org/onlinedocs/gcc-4.3.2/gcc/Atomic-Builtins.html
#endif
#endif

namespace cugar {

/// \page atomics_page Atomics
///\par
/// This \ref Atomics "module" implements basic host/device atomics.
///
/// - float  atomic_add(float* value, const float op)
/// - int32  atomic_add(int32* value, const int32 op)
/// - uint32 atomic_add(uint32* value, const uint32 op)
/// - uint64 atomic_add(uint64* value, const uint64 op)
/// - int32  atomic_sub(int32* value, const int32 op)
/// - uint32 atomic_sub(uint32* value, const uint32 op)
/// - int32  atomic_increment(int32* value)
/// - int64  atomic_increment(int64* value)
/// - int32  atomic_decrement(int32* value)
/// - int64  atomic_decrement(int64* value)
/// - uint32 atomic_or(uint32* value, const uint32 op)
/// - uint64 atomic_or(uint64* value, const uint64 op)
///

///@addtogroup Basic
///@{

///@defgroup Atomics	Atomics Module
/// This module implements basic host/device atomic counters.
///@{

CUGAR_API void host_release_fence();
CUGAR_API void host_acquire_fence();

CUGAR_API int32  host_atomic_add( int32* value, const  int32 op);
CUGAR_API uint32 host_atomic_add(uint32* value, const uint32 op);
CUGAR_API int64  host_atomic_add( int64* value, const  int64 op);
CUGAR_API uint64 host_atomic_add(uint64* value, const uint64 op);

CUGAR_API int32  host_atomic_sub( int32* value, const  int32 op);
CUGAR_API uint32 host_atomic_sub(uint32* value, const uint32 op);
CUGAR_API int64  host_atomic_sub( int64* value, const  int64 op);
CUGAR_API uint64 host_atomic_sub(uint64* value, const uint64 op);

CUGAR_API uint32 host_atomic_or(uint32* value, const uint32 op);
CUGAR_API uint64 host_atomic_or(uint64* value, const uint64 op);

CUGAR_API int32 host_atomic_increment(int32* value);
CUGAR_API int64 host_atomic_increment(int64* value);

CUGAR_API int32 host_atomic_decrement(int32* value);
CUGAR_API int64 host_atomic_decrement(int64* value);

CUGAR_API float host_atomic_add(float* value, const float op);


/// atomic add: returns the value BEFORE the addition
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float atomic_add(float* value, const float op)
{
#if defined(CUGAR_DEVICE_COMPILATION)
	return atomicAdd(value, op);
#else
	return host_atomic_add(value, op);
#endif
}

/// atomic add: returns the value BEFORE the addition
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int32 atomic_add(int32* value, const int32 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicAdd( value, op );
  #else
    return host_atomic_add( value, op );
  #endif
}

/// atomic add: returns the value BEFORE the addition
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 atomic_add(uint32* value, const uint32 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicAdd( value, op );
  #else
    return host_atomic_add( value, op );
  #endif
}

/// atomic add: returns the value BEFORE the addition
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint64 atomic_add(uint64* value, const uint64 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicAdd( (unsigned long long int*)value, (unsigned long long int)op );
  #else
    return host_atomic_add( value, op );
  #endif
}

/// atomic subtract: returns the value BEFORE the suctraction
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int32 atomic_sub(int32* value, const int32 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicSub( value, op );
  #else
    return host_atomic_sub( value, op );
  #endif
}

/// atomic subtract: returns the value BEFORE the suctraction
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 atomic_sub(uint32* value, const uint32 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicSub( value, op );
  #else
    return host_atomic_sub( value, op );
  #endif
}

/// atomic OR: returns the value BEFORE the or
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 atomic_or(uint32* value, const uint32 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicOr( value, op );
  #else
    return host_atomic_or( value, op );
  #endif
}

/// atomic OR: returns the value BEFORE the or
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint64 atomic_or(uint64* value, const uint64 op)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicOr( (unsigned long long int*)value, (unsigned long long int)op );
  #else
    return host_atomic_or( value, op );
  #endif
}

/// atomic increment: returns the value BEFORE the increment
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int32 atomic_increment(int32* value)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicAdd( value, int32(1) );
  #else
    return host_atomic_increment( value );
  #endif
}
/// atomic increment: returns the value BEFORE the increment
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int64 atomic_increment(int64* value)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return (int64)atomicAdd( (unsigned long long int*)value, (unsigned long long int)(1) );
  #else
    return host_atomic_increment( value );
  #endif
}

/// atomic decrement: returns the value BEFORE the decrement
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int32 atomic_decrement(int32* value)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return atomicSub( value, int32(1) );
  #else
    return host_atomic_decrement( value );
  #endif
}
/// atomic decrement: returns the value BEFORE the decrement
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
int64 atomic_decrement(int64* value)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
    return (int64)atomicAdd( (unsigned long long int*)value, (unsigned long long int)(-1) );
  #else
    return host_atomic_decrement( value );
  #endif
}

/// an atomic integer class
///
template<typename intT>
struct AtomicInt
{
    /// constructor
    AtomicInt() : m_value(0) {}

    /// destructor
    AtomicInt(const intT value) : m_value(value) {}

    /// increment by one
    intT increment()
    {
        return atomic_increment( (intT*)&m_value );
    }

    /// decrement by one
    intT decrement()
    {
        return atomic_decrement( (intT*)&m_value );
    }

    /// increment by one
    intT operator++(int) { return increment(); }

    /// decrement by one
    intT operator--(int) { return decrement(); }

    /// increment by one
    intT operator++() { return increment()+intT(1); }

    /// decrement by one
    intT operator--() { return decrement()-intT(1); }

    /// increment by v
    intT operator+=(const intT v) { return atomic_add( (intT*)&m_value, v ) + v; }
    /// decrement by v
    intT operator-=(const intT v) { return atomic_sub( (intT*)&m_value, v ) - v; }

    /// compare
    bool operator==(const intT value) { return m_value == value; }
    bool operator!=(const intT value) { return m_value != value; }
    bool operator>=(const intT value) { return m_value >= value; }
    bool operator<=(const intT value) { return m_value <= value; }
    bool operator>(const intT value)  { return m_value >  value; }
    bool operator<(const intT value)  { return m_value <  value; }

    volatile intT m_value;
};

typedef AtomicInt<int>     AtomicInt32;
typedef AtomicInt<int64>   AtomicInt64;

///@} Atomics
///@} Basic

} // namespace cugar
