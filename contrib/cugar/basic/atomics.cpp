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

#include <cugar/basic/atomics.h>
#include <cugar/basic/threads.h>

#ifdef WIN32
#include <windows.h>
#endif

namespace cugar {

void host_release_fence()
{
    #if defined(__GNUC__)
    // make sure the other threads see the reference count before the output is set
    __atomic_thread_fence( __ATOMIC_RELEASE );
    #endif
}

void host_acquire_fence()
{
    #if defined(__GNUC__)
    // make sure the other threads see the reference count before the output is set
    __atomic_thread_fence( __ATOMIC_ACQUIRE );
    #endif
}

#ifdef WIN32

int32 host_atomic_increment(int32* value) { return InterlockedIncrement(reinterpret_cast<LONG volatile*>(value))-1; }
int64 host_atomic_increment(int64* value) { return InterlockedIncrement64((LONGLONG*)value)-1; }

int32 host_atomic_decrement(int32* value) { return InterlockedDecrement(reinterpret_cast<LONG volatile*>(value))+1; }
int64 host_atomic_decrement(int64* value) { return InterlockedDecrement64((LONGLONG*)value)+1; }

#elif defined(__GNUC__)

int32 host_atomic_increment(int32* value) { return __atomic_fetch_add( value, 1, __ATOMIC_RELAXED ); }
int64 host_atomic_increment(int64* value) { return __atomic_fetch_add( value, 1, __ATOMIC_RELAXED ); }

int32 host_atomic_decrement(int32* value) { return __atomic_fetch_sub( value, 1, __ATOMIC_RELAXED ); }
int64 host_atomic_decrement(int64* value) { return __atomic_fetch_sub( value, 1, __ATOMIC_RELAXED ); }

#endif

int32 host_atomic_add(int32* value, const int32 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_add( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return (int32)InterlockedExchangeAdd((LONG*)value,(uint32)op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const int32 old = *value;
    *value += op;
    return old;
#endif
}
uint32 host_atomic_add(uint32* value, const uint32 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_add( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedExchangeAdd((LONG*)value,op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint32 old = *value;
    *value += op;
    return old;
#endif
}
int64 host_atomic_add(int64* value, const int64 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_add( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return (int64)InterlockedExchangeAdd64((LONGLONG*)value,(uint64)op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const int64 old = *value;
    *value += op;
    return old;
#endif
}
uint64 host_atomic_add(uint64* value, const uint64 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_add( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedExchangeAdd64((LONGLONG*)value,op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint64 old = *value;
    *value += op;
    return old;
#endif
}
int32 host_atomic_sub(int32* value, const int32 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_sub( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return (int32)InterlockedExchangeAdd((LONG*)value,(LONG)-op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const int32 old = *value;
    *value -= op;
    return old;
#endif
}
uint32 host_atomic_sub(uint32* value, const uint32 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_sub( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedExchangeAdd((LONG*)value,(LONG)-int32(op));
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint32 old = *value;
    *value -= op;
    return old;
#endif
}

int64 host_atomic_sub(int64* value, const int64 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_sub( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return (int64)InterlockedExchangeAdd64((LONGLONG*)value,(LONGLONG)-op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const int64 old = *value;
    *value -= op;
    return old;
#endif
}
uint64 host_atomic_sub(uint64* value, const uint64 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_sub( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedExchangeAdd64((LONGLONG*)value,(LONGLONG)-int64(op));
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint64 old = *value;
    *value -= op;
    return old;
#endif
}

uint32 host_atomic_or(uint32* value, const uint32 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_or( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedOr((LONG*)value,op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint32 old = *value;
    *value |= op;
    return old;
#endif
}
uint64 host_atomic_or(uint64* value, const uint64 op)
{
#if defined(__GNUC__)
    return __atomic_fetch_or( value, op, __ATOMIC_RELAXED );
#elif defined(WIN32)
    return InterlockedOr64((LONGLONG*)value,op);
#else
    Mutex mutex;
    ScopedLock lock( &mutex );

    const uint64 old = *value;
    *value |= op;
    return old;
#endif
}

CUGAR_API float host_atomic_add(float* value, const float op)
{
	while (1)
	{
		const uint32 old_bits = *(volatile uint32*)value;
		const uint32 new_bits = cugar::binary_cast<uint32>(cugar::binary_cast<float>(old_bits) + op);
	#if defined(__GNUC__)
		if (__sync_bool_compare_and_swap((uint32*)value, old_bits, new_bits))
	#elif defined(WIN32)
		if (InterlockedCompareExchange((uint32*)value, new_bits, old_bits) == old_bits)
	#endif
			return cugar::binary_cast<float>(old_bits);
	}
}

} // namespace cugar
