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

#include <cugar/basic/types.h>

namespace cugar {

#if defined(__CUDACC__)
CUGAR_FORCEINLINE CUGAR_DEVICE uint32 device_popc(const uint32 i) { return __popc(i); }
CUGAR_FORCEINLINE CUGAR_DEVICE uint32 device_popc(const uint64 i) { return __popcll(i); }
#endif

// int32 popcount
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const int32 i)
{
    return popc(uint32(i));
}
// uint32 popcount
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint32 i)
{
#if defined(CUGAR_DEVICE_COMPILATION)
    return device_popc( i );
#elif defined(__GNUC__)
    return __builtin_popcount( i );
#else
    uint32 v = i;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    v = (v + (v >> 4)) & 0x0F0F0F0F;
    return (v * 0x01010101) >> 24;
#endif
}

// uint64 popcount
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint64 i)
{
#if defined(CUGAR_DEVICE_COMPILATION)
    return device_popc( i );
#elif defined(__GNUC__)
    return __builtin_popcountll( i );
#else
    //return popc( uint32(i & 0xFFFFFFFFU) ) + popc( uint32(i >> 32) );
    uint64 v = i;
    v = v - ((v >> 1) & 0x5555555555555555U);
    v = (v & 0x3333333333333333U) + ((v >> 2) & 0x3333333333333333U);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FU;
    return (v * 0x0101010101010101U) >> 56;
#endif
}

// uint8 popcount
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint8 i)
{
    const uint32 lut[16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
    return lut[ i & 0x0F ] + lut[ i >> 4 ];
}

// find the n-th bit set in a 4-bit mask (n in [1,4])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit4(const uint32 mask, const uint32 n)
{
    const uint32 popc0 = (mask & 1u);
    const uint32 popc1 = popc0 + ((mask >> 1u) & 1u);
    const uint32 popc2 = popc1 + ((mask >> 2u) & 1u);
    return (n <= popc1) ?
           (n == popc0) ? 0u : 1u :
           (n == popc2) ? 2u : 3u;
}

// compute the pop-count of 4-bit mask
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc4(const uint32 mask)
{
    return
         (mask        & 1u) +
        ((mask >> 1u) & 1u) +
        ((mask >> 2u) & 1u) +
        ((mask >> 3u) & 1u);
}

// find the n-th bit set in a 8-bit mask (n in [1,8])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit8(const uint32 mask, const uint32 n)
{
    const uint32 popc_half = popc4( mask );
    uint32 _mask;
    uint32 _n;
    uint32 _r;

    if (n <= popc_half)
    {
        _mask = mask;
        _n    = n;
        _r    = 0u;
    }
    else
    {
        _mask = mask >> 4u;
        _n    = n - popc_half;
        _r    = 4u;
    }
    return find_nthbit4( _mask, _n ) + _r;
}

// find the n-th bit set in a 16-bit mask (n in [1,16])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit16(const uint32 mask, const uint32 n)
{
    const uint32 popc_half = popc( mask & 0xFu );

    uint32 _mask;
    uint32 _n;
    uint32 _r;

    if (n <= popc_half)
    {
        _mask = mask;
        _n    = n;
        _r    = 0u;
    }
    else
    {
        _mask = mask >> 8u;
        _n    = n - popc_half;
        _r    = 8u;
    }
    return find_nthbit8( _mask, _n ) + _r;
}

// find the n-th bit set in a 32-bit mask (n in [1,32])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit(const uint32 mask, const uint32 n)
{
    const uint32 popc_half = popc( mask & 0xFFu );

    uint32 _mask;
    uint32 _n;
    uint32 _r;

    if (n <= popc_half)
    {
        _mask = mask;
        _n    = n;
        _r    = 0u;
    }
    else
    {
        _mask = mask >> 16u;
        _n    = n - popc_half;
        _r    = 16u;
    }
    return find_nthbit16( _mask, _n ) + _r;
}

// find the n-th bit set in a 8-bit mask (n in [1,8])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit(const uint8 mask, const uint32 n)
{
    return find_nthbit8( mask, n );
}

// find the n-th bit set in a 16-bit mask (n in [1,16])
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit(const uint16 mask, const uint32 n)
{
    return find_nthbit16( mask, n );
}

#if defined(CUGAR_DEVICE_COMPILATION)
CUGAR_FORCEINLINE CUGAR_DEVICE uint32 ffs_device(const int32 x) { return __ffs(x); }
CUGAR_FORCEINLINE CUGAR_DEVICE uint32 lzc_device(const int32 x) { return __clz(x); }
#endif

// find the least significant bit set
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 ffs(const int32 x)
{
#if defined(CUGAR_DEVICE_COMPILATION)
    return ffs_device(x);
#elif defined(__GNUC__)
    return __builtin_ffs(x);
#else
    return x ? popc(x ^ (~(-x))) : 0u;
#endif
}

// count the number of leading zeros
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 lzc(const uint32 x)
{
#if defined(CUGAR_DEVICE_COMPILATION)
    return lzc_device(x);
#elif defined(__GNUC__)
    return __builtin_clz(x);
#else
    uint32 y = x;
    y |= (y >> 1);
    y |= (y >> 2);
    y |= (y >> 4);
    y |= (y >> 8);
    y |= (y >> 16);
    return (32u - popc(y));
#endif
}

} // namespace cugar
