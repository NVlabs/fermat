/*
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

#pragma once

#include <cugar/basic/types.h>

namespace cugar {

///@addtogroup BitsModule
///@{

/// int32 popcount
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const int32 i);

/// uint32 popcount
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint32 i);

/// uint8 popcount
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint8 i);

/// uint64 popcount
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc(const uint64 i);

/// find the n-th bit set in a 4-bit mask (n in [1,4])
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit4(const uint32 mask, const uint32 n);

/// compute the pop-count of 4-bit mask
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 popc4(const uint32 mask);

/// find the n-th bit set in a 8-bit mask (n in [1,8])
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 find_nthbit8(const uint32 mask, const uint32 n);

/// find the least significant bit set
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 ffs(const int32 x);


/// count the number of leading zeros
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 lzc(const uint32 x);

///@} BitsModule

} // namespace cugar

#include <cugar/bits/popcount_inl.h>