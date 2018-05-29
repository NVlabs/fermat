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

#include <cugar/linalg/vector.h>

namespace cugar
{

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE uint32 to_rgbe(const float r, const float g, const float b)
{
    float v = 0;
    if (r > v) v = r;
    if (g > v) v = g;
    if (b > v) v = b;

	uint32 x = binary_cast<uint32>(v);
    int exponent = ((x >> 23u) & 0xFF) - 126u;
    int rgbe = (exponent + 128u) & 0xFF;
    if (rgbe < 10) return 0;
    x = ((((rgbe & 0xFF) - (128u + 8u)) + 127u) << 23u) & 0x7F800000;

    float f = 1.0f / binary_cast<float>(x);
    rgbe |= ((uint32) (r * f) << 24);
    rgbe |= ((uint32) (g * f) << 16);
    rgbe |= ((uint32) (b * f) << 8);
    return rgbe;
}

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE uint32 to_rgbe(const Vector3f rgb)
{
	return to_rgbe(rgb.x, rgb.y, rgb.z);
}

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE Vector3f from_rgbe(const uint32 rgbe)
{
    //const uint32 x = ((((rgbe & 0xFF) - (128u + 8u)) + 127u) << 23u) & 0x7F800000;
	const uint32 x = (((rgbe & 0xFF) - 9u) << 23u) & 0x7F800000;

    const float f = binary_cast<float>(x);
    return Vector3f(
        f *  (rgbe >> 24),
        f * ((rgbe >> 16) & 0xFF),
        f * ((rgbe >> 8)  & 0xFF) );
}

} // namespace cugar
