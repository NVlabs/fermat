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

struct HSV
{
	HSV() {}
	HSV(float _h, float _s, float _v) : h(_h), s(_s), v(_v) {}
	HSV(float3 _in) : h(_in.x), s(_in.y), v(_in.z) {}

	float h;
	float s;
	float v;
};

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float3 hsv_to_rgb(const HSV in)
{
    float3 out;

    if (in.s <= 0.0)
	{
        out.x = in.v;
        out.y = in.v;
        out.z = in.v;
        return out;
    }
    float hh = in.h;

	if (hh >= 360.0f)
		hh = 0.0;
    
	hh /= 60.0f;

	long i = (long)hh;

	float ff = hh - i;
    float p = in.v * (1.0f - in.s);
    float q = in.v * (1.0f - (in.s * ff));
    float t = in.v * (1.0f - (in.s * (1.0f - ff)));

    switch(i) {
    case 0:
        out.x = in.v;
        out.y = t;
        out.z = p;
        break;
    case 1:
        out.x = q;
        out.y = in.v;
        out.z = p;
        break;
    case 2:
        out.x = p;
        out.y = in.v;
        out.z = t;
        break;
    case 3:
        out.x = p;
        out.y = q;
        out.z = in.v;
        break;
    case 4:
        out.x = t;
        out.y = p;
        out.z = in.v;
        break;
    case 5:
    default:
        out.x = in.v;
        out.y = p;
        out.z = q;
        break;
    }
    return out;     
}

} // namespace cugar
