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

/*! \file project.h
 *   \brief Defines a function to project spherical functions on a given basis.
 */

#pragma once

#include <cugar/basic/numbers.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>

namespace cugar {

///
/// Project a spherical function on a given basis.
/// NOTE: the coefficients for the projection are accumulated to the ones
/// passed in.
///
/// \param basis        the basis functions
/// \param fun          the function to project
/// \param a            a multiplier for the function
/// \param b            a bias for the function
/// \param n_samples    the number of samples used to evaluate the projection
/// \param coeffs       the output coefficients
/// 
template <typename Basis_type, typename Fun_type>
void project(
    const Basis_type    basis,
    const Fun_type&     fun,
    const float         a,
    const float         b,
    const int32         n_samples,
    float*              coeffs)
{
    const float w = 4.0f * M_PIf / float(n_samples);

    for (int32 s = 0; s < n_samples; ++s)
    {
        Vector2f uv;

        uv[0] = float(s) / float(n_samples);
        uv[1] = radical_inverse(s);

        const Vector3f dir = uniform_square_to_sphere( uv );

        const float f = a*fun( dir ) + b;

        for (int32 i = 0; i < Basis_type::COEFFS; ++i)
            coeffs[i] += Basis_type::eval( i, dir ) * f * w;
    }
}

} // namespace cugar
