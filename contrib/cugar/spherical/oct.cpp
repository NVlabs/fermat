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

#include <cugar/spherical/oct.h>
#include <cugar/linalg/linear.h>
#include <cugar/basic/functors.h>
#include <cugar/analysis/project.h>

namespace cugar {

Matrix<float,8,8>                   Oct_smooth_basis::s_G;
float                               Oct_smooth_basis::s_K[8];
Oct_smooth_basis::Initializer       Oct_smooth_basis::s_I;

// basis function object
struct Oct_smooth_basis_fun
{
    Oct_smooth_basis_fun(const int32 i) : m_i(i) {}

    float operator() (const Vector3f omega) const
    {
        return oct_smooth_basis( m_i, omega );
    }

    int32 m_i;
};

// initialize the Oct_smooth_basis::s_G matrix of joint dot products between basis functions
Oct_smooth_basis::Initializer::Initializer()
{
    const int32 n_samples = 256;

    // project each basis function against the full basis, forming a row of the matrix
    for (int32 i = 0; i < 8; ++i)
    {
        float coeffs[8] = {0};
        project(
            Oct_smooth_basis(),
            Oct_smooth_basis_fun( i ),
            1.0f,
            0.0f,
            n_samples,
            coeffs );

        s_G[i] = Vector<float,8>( coeffs );
    }

    // project the constant 1 function
    project(
        Oct_smooth_basis(),
        one_fun<Vector3f,float>(),
        1.0f,
        0.0f,
        n_samples,
        s_K );

    solve( s_K );
}

void Oct_basis::clamped_cosine(const Vector3f& normal, const float w, float* coeffs)
{
    const int32 n_samples = 256;

    project(
        Oct_basis(),
        clamped_cosine_functor<Vector3f>( normal ),
        w,
        0.0f,
        n_samples,
        coeffs );
}

void Oct_smooth_basis::clamped_cosine(const Vector3f& normal, const float w, float* coeffs)
{
    const int32 n_samples = 256;

    project(
        Oct_smooth_basis(),
        clamped_cosine_functor<Vector3f>( normal ),
        w,
        0.0f,
        n_samples,
        coeffs );

    solve( coeffs );
}

#define LEAST_SQUARES_SOLVER 1

void Oct_smooth_basis::solve(float* coeffs)
{
    const Vector<float,8> x = gaussian_elimination( s_G, Vector<float,8>( coeffs ) );

#if LEAST_SQUARES_SOLVER
    for (int32 i = 0; i < 8; ++i)
        coeffs[i] = x[i];
#else
    float w1 = 0.0f;
    float w2 = 0.0f;
    for (int32 i = 0; i < 8; ++i)
    {
        w1 += coeffs[i];
        w2 += x[i];
    }
    if (w1 == 0.0f)
        return;

    for (int32 i = 0; i < 8; ++i)
        coeffs[i] = coeffs[i] * w2 / w1;
#endif
}

} // namespace cugar

