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

/*! \file oct.h
 *   \brief Octahedral basis definitions and helper functions
 *
 * Defines a set of 8 orthogonal functions on the sphere, where
 * each function is constant over one of the faces of a octahedron
 * and zero outside.
 *
 * Additionally, introduces a smoothed version of these functions
 * which form a non-orthogonal system.
 */

#pragma once

#include <cugar/basic/numbers.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>

namespace cugar {

/*! \addtogroup spherical_functions Spherical
 */

/*! \addtogroup octahedral_functions Octahedral Functions
 *  \ingroup spherical_functions
 *  \{
 */

/// evaluate the i-th octahedral function
///
/// \param i        function index
/// \param d        input direction vector
CUGAR_API CUGAR_HOST_DEVICE float oct_basis(const int32 i, const Vector3f& d);

/// evaluate the i-th smoothed octahedral function
///
/// \param i        function index
/// \param d        input direction vector
CUGAR_API CUGAR_HOST_DEVICE float oct_smooth_basis(const int32 i, const Vector3f& omega);

///
/// An octahedral basis: the basis functions correspond to the characteristic
/// functions of the intersection of the octants and the sphere, and do not
/// overlap.
///
struct Oct_basis
{
    static const int32 COEFFS = 8;

    /// evaluate the i-th octahedral function
    ///
    /// \param i        function index
    /// \param d        input direction vector
    CUGAR_HOST CUGAR_DEVICE static float eval(const int32 i, const Vector3f& d) { return oct_basis( i, d ); }

    /// add a weighted basis expansion of a clamped cosine lobe to a given
    /// set of coefficients
    ///
    /// \param normal   input normal
    /// \param w        scalar weight
    /// \param coeffs   input/output coefficients
    CUGAR_API static void clamped_cosine(const Vector3f& normal, const float w, float* coeffs);

    /// return the basis expansion of a constant
    ///
    /// \param k        input constant
    /// \param coeffs   output coefficients
    static void constant(float k, float* coeffs)
    {
        for (int32 i = 0; i < 8; ++i)
            coeffs[i] = k * sqrtf(M_PIf/2.0f);
    }

    /// return the integral of a spherical hamonics function
    ///
    static float integral(const float* coeffs)
    {
        float r = 0.0f;
        for (int32 i = 0; i < 8; ++i)
            r += coeffs[i];
        return r;
    }

    /// return the integral of a spherical hamonics function
    ///
    template <typename Vector_type>
    static float integral(const Vector_type& coeffs)
    {
        float r = 0.0f;
        for (int32 i = 0; i < 8; ++i)
            r += coeffs[i];
        return r;
    }

    /// solve the linear least squares projection for a set of coefficients
    ///
    /// \param coeffs   input projection coefficients
    static void solve(float* coeffs) {}
};

///
/// A smoothed octahedral basis: the basis functions overlap and are not
/// orthogonal. As a consequence, projection requires solving a least
/// squares problem.
///
struct Oct_smooth_basis
{
    static const int32 COEFFS = 8;

    /// evaluate the i-th octahedral function
    ///
    /// \param i        function index
    /// \param d        input direction vector
    CUGAR_HOST_DEVICE static float eval(const int32 i, const Vector3f& d) { return oct_smooth_basis( i, d ); }

    /// add a weighted basis expansion of a clamped cosine lobe to a given
    /// set of coefficients
    ///
    /// \param normal   input normal
    /// \param w        scalar weight
    /// \param coeffs   input/output coefficients
	CUGAR_API static void clamped_cosine(const Vector3f& normal, const float w, float* coeffs);

    /// return the basis expansion of a constant
    ///
    /// \param k        input constant
    /// \param coeffs   output coefficients
    static void constant(float k, float* coeffs)
    {
        for (int32 i = 0; i < 8; ++i)
            coeffs[i] = k * s_K[i];
    }

    /// return the integral of a spherical hamonics function
    ///
    static float integral(const float* coeffs)
    {
        float r = 0.0f;
        for (int32 i = 0; i < 8; ++i)
            r += coeffs[i];
        return r;
    }

    /// return the integral of a spherical hamonics function
    ///
    template <typename Vector_type>
    static float integral(const Vector_type& coeffs)
    {
        float r = 0.0f;
        for (int32 i = 0; i < 8; ++i)
            r += coeffs[i];
        return r;
    }

    /// return the dot product of the i-th and j-th basis functions
    ///
    /// \param i    first function index
    /// \param j    second functio index
    static float G(const int32 i, const int32 j) { return s_G[i][j]; }

    /// solve the linear least squares projection for a set of coefficients
    ///
    /// \param coeffs   input projection coefficients
	CUGAR_API static void solve(float* coeffs);

private:
    struct Initializer { Initializer(); };

    friend Initializer;

    static Matrix<float,8,8> s_G;
    static float             s_K[8];
    static Initializer       s_I;
};

static Vector3f s_oct_verts[6] = {
    Vector3f(+1, 0, 0),
    Vector3f(-1, 0, 0),
    Vector3f( 0,+1, 0),
    Vector3f( 0,-1, 0),
    Vector3f( 0, 0,+1),
    Vector3f( 0, 0,-1),
};
static int32 s_oct_faces[8][3] = {
    {0,2,4},
    {2,1,4},
    {1,3,4},
    {3,0,4},
    {2,0,5},
    {1,2,5},
    {3,1,5},
    {0,3,5},
};

// evaluate the i-th octahedral basis
inline CUGAR_HOST CUGAR_DEVICE float oct_basis(const int32 i, const Vector3f& omega)
{
    const float norm = sqrtf(2.0f/M_PIf); // sqrt(8) / sqrt(4*PI)

    // oct_basis
    return
        ((1 - 2*(((i+1)>>1)&1))*omega[0] > 0.0f ? 1.0f : 0.0f)*
        ((1 - 2*((i>>1)&1)    )*omega[1] > 0.0f ? 1.0f : 0.0f)*
        ((1 - 2*(i>>2)        )*omega[2] > 0.0f ? 1.0f : 0.0f) * norm;
/*
    // the i-th basis is defined by all directions inside the spherical triangle {v1,v2,v3}
    const Vector3f v1 = s_oct_verts[ s_oct_faces[i][0] ];
    const Vector3f v2 = s_oct_verts[ s_oct_faces[i][1] ];
    const Vector3f v3 = s_oct_verts[ s_oct_faces[i][2] ];

    //
    // check whether omega intersects {v1,v2,v3}
    //

    // equivalent to checking that all three dot products with omega are positive, but we can make it cheaper...

    // first, assume v3 is the north/south pole, and check whether omega.z is on the same side or not.
    if (v3[2] * omega[2] < 0.0f)
        return 0.0f;

    // second, project omega on the XY plane, and test whether it intersects the edge (v1,v2):
    // in order for this to happen, both dot products of omega with v1 and v2 have to be positive.
    if (dot(v1,omega) < 0.0f || dot(v2,omega) < 0.0f)
        return 0.0f;

    return norm;
    */
}

// evaluate the i-th smooth octahedral basis
inline CUGAR_HOST_DEVICE float oct_smooth_basis(const int32 i, const Vector3f& omega)
{
    const Vector3f c(
        -0.66666666666666666666f * (((i+1)>>1) & 1) + 0.33333333333333333333f,
        -0.66666666666666666666f * ((i>>1)     & 1) + 0.33333333333333333333f,
        -0.66666666666666666666f * (i>>2)           + 0.33333333333333333333f );

    const float N    = 2.0f;
    const float norm = 2.170803763674771f; //sqrtf( (N + 1) / (2.0f * M_PIf) );

    const float d = max( dot( omega, c ), 0.0f );
    return fast_pow( d, N ) * norm;
}

/*! \}
 */

} // namespace cugar
