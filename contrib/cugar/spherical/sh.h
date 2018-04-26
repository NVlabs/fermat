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

/*! \file sh.h
 *   \brief Defines spherical and zonal harmonics functions and classes.
 */

#pragma once

#include <cugar/basic/numbers.h>
#include <cugar/linalg/vector.h>
#include <cugar/analysis/project.h>
#include <cugar/basic/functors.h>

namespace cugar {

/*! \addtogroup spherical_functions Spherical
 */

/*! \addtogroup spherical_harmonics Spherical and Zonal Harmonics
 *  \ingroup spherical_functions
 *  \{
 */

/// evaluate the (l,m)-th basis function on a given vector
///
/// \param l    band index
/// \param m    subband index
/// \param v    input vector
template <typename Vector3>
CUGAR_HOST_DEVICE float sh(const int32 l, const int32 m, const Vector3& v);

/// evaluate the (l,m)-th basis function on a given vector, where
/// l is determined at compile-time.
///
/// \param m    subband index
/// \param v    input vector
template <int32 l, typename Vector3>
CUGAR_HOST_DEVICE float sh(const int32 m, const Vector3& v);

/// evaluate the (l,m)-th basis function on a given vector, where
/// l and m are determined at compile-time.
///
/// \param v    input vector
template <int32 l, int32 m, typename Vector3>
CUGAR_HOST_DEVICE float sh(const Vector3& v);

/// rotate a zonal harmonics to an arbitrary direction vector
///
/// \param L            number of bands
/// \param zh_coeff     input Zonal Harmonics coefficients
/// \param d            input vector
/// \param sh_coeff     output Spherical Harmonics coefficients
template <typename ZHVector, typename SHVector, typename Vector3>
CUGAR_HOST_DEVICE void rotate_ZH(const int32 L, const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff);

/// rotate a zonal harmonics to an arbitrary direction vector, with
/// the number of bands specified at compile-time.
///
/// \param zh_coeff     input Zonal Harmonics coefficients
/// \param d            input vector
/// \param sh_coeff     output Spherical Harmonics coefficients
template <int32 L, typename ZHVector, typename SHVector, typename Vector3>
CUGAR_HOST_DEVICE void rotate_ZH(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff);

/// return the (l,m) spherical harmonics coefficient of a zonal harmonics
/// function rotated to match a given axis.
///
/// \param zh_l         l-band zonal harmonics coefficient
/// \param d            input vector
template <int32 l, int32 m, typename Vector3>
CUGAR_HOST_DEVICE float rotate_ZH(const float zh_l, const Vector3& d);

///
/// Spherical harmonics basis functions of order L
///
template <int32 L>
struct SH_basis
{
    static const int32 ORDER  = L;
    static const int32 COEFFS = L*L;

    /// evaluate the i-th coefficient at a given point
    ///
    /// \param i    coefficient index
    /// \param d    direction vector
    template <typename Vector3>
    static CUGAR_HOST_DEVICE float eval(const int32 i, const Vector3& d);

    /// add a weighted basis expansion of a clamped cosine lobe to a given
    /// set of coefficients
    ///
    /// \param normal   input normal
    /// \param w        scalar weight
    /// \param coeffs   input/output coefficients
    static CUGAR_HOST_DEVICE void clamped_cosine(const Vector3f& normal, const float w, float* coeffs);

    /// return the basis expansion of a constant
    ///
    /// \param k        input constant
    /// \param coeffs   output coefficients
    static CUGAR_HOST_DEVICE void constant(float k, float* coeffs);

    /// return the integral of a spherical hamonics function
    ///
    static CUGAR_HOST_DEVICE float integral(const float* coeffs) { return coeffs[0]; }

    /// return the integral of a spherical hamonics function
    ///
    template <typename Vector_type>
    static CUGAR_HOST_DEVICE float integral(const Vector_type& coeffs) { return coeffs[0]; }

    /// solve the linear least squares projection for a set of coefficients
    ///
    /// \param coeffs   input projection coefficients
    static CUGAR_HOST_DEVICE void solve(float* coeffs) {}
};

/// evaluate the associated Legendre polynomial P_l^m on x = cos(theta), y = sin(theta) = sqrt(1 - x*x)
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float sh_legendre_polynomial(const uint32 l, const uint32 m, const float x, const float y);

/// evaluate the associated Legendre integrals for a basis of order n
/// see Algorithm 1 in:
///
///   Importance Sampling Spherical Harmonics,
///   W.Jarosz, N.Carr, H.W.Jensen
///
/// \param n		SH order
/// \param x		cos(theta), the point of evaluation of the associated (indefinite) Legendre integrals
/// \param r		a pointer to the output evaluations for all (l,m) with (0 <= l < n), (0 <= m <= l)
///
template <typename OutputIterator>
CUGAR_HOST_DEVICE
void sh_legendre_integrals(const uint32 n, const float x, OutputIterator r);

/// evaluate the integral of the polar angle function Phi^m(phi)
/// see equation 14 in:
///
///   Importance Sampling Spherical Harmonics,
///   W.Jarosz, N.Carr, H.W.Jensen
///
/// \param m		required order
/// \param phi	the point of evaluation, in angular coordinates
///
CUGAR_HOST_DEVICE
float sh_polar_integral(const int32 m, const float phi);

/*! \}
 */

} // namespace cugar

#include <cugar/spherical/sh_inline.h>
