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

namespace cugar {

// rotate a zonal harmonics to an arbitrary direction vector
//
// \param L            number of bands
// \param zh_coeff     input Zonal Harmonics coefficients
// \param d            input vector
// \param sh_coeff     output Spherical Harmonics coefficients
template <typename ZHVector, typename SHVector, typename Vector3>
CUGAR_HOST_DEVICE void rotate_ZH(const int32 L, const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
{
    for (int32 l = 0; l < L; ++l)
        for (int32 m = -l; m <= l; ++m)
            sh_coeff[ l*l + m+l ] = sqrtf( 4.0f*M_PIf / float(2*l + 1) ) * zh_coeff[l] * sh( l, m, d );
}

// return the (l,m) spherical harmonics coefficient of a zonal harmonics
// function rotated to match a given axis.
//
// \param zh_l         l-band zonal harmonics coefficient
// \param d            input vector
template <int32 l, int32 m, typename Vector3>
CUGAR_HOST_DEVICE float rotate_ZH(const float zh_l, const Vector3& d)
{
    return sqrtf( 4.0f*M_PIf / float(2*l + 1) ) * zh_l * sh( l, m, d );
}

template <int32 l>
struct ZH_rotation
{
    template <int32 m>
    struct Apply
    {
        template <typename ZHVector, typename SHVector, typename Vector3>
        CUGAR_HOST_DEVICE static void eval(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
        {
            sh_coeff[ l*l + m+l ] = sqrtf( 4.0f*M_PIf / float(2*l + 1) ) * zh_coeff[l] * sh<l,m>( d );
            Apply<m+1>::eval( zh_coeff, d, sh_coeff );
        }
    };
    template <>
    struct Apply<l>
    {
        template <typename ZHVector, typename SHVector, typename Vector3>
        CUGAR_HOST_DEVICE static void eval(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
        {
            sh_coeff[ l*l + l+l ] = sqrtf( 4.0f*M_PIf / float(2*l + 1) ) * zh_coeff[l] * sh<l,l>( d );
        }
    };

    template <typename ZHVector, typename SHVector, typename Vector3>
    CUGAR_HOST_DEVICE static void eval(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
    {
        Apply<-l>::eval( zh_coeff, d, sh_coeff );
        if (l > 0)
            ZH_rotation<l-1>::eval( zh_coeff, d, sh_coeff );
    }
};
template <>
struct ZH_rotation<0>
{
    template <typename ZHVector, typename SHVector, typename Vector3>
    CUGAR_HOST_DEVICE static void eval(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
    {
        sh_coeff[0] = sqrtf( 4.0f*M_PIf ) * zh_coeff[0] * sh<0,0>( d );
    }
};

// rotate a zonal harmonics to an arbitrary direction vector, with
// the number of bands specified at compile-time.
//
// \param zh_coeff     input Zonal Harmonics coefficients
// \param d            input vector
// \param sh_coeff     output Spherical Harmonics coefficients
template <int32 L, typename ZHVector, typename SHVector, typename Vector3>
CUGAR_HOST_DEVICE void rotate_ZH(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff)
{
    ZH_rotation<L-1>::eval( zh_coeff, d, sh_coeff );
}

// evaluate the (l,m)-th basis function on a given vector
//
// \param l    band index
// \param m    subband index
// \param v    input vector
template <typename Vector3>
CUGAR_HOST_DEVICE float sh(const int32 l, const int32 m, const Vector3& v)
{
#if 0
    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? v[0] : (m == 0 ? v[2] : v[1]));
    else if (l == 2)
    {
        if (m == 0)
            return 0.315392f * (3*v[2]*v[2] - 1.0f);
        else if (m == 2)
            return 0.546274f * (v[0]*v[0] - v[1]*v[1]);
        else if (m == -2)
            return 1.092548f * v[0]*v[2];
        else if (m == -1)
            return 1.092548f * v[1]*v[2];
        else
            return 1.092548f * v[0]*v[1];
    }
#else
    const float X = v[0];
    const float Y = v[1];
    const float Z = v[2];

    const float m_15_over_4sqrtPI           = 0.54627419f; //sqrtf(15.0f)/(4.0f*sqrtf(M_PIf));
    const float m_15_over_2sqrtPI           = 1.09254837f; //sqrtf(15.0f)/(2.0f*sqrtf(M_PIf));
    const float m_5_over_4sqrtPI            = 0.31539154f; //sqrtf(5.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_8sqrtPI  = 0.59004354f; //sqrtf(2.0f*35.0f)/(8.0f*sqrtf(M_PIf));
    //const float m_sqrt2sqrt35_over_4sqrtPI  = 1.18008709f; //sqrtf(2.0f*35.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt7_over_4sqrtPI        = 0.37317631f; //sqrtf(7.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt21_over_8sqrtPI  = 0.45704576f; //sqrtf(2.0f*21.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_4sqrtPI      = 1.44530571f; //sqrtf(105.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_2sqrtPI      = 2.89061141f; //sqrtf(105.0f)/(2.0f*sqrtf(M_PIf));

    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? -Y : (m == 0 ? Z : -X));
    else if (l == 2)
    {
        if (m == 0)
            return m_5_over_4sqrtPI * (3*Z*Z - 1.0f);
        else if (m == 1)
            return -m_15_over_2sqrtPI * X*Z;
        else if (m == 2)
            return m_15_over_4sqrtPI * (X*X - Y*Y);
        else if (m == -1)
            return -m_15_over_2sqrtPI * Y*Z;
        else if (m == -2)
            return m_15_over_2sqrtPI * X*Y;
    }
    else if (l == 3)
    {
        if (m == 0)
            return m_sqrt7_over_4sqrtPI * Z * (5*Z*Z - 3);
        else if (m == 1)
            return -m_sqrt2sqrt21_over_8sqrtPI * X * (5*Z*Z - 1);
        else if (m == 2)
            return m_sqrt105_over_4sqrtPI * (X*X - Y*Y) * Z;
        else if (m == 3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (X*X - 3*Y*Y)*X;
        else if (m == -1)
            return -m_sqrt2sqrt21_over_8sqrtPI * Y * (5*Z*Z - 1);
        else if (m == -2)
            return m_sqrt105_over_2sqrtPI * X*Y*Z;
        else if (m == -3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (3*X*X - Y*Y)*Y;
    }
#endif
    return 0.0f;
}
// evaluate the (l,m)-th basis function on a given vector, where
// l and m are determined at compile-time.
//
// \param v    input vector
template <int32 l, int32 m, typename Vector3>
CUGAR_HOST_DEVICE float sh(const Vector3& v)
{
    const float X = v[0];
    const float Y = v[1];
    const float Z = v[2];

    const float m_15_over_4sqrtPI           = 0.54627419f; //sqrtf(15.0f)/(4.0f*sqrtf(M_PIf));
    const float m_15_over_2sqrtPI           = 1.09254837f; //sqrtf(15.0f)/(2.0f*sqrtf(M_PIf));
    const float m_5_over_4sqrtPI            = 0.31539154f; //sqrtf(5.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_8sqrtPI  = 0.59004354f; //sqrtf(2.0f*35.0f)/(8.0f*sqrtf(M_PIf));
    //const float m_sqrt2sqrt35_over_4sqrtPI  = 1.18008709f; //sqrtf(2.0f*35.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt7_over_4sqrtPI        = 0.37317631f; //sqrtf(7.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt21_over_8sqrtPI  = 0.45704576f; //sqrtf(2.0f*21.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_4sqrtPI      = 1.44530571f; //sqrtf(105.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_2sqrtPI      = 2.89061141f; //sqrtf(105.0f)/(2.0f*sqrtf(M_PIf));

    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? -Y : (m == 0 ? Z : -X));
    else if (l == 2)
    {
        if (m == 0)
            return m_5_over_4sqrtPI * (3*Z*Z - 1.0f);
        else if (m == 1)
            return -m_15_over_2sqrtPI * X*Z;
        else if (m == 2)
            return m_15_over_4sqrtPI * (X*X - Y*Y);
        else if (m == -1)
            return -m_15_over_2sqrtPI * Y*Z;
        else if (m == -2)
            return m_15_over_2sqrtPI * X*Y;
    }
    else if (l == 3)
    {
        if (m == 0)
            return m_sqrt7_over_4sqrtPI * Z * (5*Z*Z - 3);
        else if (m == 1)
            return -m_sqrt2sqrt21_over_8sqrtPI * X * (5*Z*Z - 1);
        else if (m == 2)
            return m_sqrt105_over_4sqrtPI * (X*X - Y*Y) * Z;
        else if (m == 3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (X*X - 3*Y*Y)*X;
        else if (m == -1)
            return -m_sqrt2sqrt21_over_8sqrtPI * Y * (5*Z*Z - 1);
        else if (m == -2)
            return m_sqrt105_over_2sqrtPI * X*Y*Z;
        else if (m == -3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (3*X*X - Y*Y)*Y;
    }
    return 0.0f;
}

// evaluate the (l,m)-th basis function on a given vector, where
// l is determined at compile-time.
//
// \param m    subband index
// \param v    input vector
template <int32 l, typename Vector3>
CUGAR_HOST_DEVICE float sh(const int32 m, const Vector3& v)
{
    if (l == 0)
        return sh<0,0>( v );
    else if (l == 1)
    {
        if (m == -1)
            return sh<1,-1>( v );
        else if (m == 0)
            return sh<1,0>( v );
        else
            return sh<1,1>( v );
    }
    else if (l == 2)
    {
        if (m == 0)
            return sh<2,0>( v );
        else if (m == 1)
            return sh<2,1>( v );
        else if (m == 2)
            return sh<2,2>( v );
        else if (m == -1)
            return sh<2,-1>( v );
        else if (m == -2)
            return sh<2,-2>( v );
    }
    else if (l == 3)
    {
        if (m == 0)
            return sh<3,0>( v );
        else if (m == 1)
            return sh<3,1>( v );
        else if (m == 2)
            return sh<3,2>( v );
        else if (m == 3)
            return sh<3,3>( v );
        else if (m == -1)
            return sh<3,-1>( v );
        else if (m == -2)
            return sh<3,-2>( v );
        else if (m == -3)
            return sh<3,-3>( v );
    }
    return 0.0f;
}

// evaluate the i-th coefficient at a given point
//
// \param i    coefficient index
// \param d    direction vector
template <int32 L>
template <typename Vector3>
CUGAR_HOST_DEVICE float SH_basis<L>::eval(const int32 i, const Vector3& d)
{
    if (i == 0)
        return sh<0>( 0, d );
    else if (i < 4)
        return sh<1>( i - 2, d );
    else if (i < 9)
        return sh<2>( i - 6, d );
    else
        return sh<3>( i - 12, d );
}

// add a weighted basis expansion of a clamped cosine lobe to a given
// set of coefficients
//
// \param normal   input normal
// \param w        scalar weight
// \param coeffs   input/output coefficients
template <int32 L>
CUGAR_HOST_DEVICE void SH_basis<L>::clamped_cosine(const Vector3f& normal, const float w, float* coeffs)
{
    const float zh[4] = {
        0.886214f,
        1.023202f,
        0.495443f,
        0.013224f };

    float sh[COEFFS];
    rotate_ZH<L>( zh, normal, sh );

    for (uint32 i = 0; i < COEFFS; ++i)
        coeffs[i] += sh[i] * w;
}

// return the basis expansion of a constant
//
// \param k        input constant
// \param coeffs   output coefficients
template <int32 L>
CUGAR_HOST_DEVICE void SH_basis<L>::constant(float k, float* coeffs)
{
    coeffs[0] = k * 2.0f*sqrtf(M_PIf);
    for (int32 i = 1; i < COEFFS; ++i)
        coeffs[i] = 0.0f;
}

// evaluate the associated Legendre polynomial P_l^m on x = cos(theta), y = sin(theta) = sqrt(1 - x*x)
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float sh_legendre_polynomial(const uint32 l, const uint32 m, const float x, const float y)
{
	if (l == 0)
		return 1;
	else if (l == 1)
	{
		if (m == 0)
			return x;
		else
			return -y;
	}
	else if (l == 2)
	{
		if (m == 0)
			return 0.5f * (3 * x*x - 1);
		else if (m == 1)
			return -3 * x * y;
		else
			return 3 * y*y;
	}
	else if (l == 3)
	{
		if (m == 0)
			return 0.5f * x * (5 * x*x - 3);
		else if (m == 1)
			return -3.0f / 2.0f * (5 * x*x - 1) * y;
		else if (m == 2)
			return 15.0f * x * y*y;
		else
			return -15.0f * y*y*y;
	}
	return 0.0f;
}

// evaluate the associated Legendre integrals for a basis of order n
// see Algorithm 1 in:
//
//   Importance Sampling Spherical Harmonics,
//   W.Jarosz, N.Carr, H.W.Jensen
//
template <typename OutputIterator>
CUGAR_HOST_DEVICE
void sh_legendre_integrals(const uint32 n, const float x, OutputIterator r)
{
	const float y = sqrtf(1.0f - x*x);

	// Evaluate equation 11:
	r[0] = x;

	if (n == 1)
		return;

	// Evaluate equation 12 and 13:
	r[1 * n + 0] = x * 0.5f;
	r[1 * n + 1] = 0.5f * (x * sqrtf(1 - x*x) + asin(x));

	for (uint32 l = 2; l < n; ++l)
	{
		for (uint32 m = 0; m < l - 1; ++m)
		{
			// Evaluate equation 9:
			const float g_x = (2 * l - 1)*(1 - x*x) * sh_legendre_polynomial(l - 1, m, x, y);
			r[l * n + m] = ((l - 2)*(l - 1 + m) * r[(l - 2)*n + m] - g_x) / float( (l+1)*(l-m) );
		}

		// Evaluate special case of equation 9:
		r[l * n + l - 1] = -(2*l - 1)/float(l+1) * ((1 - x*x) * sh_legendre_polynomial(l - 1, l - 1, x, y));

		// Evaluate Equation 10:
		r[l * n + l] = 1 / float(l + 1) * (l*(2*l - 3)*(2*l - 1) * r[(l - 2)*n + l - 2] + x * sh_legendre_polynomial( 1, 1, x ));
	}
}

// evaluate the integral of the polar angle function Phi^m(phi)
// see equation 14 in:
//
//   Importance Sampling Spherical Harmonics,
//   W.Jarosz, N.Carr, H.W.Jensen
//
// \param m		required order
// \param phi	the point of evaluation, in angular coordinates
//
CUGAR_HOST_DEVICE
float sh_polar_integral(const int32 m, const float phi)
{
	return 1 / float(m) * (m >= 0 ? sin(m*phi) : cos(m*phi));
}

} // namespace cugar

