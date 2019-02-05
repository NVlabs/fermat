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

/*! \file weyl.h
 *   \brief Define a Weyl sequence sampler.
 */

#pragma once

#include <cugar/basic/types.h>
#include <limits.h>

namespace cugar {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup weyl Weyl Sequences
 *  \ingroup sampling
 *  \{
 */

///
/// Weyl equidistributed sequence sampler class
///
/// Weyl's equidistribution theorem says that the sequence of all multiples of an irrational number r is equidistributed modulo 1.
/// This sampler uses such a generator for each dimension of a multi-dimensional sequence, where r is varied for each dimension.
///
class Weyl_sampler
{
  public:
    /// empty constructor
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE Weyl_sampler() : m_dim( unsigned int(-1) ), m_r(0), m_i(1) {}

    /// constructor
    ///
    /// \param instance     instance number
    /// \param seed         randomization seed
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE Weyl_sampler(unsigned int instance, unsigned int seed = 1) : m_dim( unsigned int(-1) ), m_r(seed), m_i(instance+1) {}

    /// return next sample (iterating over dimensions)
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE float sample()
    {
        next_dim();
        //return cugar::mod( float(m_i) * m_r, 1.0f );
        return cugar::mod( float(m_i) * (float(m_r) / float(UINT_MAX)), 1.0f );
			// NOTE: The sequence of all multiples of an irrational number (m_r) is equidistributed modulo 1
    }

	/// return next sample (iterating over dimensions)
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE float next() { return sample(); } // random number generator interface

  private:

    /// advance to next dimension
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE void next_dim()
    {
        m_dim++;
        //m_r = sqrtf( float(s_primes[m_dim]) );
        m_r = m_r * 1103515245u + 12345u;
    }

    unsigned int m_dim;
    unsigned int m_r;
    unsigned int m_i;
};

///
/// A Weyl equidistributed sequence sampler class optimized for a given number of dimensions
/// (see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences)
///
/// Weyl's equidistribution theorem says that the sequence of all multiples of an irrational number r is equidistributed modulo 1.
/// This sampler uses such a generator for each dimension of a multi-dimensional sequence, where r is varied for each dimension.
///
class Weyl_sampler_opt
{
  public:
    /// constructor
    ///
	/// \param n_dims       total number of dimensions
    /// \param instance     instance number
    /// \param seed         randomization seed
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE Weyl_sampler_opt(unsigned int n_dims = 1, unsigned int instance = 0, float seed = 0.0f) :
		m_dim( unsigned int(-1) ), m_seed(seed), m_i(instance+1), m_alpha(1.0f)
	{
		m_gamma = 1.0f / gamma(n_dims);
	}

    /// return next sample (iterating over dimensions)
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE float sample()
	{
		next_dim();

		// NOTE: The sequence of all multiples of an irrational number (m_alpha) is equidistributed modulo 1
		//

	  #if 1 // higher precision
		const double ia = double(m_seed) + double(m_i) * double(m_alpha);
		return float(ia - floor(ia));
	  #else // single precision suffers for large values of m_i
		const float ia = m_seed + float(m_i) * m_alpha;
		return ia - floorf(ia);
	  #endif
    }

	/// return next sample (iterating over dimensions)
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE float next() { return sample(); } // random number generator interface

	/// advance to the next instance
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE unsigned int next_instance() { return ++m_i; }

    /// advance to the next dimension
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE unsigned int next_dim()
	{
		++m_dim;
		
		//init_alpha();
		m_alpha *= m_gamma; // faster, but lower precision for large m_dim
		//m_alpha = fmodf( m_alpha, 1.0f );

		return m_dim;
	}

	/// set the instance number
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE void set_instance(unsigned int i) { m_i = i+1; }

	/// set the dimension index
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE void set_dim(unsigned int d) { m_dim = d-1; init_alpha(); }

private:

    /// initialize the sequence parameter for a given dimension index
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE void init_alpha()
    {
		m_alpha = powf( m_gamma, float(m_dim+1) );
		//m_alpha = fmodf( m_alpha, 1.0f );
    }

	/// compute the optimal coefficient phi_d for a d-dimensional sequence as the solution of the equation x^(d+1) = x + 1
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE static float gamma(unsigned int d)
    {
		const float c_gamma[128] = {
			1.6180339887f, 1.3247179572f, 1.2207440846f, 1.1673039783f,
			1.1347241384f, 1.1127756843f, 1.0969815578f, 1.0850702455f,
			1.0757660661f, 1.0682971889f, 1.0621691679f, 1.0570505752f,
			1.0527109201f, 1.0489849348f, 1.0457510242f, 1.0429177323f,
			1.0404149478f, 1.0381880194f, 1.0361937171f, 1.0343973961f,
			1.0327709664f, 1.0312914125f, 1.0299396967f, 1.0286999355f,
			1.0275587724f, 1.0265048941f, 1.0255286548f, 1.0246217791f,
			1.0237771279f, 1.0229885089f, 1.0222505253f, 1.0215584519f,
			1.0209081336f, 1.0202959021f, 1.0197185065f, 1.0191730558f,
			1.0186569702f, 1.0181679403f, 1.0177038930f, 1.0172629613f,
			1.0168434601f, 1.0164438641f, 1.0160627892f, 1.0156989769f,
			1.0153512803f, 1.0150186517f, 1.0147001323f, 1.0143948430f,
			1.0141019764f, 1.0138207892f, 1.0135505966f, 1.0132907659f,
			1.0130407123f, 1.0127998942f, 1.0125678091f, 1.0123439905f,
			1.0121280045f, 1.0119194469f, 1.0117179410f, 1.0115231351f,
			1.0113347005f, 1.0111523296f, 1.0109757344f, 1.0108046448f,
			1.0106388073f, 1.0104779837f, 1.0103219500f, 1.0101704953f,
			1.0100234209f, 1.0098805397f, 1.0097416747f, 1.0096066589f,
			1.0094753346f, 1.0093475523f, 1.0092231706f, 1.0091020557f,
			1.0089840804f, 1.0088691242f, 1.0087570728f, 1.0086478173f,
			1.0085412545f, 1.0084372859f, 1.0083358181f, 1.0082367617f,
			1.0081400320f, 1.0080455479f, 1.0079532320f, 1.0078630105f,
			1.0077748131f, 1.0076885723f, 1.0076042238f, 1.0075217059f,
			1.0074409597f, 1.0073619287f, 1.0072845589f, 1.0072087984f,
			1.0071345975f, 1.0070619086f, 1.0069906859f, 1.0069208854f,
			1.0068524651f, 1.0067853844f, 1.0067196043f, 1.0066550873f,
			1.0065917975f, 1.0065297001f, 1.0064687617f, 1.0064089503f,
			1.0063502348f, 1.0062925853f, 1.0062359732f, 1.0061803706f,
			1.0061257508f, 1.0060720880f, 1.0060193572f, 1.0059675344f,
			1.0059165963f, 1.0058665204f, 1.0058172851f, 1.0057688693f,
			1.0057212528f, 1.0056744159f, 1.0056283396f, 1.0055830056f,
			1.0055383960f, 1.0054944937f, 1.0054512819f, 1.0054087445f
		};

		if (d <= 128)
			return c_gamma[d-1];
		else
		{
			// use Newton-Raphson
			double x = 1.0f;
			for (uint32 i = 0; i < 20; ++i)
				x = x - (pow(x,double(d+1))-x-1.f) / (double(d+1)*pow(x,double(d))-1.f);
			return float(x);
		}
    }

	unsigned int m_dim;
    unsigned int m_i;
	float        m_gamma;
	float        m_alpha;
    float        m_seed;
};

/*! \}
 */

} // namespace cugar
