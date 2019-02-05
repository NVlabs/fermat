/*
 * Copyright (c) 2010-2019, NVIDIA Corporation
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

/*! \file distributions.h
 *   \brief Defines various distributions.
 */

#ifndef __CUGAR_DISTRIBUTIONS_H
#define __CUGAR_DISTRIBUTIONS_H

#include <cugar/sampling/random.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>

namespace cugar {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup distributions Distributions
 *  \ingroup sampling
 * \par
 *  This module provides a list of classes that provide methods to transform uniformly distributed numbers into
 *  different types of random distributions, and evaluate their density.
 *  Each such class provides the following basic interface:
 *  \code
 *  struct Distribution
 *  {
 *      // map a uniformly-distributed number 'U into the desired distribution
 *      float map(float U) const;
 *
 *      // evaluate the density of the distribution at the point 'x
 *      float density(float x) const;
 *  }
 *  \endcode
 * \par
 *  The module further provides an adaptor for random number generators, that can be used to adapt a
 *  generator to the desired distribution.
 *  e.g.
 *  \code
 *  Random              random_generator;
 *  Cosine_distribution cos_distribution;
 *  Transform_generator<Random,Cosine_distribution> cos_generator(random_generator,cos_distribution);
 *
 *  // approximate the average of the cosine distribution
 *  float avg = 0.0f;
 *  for (uint32 i = 0; i < 8; ++)
 *     avg += cos_distribution.next();
 *  avg /= 8;
 *  \endcode
 *
 *  \{
 */

///
/// Base distribution class
///
template <typename Derived_type>
struct Base_distribution
{
    /// return the next number in the sequence mapped through the distribution
    ///
    /// \param gen      random number generator
    template <typename Generator>
	inline float next(Generator& gen) const
	{
        return static_cast<const Derived_type*>(this)->map( gen.next() );
    }
};

///
/// Uniform distribution in [0,1]
///
struct Uniform_distribution : Base_distribution<Uniform_distribution>
{
    /// constructor
    ///
    /// \param b    distribution parameter
	CUGAR_HOST_DEVICE Uniform_distribution(const float b) : m_b( b ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const { return m_b * (U*2.0f - 1.0f); }

    /// probability density function
    ///
    /// \param x    sample location
    inline CUGAR_HOST_DEVICE float density(const float x) const
    {
        return (x >= -m_b && x <= m_b) ? m_b * 2.0f : 0.0f;
    }

private:
    float m_b;
};
///
/// Cosine distribution
///
struct Cosine_distribution : Base_distribution<Cosine_distribution>
{
    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
    {
        return asin( 0.5f * U ) * 2.0f / M_PIf;
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline CUGAR_HOST_DEVICE float density(const float x) const
    {
        if (x >= -1.0f && x <= 1.0f)
            return M_PIf*0.25f * cosf( x * M_PIf*0.5f );
        return 0.0f;
    }
};
///
/// Pareto distribution
///
struct Pareto_distribution : Base_distribution<Pareto_distribution>
{
    /// constructor
    ///
    /// \param a    distribution parameter
    /// \param min  distribution parameter
	CUGAR_HOST_DEVICE Pareto_distribution(const float a, const float min) : m_a( a ), m_inv_a( 1.0f / a ), m_min( min ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
    {
        return U < 0.5f ?
             m_min / powf( (0.5f - U)*2.0f, m_inv_a ) :
            -m_min / powf( (U - 0.5f)*2.0f, m_inv_a );
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline CUGAR_HOST_DEVICE float density(const float x) const
    {
        if (x >= -m_min && x <= m_min)
            return 0.0f;

        return 0.5f * m_a * powf( m_min, m_a ) / powf( fabsf(x), m_a + 1.0f );
    }

private:
    float m_a;
	float m_inv_a;
    float m_min;
};
///
/// Bounded Pareto distribution
///
struct Bounded_pareto_distribution : Base_distribution<Bounded_pareto_distribution>
{
    /// constructor
    ///
    /// \param a    distribution parameter
    /// \param min  distribution parameter
    /// \param max  distribution parameter
	CUGAR_HOST_DEVICE Bounded_pareto_distribution(const float a, const float min, const float max) :
        m_a( a ), m_inv_a( 1.0f / a ), m_min( min ), m_max( max ),
        m_min_a( powf( m_min, m_a ) ),
        m_max_a( powf( m_max, m_a ) ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
    {
        if (U < 0.5f)
        {
            const float u = (0.5f - U)*2.0f;
            return powf( -(u * m_max_a - u * m_min_a - m_max_a) / (m_max_a*m_min_a), -m_inv_a);
        }
        else
        {
            const float u = (U - 0.5f)*2.0f;
            return -powf( -(u * m_max_a - u * m_min_a - m_max_a) / (m_max_a*m_min_a), -m_inv_a);
        }
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline CUGAR_HOST_DEVICE float density(const float x) const
    {
        if (x >= -m_min && x <= m_min)
            return 0.0f;
        if (x <= -m_max || x >= m_max)
            return 0.0f;

        return 0.5f * m_a * m_min_a * powf( fabsf(x), -m_a - 1.0f ) / (1.0f - m_min_a / m_max_a);
    }

private:
    float m_a;
	float m_inv_a;
    float m_min;
    float m_max;
    float m_min_a;
    float m_max_a;
};
///
/// Bounded exponential distribution
///
struct Bounded_exponential : Base_distribution<Bounded_exponential>
{
    /// constructor
    ///
    /// \param b    distribution parameter
	CUGAR_HOST_DEVICE Bounded_exponential(const float b) :
		m_s1( b / 16.0f ),
		m_s2( b ),
		m_ln( -logf(m_s2/m_s1) ) {}

	/// constructor
	///
	/// \param b    distribution parameter
	CUGAR_HOST_DEVICE Bounded_exponential(const float b1, const float b2) :
		m_s1(b1),
		m_s2(b2),
		m_ln(-logf(m_s2 / m_s1)) {}

	/// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
	{
		return U < 0.5f ?
			+m_s2 * expf( m_ln*(0.5f - U)*2.0f ) :
			-m_s2 * expf( m_ln*(U - 0.5f)*2.0f );
	}
	/// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const float x) const
	{
		// positive x:
		// => x / s2 = exp( ln * (0.5 - U) * 2
		// => log( x / s2 ) = ln * (0.5 - U) * 2
		// => log( x / s2 ) / (2 * ln) = (0.5 - U)
		// => U = 0.5 - log( x / s2 ) / (2 * ln)
		// => CDF(x) = 1 - (0.5 - log( x / s2 ) / (2 * ln))
		// => PDF(x) = 1 / (x * 2 * ln) 

		// negative x:
		// => -x / s2 = exp( ln * (U - 0.5) * 2
		// => log( -x / s2 ) = ln * (U - 0.5) * 2
		// => log( -x / s2 ) / (2 * ln) = (U - 0.5)
		// => U = log( -x / s2 ) / (2 * ln) + 0.5
		// => CDF(x) = 1 - (log( -x / s2 ) / (2 * ln) + 0.5)
		// => PDF(x) = -1 / (x * 2 * ln)
		return
			x > +m_s2 ?  0.5f / (x * 2 * m_ln) :
			x < -m_s2 ? -0.5f / (x * 2 * m_ln) :
						 0.0f;
	}

private:
    float m_s1;
	float m_s2;
	float m_ln;
};
///
/// Cauchy distribution
///
struct Cauchy_distribution : Base_distribution<Cauchy_distribution>
{
    /// constructor
    ///
    /// \param gamma    distribution parameter
	CUGAR_HOST_DEVICE Cauchy_distribution(const float gamma) :
		m_gamma( gamma ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
	{
		return m_gamma * tanf( float(M_PI) * (U - 0.5f) );
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const float x) const
	{
		return (m_gamma / (x*x + m_gamma*m_gamma)) / float(M_PI);
	}

private:
    float m_gamma;
};
///
/// Exponential distribution
///
struct Exponential_distribution : Base_distribution<Exponential_distribution>
{
    /// constructor
    ///
    /// \param lambda   distribution parameter
	CUGAR_HOST_DEVICE Exponential_distribution(const float lambda) :
		m_lambda( lambda ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline CUGAR_HOST_DEVICE float map(const float U) const
	{
		const float eps = 1.0e-5f;
		return U < 0.5f ?
			-logf( cugar::max( (0.5f - U)*2.0f, eps ) ) / m_lambda :
			 logf( cugar::max( (U - 0.5f)*2.0f, eps ) ) / m_lambda;
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const float x) const
	{
		return 0.5f * m_lambda * expf( -m_lambda * fabsf(x) );
	}

private:
    float m_lambda;
};
///
/// Symmetric 2d Gaussian distribution
///
struct Gaussian_distribution_symm_2d
{
    /// constructor
    ///
    /// \param sigma    variance
	CUGAR_HOST_DEVICE Gaussian_distribution_symm_2d(const float sigma) :
		m_sigma( sigma ) {}

    /// transform a uniformly distributed vector through the distribution
    ///
    /// \param uv   real numbers to transform
	inline CUGAR_HOST_DEVICE Vector2f map(const Vector2f uv) const
	{
		const float eps = 1.0e-5f;
		const float r = m_sigma * sqrtf( - 2.0f * logf(cugar::max( uv[0], eps ) ) );
		return Vector2f(
			r * cosf( 2.0f * float(M_PI) * uv[1] ),
			r * sinf( 2.0f * float(M_PI) * uv[1] ) );
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const Vector2f x) const
	{
        return expf( -square_length(x) / (2.0f * m_sigma*m_sigma) ) / (2.0f * float(M_PI) * m_sigma*m_sigma);
	}

private:
	float m_sigma;
};
///
/// 2d Gaussian distribution with fully prescribed covariance matrix
///
struct Gaussian_distribution_2d
{
    enum MatrixType { COVARIANCE_MATRIX, PRECISION_MATRIX };

	/// constructor
	///
	/// \param mu           mean
	/// \param matrix       covariance | precision matrix
	/// \param matrix_type  specify which matrix is passed
	CUGAR_HOST_DEVICE Gaussian_distribution_2d()
	{
		m_mu = cugar::Vector2f(0.0f, 0.0f);

		m_prec[0] = cugar::Vector2f(1.0f, 0.0f);
		m_prec[1] = cugar::Vector2f(0.0f, 1.0f);
		m_chol[0] = cugar::Vector2f(1.0f, 0.0f);
		m_chol[1] = cugar::Vector2f(0.0f, 1.0f);

		m_norm = 1.0f;
	}
		
	/// constructor
    ///
    /// \param mu           mean
    /// \param matrix       covariance | precision matrix
    /// \param matrix_type  specify which matrix is passed
	CUGAR_HOST_DEVICE Gaussian_distribution_2d(const Vector2f mu, const Matrix2x2f matrix, const MatrixType matrix_type = COVARIANCE_MATRIX) :
		m_mu( mu )
    {
        if (matrix_type == COVARIANCE_MATRIX)
        {
            // compute the precision matrix
            const Matrix2x2f& sigma = matrix;
            invert( sigma, m_prec );

            // compute the cholesky factorization of sigma
            cholesky( sigma, m_chol );

            // compute the normalization constant
            m_norm = 1.0f / sqrtf( 2.0f * float(M_PI) * det(sigma) );
        }
        else
        {
            m_prec = matrix;

            // compute the covariance matrix
            Matrix2x2f sigma;
            invert( m_prec, sigma );

            // compute the cholesky factorization of sigma
            cholesky( sigma, m_chol );

            // compute the normalization constant
            m_norm = 1.0f / sqrtf( 2.0f * float(M_PI) * det(sigma) );
        }
    }

    /// constructor
    ///
    /// \param sigma    covariance matrix
    /// \param prec     precision matrix
	CUGAR_HOST_DEVICE Gaussian_distribution_2d(const Vector2f mu, const Matrix2x2f sigma, const Matrix2x2f prec) :
        m_mu( mu ), m_prec( prec )
    {
        cholesky( sigma, m_chol );

        // compute the normalization constant
        m_norm = 1.0f / sqrtf( 2.0f * float(M_PI) * det(sigma) );
    }

    /// transform a uniformly distributed vector through the distribution
    ///
    /// \param uv   real numbers to transform
	inline CUGAR_HOST_DEVICE Vector2f map(const Vector2f uv) const
	{
		const float eps = 1.0e-5f;
		const float r = sqrtf( - 2.0f * logf(cugar::max( uv[0], eps ) ) );
        const Vector2f normal = Vector2f(
			r * cosf( 2.0f * float(M_PI) * uv[1] ),
			r * sinf( 2.0f * float(M_PI) * uv[1] ) );

        return m_mu + m_chol * normal;
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const Vector2f x) const
	{
        const float x2 = dot( x - m_mu, m_prec * (x - m_mu) );
        return m_norm * expf( -0.5f * x2 );
	}

    /// return the mean vector
    ///
	CUGAR_HOST_DEVICE Vector2f mean() const
	{
        return m_mu;
	}

    /// return the precision matrix
    ///
	CUGAR_HOST_DEVICE const Matrix2x2f& precision() const
	{
        return m_prec;
	}

    /// return the covariance matrix
    ///
	CUGAR_HOST_DEVICE const Matrix2x2f covariance() const
	{
        Matrix2x2f sigma;
        invert( m_prec, sigma );
        return sigma;
	}

private:
    Vector2f   m_mu;
    Matrix2x2f m_prec;
    Matrix2x2f m_chol;
    float      m_norm;
};

///
/// Wrapper class to transform a random number generator with a given distribution
///
template <typename Generator, typename Distribution>
struct Transform_generator
{
    /// constructor
    ///
    /// \param gen      generator to wrap
    /// \param dist     transforming distribution
    CUGAR_HOST_DEVICE Transform_generator(Generator& gen, const Distribution& dist) : m_gen( gen ), m_dist( dist ) {}

    /// return the next number in the sequence
    ///
	inline CUGAR_HOST_DEVICE float next() const
	{
        return m_dist.map( m_gen.next() );
    }
    /// probability density function
    ///
    /// \param x    sample location
	inline CUGAR_HOST_DEVICE float density(const float x) const
	{
        return m_dist.density( x );
    }

    Generator&   m_gen;
    Distribution m_dist;
};

///
/// Wrapper class to generate Gaussian distributed numbers out of a uniform
/// random number generator
///
struct Gaussian_generator
{
    /// constructor
    ///
    /// \param sigma    variance
	CUGAR_HOST_DEVICE Gaussian_generator(const float sigma) :
		m_sigma( sigma ),
		m_cached( false ) {}

    /// return the next number in the sequence
    ///
    /// \param random   random number generator
    template <typename Generator>
	inline CUGAR_HOST_DEVICE float next(Generator& random)
	{
		if (m_cached)
		{
			m_cached = false;
			return m_cache;
		}

		const Vector2f uv( random.next(), random.next() );
		const float eps = 1.0e-5f;
		const float r = m_sigma * sqrtf( - 2.0f * logf(cugar::max( uv[0], eps ) ) );
		const float y0 = r * cosf( 2.0f * float(M_PI) * uv[1] );
		const float y1 = r * sinf( 2.0f * float(M_PI) * uv[1] );

		m_cache  = y1;
		m_cached = true;
		return y0;
	}
    /// probability density function
    ///
    /// \param x    sample location
    inline CUGAR_HOST_DEVICE float density(const float x) const
    {
        const float SQRT_TWO_PI = sqrtf(2.0f * M_PIf);
        return expf( -x*x/(2.0f*m_sigma*m_sigma) ) / (SQRT_TWO_PI*m_sigma);
    }

private:
    float  m_sigma;
	float  m_cache;
	bool   m_cached;
};

/*! \}
 */

} // namespace cugar

#endif