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

    /// return next sample
    ///
    CUGAR_HOST_DEVICE CUGAR_FORCEINLINE float sample()
    {
        next_dim();
        //return cugar::mod( float(m_i) * m_r, 1.0f );
        return cugar::mod( float(m_i) * (float(m_r) / float(UINT_MAX)), 1.0f );
			// NOTE: The sequence of all multiples of an irrational number (m_r) is equidistributed modulo 1
    }
    /// return next sample
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

/*! \}
 */

} // namespace cugar
