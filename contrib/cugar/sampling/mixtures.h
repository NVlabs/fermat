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

/*! \file distributions.h
 *   \brief Defines various distributions.
 */

#pragma once

#include <cugar/sampling/random.h>
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>

namespace cugar {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup distributions Distributions
 *  \ingroup sampling
 *  \{
 */

///
/// A statically-sized mixture of parametric distributions
///
/// \tparam TDistribution		the underlying distribution type
/// \tparam NCOMP				the number of components, each of type TDistribution
///
template <typename TDistribution, uint32 NCOMP>
struct Mixture
{
    typedef TDistribution   distribution_type;

    static const uint32     NUM_COMPONENTS = NCOMP;


    /// constructor
    ///
    CUGAR_HOST_DEVICE Mixture()
	{
		// initialize the weights
		for (uint32 i = 0; i < NUM_COMPONENTS; ++i)
			weights[i] = 1.0f / NUM_COMPONENTS;
	}

    /// size of the mixture
    ///
    CUGAR_HOST_DEVICE 
    uint32 size() const { return NCOMP; }

    /// probability density function
    ///
    /// \param x    sample location
	CUGAR_HOST_DEVICE float density(const Vector2f x) const
	{
        float p = 0.0f;
        for (uint32 i = 0; i < NCOMP; ++i)
            p += weights[i] * comps[i].density(x);
        return p;
	}

    /// probability density function of the i-th component
    ///
    /// \param x    sample location
	CUGAR_HOST_DEVICE float density(const uint32 i, const Vector2f x) const
	{
        return comps[i].density(x);
	}

    /// return the i-th component
    ///
    /// \param i    component index
    CUGAR_HOST_DEVICE const distribution_type& component(const uint32 i) const { return comps[i]; }

    /// return the i-th component
    ///
    /// \param i    component index
	CUGAR_HOST_DEVICE distribution_type& component(const uint32 i) { return comps[i]; }

	/// set the i-th component
	///
	/// \param i    component index
	CUGAR_HOST_DEVICE void set_component(const uint32 i, const distribution_type& c, const float w) { comps[i] = c; weights[i] = w; }

	/// set the i-th component
    ///
    /// \param i    component index
	CUGAR_HOST_DEVICE void set_component(const uint32 i, const distribution_type& c) { comps[i] = c; }

    /// return the i-th component weight
    ///
    /// \param i    component index
	CUGAR_HOST_DEVICE float weight(const uint32 i) const { return weights[i]; }

    /// return the i-th component weight
    ///
    /// \param i    component index
	CUGAR_HOST_DEVICE void set_weight(const uint32 i, const float w) { weights[i] = w; }

private:
    TDistribution   comps[NCOMP];
    float           weights[NCOMP];
};

/*! \}
 */

} // namespace cugar
