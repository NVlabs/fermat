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

/*! \file sampler.h
 *   \brief Defines several multidimensional samplers.
 *
 * This module provides several multi-dimensional samplers, ranging from
 * latin hypercube to multi-jittered.
 * It also provides helper objects to combine sample sets either by
 * layering of different sets for different dimensions, or by CP-rotations.
 * A sample set is represented as an image I, such that I[i][d] represents
 * the d-th coordinate of the i-th sample.
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/sampling/random.h>
#include <vector>
#include <algorithm>


namespace cugar {

/*! \addtogroup sampling Sampling
 *  \{
 */

 /*! \addtogroup multijitter Multi-Jittering
 *  \ingroup sampling
 *  This module defines utilities to construct multi-jittered sampling patterns
 *  \{
 */

/// The repeatable correlated multi-jitter function described in:
/// "Correlated Multi-Jittered Sampling", by Andrew Kensler
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float2 correlated_multijitter(const uint32 s, const uint32 m, const uint32 n, const uint32 p)
{
	const uint32 sx = permute(s % m, m, p * 0xa511e9b3);
	const uint32 sy = permute(s / m, n, p * 0x63d83595);
	const float jx = randfloat(s, p * 0xa399d265);
	const float jy = randfloat(s, p * 0x711ad6a5);
	return make_float2(
		(s % m + (sy + jx) / n) / m,
		(s / m + (sx + jy) / m) / n );
}

/// A stateful sampler that implements the repeatable correlated multi-jitter function described in:
/// "Correlated Multi-Jittered Sampling", by Andrew Kensler
///
struct CorrelatedMJSampler
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	CorrelatedMJSampler(const uint32 _s, const uint32 _m, const uint32 _n, const uint32 _p) :
		s(_s), m(_m), n(_n), p(_p), c(-1.0f) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float next()
	{
		if (c != -1.0f)
		{
			const float r = c; c = -1.0f;
			return r;
		}
		else
		{
			const float2 r = correlated_multijitter( s, m, n, p++ );
			c = r.y;
			return r.x;
		}
	}

	uint32 s;
	uint32 m;
	uint32 n;
	uint32 p;
	float  c;
};

///
/// Multi-Jittered Sampler
///
struct MJSampler
{
	enum Ordering
	{
		kXY,
		kRandom
	};

    /// constructor
    ///
    MJSampler(const uint32 seed = 0) : m_random(seed) {}

	/// get a set of 2d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,2>*	samples,
		Ordering		ordering = kRandom);

	/// get a set of 3d stratified samples
	/// the first 2 dimensions are multi-jittered, the third one
	/// is selected with latin hypercube sampliing wrt the first 2.
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,3>*	samples);

	/// get a set of 4d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,4>*	samples,
		Ordering		ordering = kRandom);

	struct Sample
	{
		uint32 x;
		uint32 y;
	};
	std::vector<Sample> m_sample_xy;
	Random				m_random;
};

/*! \}
*/

/*! \}
 */

} // namespace cugar

#include <cugar/sampling/multijitter_inline.h>
