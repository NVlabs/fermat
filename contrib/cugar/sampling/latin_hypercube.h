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

 /*! \addtogroup LatinyHypercubeModule Latin Hypercube
 *  \ingroup sampling
 *  This module defines utilities to construct Latin Hypercube sampling patterns
 *  \{
 */

///
/// Latin-Hypercube Sampler
///
struct LHSampler
{
    /// constructor
    ///
	LHSampler(const uint32 seed = 0) : m_random(seed) {}

	/// get a set of stratified samples
	///
	template <typename T, uint32 DIM>
	void sample(
		const uint32	n_samples,
		Vector<T,DIM>*	samples);

	/// get a set of stratified d-dimensional samples
	/// the output vectors can be stored either in AOS or SOA layout, if the latter, the j-th coordinate of the i-th point is stored in samples[i + j * n_samples]
	///
	template <bool SOA, typename T>
	void sample(
		const uint32	n_samples,
		const uint32	n_dims,
		T*				samples);

	Random				m_random;
};

/*! \}
*/

/*! \}
 */

} // namespace cugar

#include <cugar/sampling/latin_hypercube_inline.h>
