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

/*! \file processes.h
 *   \brief Defines utility functions to evaluate some stochastic processes.
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/sampling/distributions.h>
#include <cugar/sampling/sobol.h>
#include <cugar/sampling/sampler.h>

namespace cugar {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup processes Stochastic Processes
 *  \ingroup sampling
 *  \{
 */

///
/// Evaluate a DIM-dimensional Brownian bridge of length L at time t,
/// using a black-box Gaussian generator.
///
template <typename Generator, uint32 DIM>
Vector<float,DIM> brownian_bridge(
    const uint32        L,
    const uint32        t,
    Generator           gaussian);

///
/// A simple utility function to generate a DIM-dimensional Gaussian point
/// using the i-th point of a sample sequence, shifted by a Cranley-Patterson
/// rotation.
///
template <uint32 DIM, typename Distribution, typename Sampler_type>
Vector<float,DIM> generate_point(
    Sampler_type&             sampler,
    Distribution&             gaussian);

///
/// Evaluate the i/N-th DIM-dimensional Brownian bridge of length L at time t.
/// The bridges are created using L copies of a DIM-dimensional Sobol sequence,
/// first permuted and then shifted through Cranley-Patterson rotations.
/// The vector of permutations must contain L permutations of the indices [0,N-1],
/// and the vector of rotations must contain L * (DIM + (DIM & 1)) entries.
///
template <uint32 DIM, typename Sequence, typename Distribution>
Vector<float,DIM> brownian_bridge(
    Distribution&               gaussian,
    const float                 sigma,
    const uint32                N,
    const uint32                i,
    const uint32                L,
    const uint32                t,
    const Sequence&             sequence);

/*! \}
 */

} // namespace cugar

#include <cugar/sampling/processes_inline.h>
