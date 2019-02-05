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

/*! \file random.h
 *   \brief Defines several random samplers
 *
 * This module provides several random samplers.
 */

#pragma once

#include <cugar/basic/types.h>


namespace cugar {

///
/// \page sampling_page Sampling Module
///\par
/// This \ref sampling "module" provides all types of sampling constructs, including
/// sampling sequences, distributions, and stochastic processes.
///
/// - IRandom
/// - Random
/// - \ref LFSRModule
/// - \ref distributions
/// - \ref processes
/// - \ref multijitter
/// - \ref cp_rotations
/// - \ref ExpectationMaximizationModule
///

/*! \addtogroup sampling Sampling
 *  \{
 */

#define CUGAR_RAND_A 1664525
#define CUGAR_RAND_C 1013904223

/// A very simple Linear Congruential Generator
///
struct IRandom
{
	static const uint32 MAX = 0xFFFFFFFF;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE IRandom(const uint32 s = 0) : m_s(s) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 next() { m_s = m_s*CUGAR_RAND_A + CUGAR_RAND_C; return m_s; }

	uint32 m_s;
};

/// A very simple Linear Congruential Generator
///
struct Random
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE Random(const uint32 s = 0) : m_irand(s) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE float next() { return float(m_irand.next())/float(IRandom::MAX); }

	IRandom m_irand;
};

/*! \}
 */

} // namespace cugar
