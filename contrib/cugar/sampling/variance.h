/*
 * Copyright (c) 2010-2016, NVIDIA Corporation
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

/*! \file variance.h
 *   \brief Defines utilities for variance estimation
 *
 * This module provides utilities for variance estimation.
 */

#pragma once

#include <cugar/basic/types.h>


namespace cugar {

/*! \addtogroup sampling Sampling
 *  \{
 */

/// An implementation of a robust online variance estimator, using Welford's algorithm
///
template <typename T>
struct Variance_estimator
{
	CUGAR_HOST_DEVICE
	Variance_estimator() : m_mean(0.0f), m_M2(0.0f), m_n(0.0f) {}

	CUGAR_HOST_DEVICE
	Variance_estimator& operator=(const Variance_estimator& other)
	{
		m_n		= other.m_n;
		m_mean	= other.m_mean;
		m_M2	= other.m_M2;
		return *this;
	}

	CUGAR_HOST_DEVICE
	Variance_estimator& operator+=(const float x)
	{
		m_n += 1.0f;
		const T delta = x - m_mean;
		m_mean += delta / m_n;
		const T delta2 = x - m_mean;
		m_M2 += delta*delta2;
		return *this;
	}

	CUGAR_HOST_DEVICE
	float mean()	 const { return m_mean; }

	CUGAR_HOST_DEVICE
	float variance() const { return m_n > 1 ? m_M2 / (m_n - 1) : 0.0f; }

	T		m_mean;
	T		m_M2;
	float	m_n;
};

/*! \}
 */

} // namespace cugar
