/*
 * cugar
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/linalg/matrix.h>
#include <cugar/sampling/distributions.h>
#include <cugar/sampling/mixtures.h>

namespace cugar {

///@addtogroup sampling
///@{

///@defgroup ExpectationMaximizationModule Expectation-Maximization
/// This module provides functions for performing Expectation-Maximization, both offline, online and with step-wise algorithms
///@{

/// Traditional Expectation Maximization for a statically-sized batch.
/// This algorithm can also be called over and over to refine the same mixture in step-wise fashion,
/// on fixed size mini-batches.
///
/// \tparam N				batch size
/// \tparam NCOMPONENTS		number of mixture components
///
/// \param mixture			input/output mixture to be learned
/// \param eta				learning factor, typically in the range [1,2)
/// \param x				samples
/// \param w				sample weights
///
template <uint32 N, uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void EM(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f*										x,
	const float*										w);

/// Online joint-entropy Expectation Maximization, as described in:
///
///>    <b>Batch and On-line Parameter Estimation of Gaussian Mixtures Based on the Joint Entropy</b>,
///>    Yoram Singer, Manfred K. Warmuth
///
/// and further extended to support importance sampling weights w.
///
/// \tparam NCOMPONENTS		number of mixture components
///
/// \param mixture			input/output mixture to be learned
/// \param eta				learning factor: can start in the range [1,2) and follow a decaying schedule
/// \param x				sample
/// \param w				sample weight
///
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void joint_entropy_EM(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f										x,
	const float											w = 1.0f);

/// Online step-wise joint-entropy Expectation Maximization, as described in:
///
///>    <b>Batch and On-line Parameter Estimation of Gaussian Mixtures Based on the Joint Entropy</b>,
///>    Yoram Singer, Manfred K. Warmuth
///
/// and further extended to support importance sampling weights w.
///
/// \tparam NCOMPONENTS		number of mixture components
///
/// \param mixture			input/output mixture to be learned
/// \param eta				learning factor, typically in the range [1,2)
/// \param x				samples
/// \param w				sample weights
///
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void joint_entropy_EM(
    Mixture<Gaussian_distribution_2d,NCOMPONENTS>&		mixture,
    const float											eta,
    const uint32										N,
    const Vector2f*										x,
    const float*										w);

/// Online step-wise Expectation-Maximization E-Step, as described in:
///
///>    <b>On-line Learning of Parametric Mixture Models for Light Transport Simulation</b>,
///>    Vorba et al.
///
/// \tparam NCOMPONENTS		number of mixture components
///
/// \param mixture			input/output mixture to be learned
/// \param eta				learning factor: must follow a schedule equal to i^-alpha	where i is the sample index, and alpha is in [0.6,0.9]
/// \param x				sample
/// \param w				sample weight
/// \param u				sufficient statistics
///
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void stepwise_E(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f										x,
	const float											w,
	Matrix<float, NCOMPONENTS, 8>&						u);					// sufficient statistics

/// Online step-wise Expectation-Maximization M-Step, as described in:
///
///>    <b>On-line Learning of Parametric Mixture Models for Light Transport Simulation</b>,
///>    Vorba et al.
///
/// \tparam NCOMPONENTS		number of mixture components
///
/// \param mixture			input/output mixture to be learned
/// \param eta				learning factor: must follow a schedule equal to i^-alpha	where i is the sample index, and alpha is in [0.6,0.9]
/// \param x				sample
/// \param w				sample weight
/// \param u				sufficient statistics
///
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void stepwise_M(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const Matrix<float, NCOMPONENTS, 8>&				u,					// sufficient statistics
	const uint32										N);					// total number of points

///@} ExpectationMaximizationModule
///@} sampling

} // namespace cugar

#include <cugar/sampling/em_inl.h>
