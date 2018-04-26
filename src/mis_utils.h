/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <types.h>

/// The usual balance heuristic
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float balance_heuristic(const float p1, const float p2)
{
	return p1 / (p1 + p2);
}

/// The usual power heuristic with beta = 2
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float power_heuristic(const float p1, const float p2)
{
	return (p1 * p1) / (p1 * p1 + p2 * p2);
}

/// The usual cutoff heuristic
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float cutoff_heuristic(const float p1, const float p2, const float alpha = 0.1f)
{
	const float p_max = fmaxf(p1, p2);
	const float q1 = p1 < alpha * p_max ? 0.0f : p1;
	const float q2 = p2 < alpha * p_max ? 0.0f : p2;
	return q1 / (q1 + q2);
}

FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float soft_cutoff_factor(const float w, const float threshold = 0.05f)
{
	const float t = fminf(w / threshold, 1.0f);

	return t * t * (3.0f - 2.0f * t);
}

/// A hybrid between the balance and maximum heuristic, where low probability values get soft-cutoff towards zero
/// before applying the balance heuristic
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float soft_cutoff_heuristic(const float p1, const float p2, const float threshold = 0.05f)
{
	// a balance heuristic which applies soft-clamping to low relative values of p_i
	const float q1 = p1 * soft_cutoff_factor(p1 / (p1 + p2), threshold);
	const float q2 = p2 * soft_cutoff_factor(p2 / (p1 + p2), threshold);
	return q1 / (q1 + q2);
}

/// A balance heuristic with downweighting of MIS values below a certain threshold
/// NOTE: this heuristic is biased!
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float biased_downweighting_heuristic(const float p1, const float p2, const float threshold = 0.05f)
{
	// a balance heuristic which applies soft-clamping to low relative values of p_i
	const float w = p1 / (p1 + p2);
	return w * soft_cutoff_factor(w, threshold);
}

/// An enumeration of all MIS heuristics
///
enum MisHeuristicType
{
	BALANCE_HEURISTIC				= 0,
	POWER_HEURISTIC					= 1,
	CUTOFF_HEURISTIC				= 2,
	SOFT_CUTOFF_HEURISTIC			= 3,
	BIASED_DOWNWEIGHTING_HEURISTIC	= 4
};

/// a runtime selector
///
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float mis_heuristic(const float p1, const float p2, const MisHeuristicType selector)
{
	if (selector == BALANCE_HEURISTIC)
		return balance_heuristic(p1, p2);
	else if (selector == POWER_HEURISTIC)
		return power_heuristic(p1, p2);
	else if (selector == CUTOFF_HEURISTIC)
		return cutoff_heuristic(p1, p2);
	else if (selector == SOFT_CUTOFF_HEURISTIC)
		return soft_cutoff_heuristic(p1, p2);
	else
		return biased_downweighting_heuristic(p1, p2);
}

/// a compile-time selector
///
template <MisHeuristicType SELECTOR>
FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
float mis_heuristic(const float p1, const float p2)
{
	if (SELECTOR == BALANCE_HEURISTIC)
		return balance_heuristic(p1, p2);
	else if (SELECTOR == POWER_HEURISTIC)
		return power_heuristic(p1, p2);
	else if (SELECTOR == CUTOFF_HEURISTIC)
		return cutoff_heuristic(p1, p2);
	else if (SELECTOR == SOFT_CUTOFF_HEURISTIC)
		return soft_cutoff_heuristic(p1, p2);
	else
		return biased_downweighting_heuristic(p1, p2);
}