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

/*! \file lfsr.h
 *   \brief Defines several random samplers
 *
 * This module provides linear-feedback shift register random samplers.
 */

#pragma once

#include <cugar/basic/types.h>


namespace cugar {

///@addtogroup sampling Sampling
///@{

///@defgroup LFSRModule Linear-Feedback Shift Register Generators
/// An implementation of the small linear feedback shift registers (LFSR) for
/// use in a Markov chain quasi-Monte Carlo context, as described in S. Chen,
/// M. Matsumoto, T. Nishimura, and A. Owen: "New inputs and methods for Markov
/// chain quasi-Monte Carlo", and S. Chen: "Consistence and convergence rate of
/// Markov chain quasi-Monte Carlo with examples".
/// Apart from the matrix that describes the recurrence, this class is stateless,
/// and can therefore be easily shared in a multi-threaded context.
///
/// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
///@{

/// An implementation of the small linear feedback shift registers (LFSR) for
/// use in a Markov chain quasi-Monte Carlo context, as described in S. Chen,
/// M. Matsumoto, T. Nishimura, and A. Owen: "New inputs and methods for Markov
/// chain quasi-Monte Carlo", and S. Chen: "Consistence and convergence rate of
/// Markov chain quasi-Monte Carlo with examples".
/// Apart from the matrix that describes the recurrence, this class is stateless,
/// and can therefore be easily shared in a multi-threaded context.
///
/// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
///
class LFSRGeneratorMatrix // Markov chain quasi-Monte Carlo linear feedback shift register.
{
public:
	enum Offset_type
	{
		ORIGINAL, // The original ones from the paper.
		GOOD_PROJECTIONS // Maximized minimum distance for some projections.
	};

	// Construct a small LFSR with period 2^m - 1, for 3 <= m <= 32.
	// The offset_type describes which set of offset values to use.
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	LFSRGeneratorMatrix() {}

	// Construct a small LFSR with period 2^m - 1, for 3 <= m <= 32.
	// The offset_type describes which set of offset values to use.
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	explicit LFSRGeneratorMatrix(uint32 m, Offset_type offset_type = GOOD_PROJECTIONS);

	// Update the given state, and return the next value in the sequence,
	// using the scramble value as a random bit shift. For the first invocation
	// an arbitrary state value 0 < *state < 2^m may be used. The scramble parameter
	// may be an arbitrary (possibly random) value. To generate the scrambled
	// coordinates of the origin it is valid to pass *state == 0, but in this case
	// the state will not be updated.
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	float next(uint32 scramble, uint32* state) const;

public:
	uint32 m_m; // period 2^m - 1
	uint32 m_f[32]; // f_d^s
};

/// An implementation of the small linear feedback shift registers (LFSR) for
/// use in a Markov chain quasi-Monte Carlo context, as described in S. Chen,
/// M. Matsumoto, T. Nishimura, and A. Owen: "New inputs and methods for Markov
/// chain quasi-Monte Carlo", and S. Chen: "Consistence and convergence rate of
/// Markov chain quasi-Monte Carlo with examples".
/// Apart from the matrix that describes the recurrence, this class is stateless,
/// and can therefore be easily shared in a multi-threaded context.
///
/// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
///
struct LFSRRandomStream
{
	// Set the initial state, using the scramble value as a random bit shift. For the first invocation
	// an arbitrary state value 0 < *state < 2^m may be used. The scramble parameter
	// may be an arbitrary (possibly random) value. To generate the scrambled
	// coordinates of the origin it is valid to pass *state == 0, but in this case
	// the state will not be updated.
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	LFSRRandomStream(LFSRGeneratorMatrix* matrix, const uint32 state, const uint32 scramble = 1361u) :
		m_matrix(matrix), m_state(state ? state : 0xFFFFFFFF), m_scramble(scramble) {}

	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	float next();

	LFSRGeneratorMatrix* m_matrix;
	uint32               m_state;
	uint32               m_scramble;
};

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
LFSRGeneratorMatrix::LFSRGeneratorMatrix(const uint32 m, const LFSRGeneratorMatrix::Offset_type offset_type)
	: m_m(m)
{
	// Table taken from T. Hansen and G. Mullen:
	// "Primitive Polynomials over Finite Fields".
	// It is implied that the coefficient for t^m is 1.
	const uint32 primitive_polynomials[32 - 3 + 1] =
	{
		(1 << 1) | 1,                       // 3
		(1 << 1) | 1,                       // 4
		(1 << 2) | 1,                       // 5
		(1 << 1) | 1,                       // 6
		(1 << 1) | 1,                       // 7
		(1 << 4) | (1 << 3) | (1 << 2) | 1, // 8
		(1 << 4) | 1,                       // 9
		(1 << 3) | 1,                       // 10
		(1 << 2) | 1,                       // 11
		(1 << 6) | (1 << 4) | (1 << 1) | 1, // 12
		(1 << 4) | (1 << 3) | (1 << 1) | 1, // 13
		(1 << 5) | (1 << 3) | (1 << 1) | 1, // 14
		(1 << 1) | 1,                       // 15
		(1 << 5) | (1 << 3) | (1 << 2) | 1, // 16
		(1 << 3) | 1,                       // 17
		(1 << 7) | 1,                       // 18
		(1 << 5) | (1 << 2) | (1 << 1) | 1, // 19
		(1 << 3) | 1,                       // 20
		(1 << 2) | 1,                       // 21
		(1 << 1) | 1,                       // 22
		(1 << 5) | 1,                       // 23
		(1 << 4) | (1 << 3) | (1 << 1) | 1, // 24
		(1 << 3) | 1,                       // 25
		(1 << 6) | (1 << 2) | (1 << 1) | 1, // 26
		(1 << 5) | (1 << 2) | (1 << 1) | 1, // 27
		(1 << 3) | 1,                       // 28
		(1 << 2) | 1,                       // 29
		(1 << 6) | (1 << 4) | (1 << 1) | 1, // 30
		(1 << 3) | 1,                       // 31
		(1 << 7) | (1 << 6) | (1 << 2) | 1  // 32
	};

	// The original offsets 10 <= m <= 32 are taken from S. Chen, M. Matsumoto, T. Nishimura,
	// and A. Owen: "New inputs and methods for Markov chain quasi-Monte Carlo".
	// The alternative set of offsets for 3 <= m <= 16 was computed by Leonhard Gruenschloss.
	// They should also yield maximal equidistribution as described in P. L'Ecuyer: "Maximally
	// Equidistributed Combined Tausworthe Generators", but their projections might have better
	// maximized minimum distance properties.
	const uint32 offsets[32 - 3 + 1][2] =
	{
		// org / good proj.
		{ 1,    1 }, // 3
		{ 2,    2 }, // 4
		{ 15,   15 }, // 5
		{ 8,    8 }, // 6
		{ 4,    4 }, // 7
		{ 41,   41 }, // 8
		{ 113,  113 }, // 9
		{ 115,  226 }, // 10 *
		{ 291,  520 }, // 11 *
		{ 172,  1583 }, // 12 *
		{ 267,  2242 }, // 13 *
		{ 332,  2312 }, // 14 *
		{ 388,  38 }, // 15 *
		{ 283,  13981 }, // 16 *
		{ 514,  514 }, // 17
		{ 698,  698 }, // 18
		{ 706,  706 }, // 19
		{ 1304, 1304 }, // 20
		{ 920,  920 }, // 21
		{ 1336, 1336 }, // 22
		{ 1236, 1236 }, // 23
		{ 1511, 1511 }, // 24
		{ 1445, 1445 }, // 25
		{ 1906, 1906 }, // 26
		{ 1875, 1875 }, // 27
		{ 2573, 2573 }, // 28
		{ 2633, 2633 }, // 29
		{ 2423, 2423 }, // 30
		{ 3573, 3573 }, // 31
		{ 3632, 3632 }  // 32
	};

	// Construct the matrix that corresponds to a single transition.
	uint32 matrix[32];
	matrix[m - 1] = 0;
	for (uint32 i = 1, pp = primitive_polynomials[m - 3]; i < m; ++i, pp >>= 1)
	{
		matrix[m - 1] |= (pp & 1) << (m - i); // Reverse bits.
		matrix[i - 1] = 1 << (m - i - 1);
	}

	// Apply the matrix exponentiation according to the offset.
	uint32 result0[32], result1[32]; // Storage for temporary results.
	for (unsigned i = 0; i < m; ++i)
		result0[i] = matrix[i]; // Copy over row.
	uint32* in = result0;
	uint32* out = result1;
	const uint32 offset = offsets[m - 3][static_cast<int>(offset_type)];
	for (unsigned i = 1; i < offset; ++i)
	{
		// Perform matrix multiplication: out = in * matrix.
		for (uint32 y = 0; y < m; ++y)
		{
			out[y] = 0;
			for (uint32 x = 0; x < m; ++x)
				for (uint32 i = 0; i < m; ++i)
					out[y] ^= (((in[y] >> i) & (matrix[m - i - 1] >> x)) & 1) << x;
		}

		// Swap input and output.
		unsigned* tmp = in;
		in = out;
		out = tmp;
	}

	// Transpose the result for simpler multiplication.
	for (uint32 y = 0; y < m; ++y)
	{
		m_f[y] = 0;
		for (uint32 x = 0; x < m; ++x)
			m_f[y] |= ((in[x] >> y) & 1) << (m - x - 1);
	}
}

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
float LFSRGeneratorMatrix::next(const uint32 scramble, uint32* const state) const
{
	//check(state);

	// Compute the matrix-vector multiplication using one matrix column at a time.
	uint32 result = 0;
	for (uint32 i = 0, s = *state; s; ++i, s >>= 1)
	{
		if (s & 1)
			result ^= m_f[i];
	}

#if !defined(FLT_EPSILON)
	const float FLT_EPSILON = 1.0e-10f;
#endif

	//check(result <= ~(1u << m_m));
	*state = result; // Write result back.
	result = (result << (32 - m_m)) ^ scramble; // Apply scrambling.
	const float fresult = result * (1.f / float(uint64(1ULL) << 32)); // Map to [0, 1).
	return fresult <= 1.0f - FLT_EPSILON ?
		fresult : 1.0f - FLT_EPSILON;
}

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
float LFSRRandomStream::next()
{
	return m_matrix->next(m_scramble, &m_state);
}

///@}
///@}

} // namespace cugar
