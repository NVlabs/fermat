/*
 * Fermat
 *
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <tiled_sampling.h>
#include <buffers.h>

///@addtogroup Fermat
///@{

///@addtogroup SamplingModule
///@{

/// A tiled sample sequence view.
///
///\par
/// Basically, represents a tiled 2d grid of N-dimensional sample sequences, which can be conveniently
/// "sliced" in 1d or 2d sections (each such slice representing a 2d grid of 1d or 2d samples).
/// The per-pixel sequences are constructed by "offsetting" a base, common sequence using a set of per-pixel
/// N-dimensional shifts. The offsetting process is a regular Cranley-Patterson rotation, i.e. the <i>d</i>-th
/// coordinate of the final sequence for pixel <i>p</i> is given by:
///\n
///\code
///    sequence(p,d) = mod( base_sequence[d] + shift[p][d], 1 );
///\endcode
///
struct FERMAT_API TiledSequenceView
{
	FERMAT_HOST_DEVICE
	uint32 shift_index(const uint32 idx, const uint32 dim) const
	{
		return dim*tile_size*tile_size + (idx & (tile_size*tile_size - 1u));
	}

	FERMAT_HOST_DEVICE
	float sample(const uint32 idx, const uint32 dim) const
	{
		FERMAT_ASSERT(dim < n_dimensions);
		return samples[shift_index(idx, dim)];
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float shift(const uint32 idx, const uint32 dim) const
	{
		FERMAT_ASSERT(dim < n_dimensions);
		return shifts[shift_index(idx, dim)];
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float shift(const uint32 pixel_x, const uint32 pixel_y, const uint32 dim) const
	{
		FERMAT_ASSERT(dim < n_dimensions);
		const uint32 shift_x = pixel_x & (tile_size - 1);
		const uint32 shift_y = pixel_y & (tile_size - 1);
		const uint32 shift_i = shift_x + shift_y*tile_size;
		return shift(shift_i,dim);
	}

	FERMAT_HOST_DEVICE
	float sample_1d(const uint32 tile_idx, const uint32 idx, const uint32 dim) const
	{
		FERMAT_ASSERT(dim < n_dimensions);
		return fmodf(sample(idx, dim) + shift(tile_idx, dim), 1.0f);
	}

	FERMAT_HOST_DEVICE
	float sample_2d(const uint32 pixel_x, const uint32 pixel_y, const uint32 dim) const
	{
		FERMAT_ASSERT(dim < n_dimensions);
		const uint32 shift_x = pixel_x & (tile_size - 1);
		const uint32 shift_y = pixel_y & (tile_size - 1);
		const uint32 shift = shift_x + shift_y*tile_size;

		const uint32 tile_x = (pixel_x / tile_size) & (tile_size - 1);
		const uint32 tile_y = (pixel_y / tile_size) & (tile_size - 1);
		const uint32 tile = tile_x + tile_y*tile_size;

		return sample_1d(tile, shift, dim);
	}

	uint32		 n_dimensions;
	uint32		 tile_size;
	const float* samples;
	const float* shifts;
};

/// A tiled sample sequence.
///
///\par
/// Basically, represents a tiled 2d grid of N-dimensional sample sequences, which can be conveniently
/// "sliced" in 1d or 2d sections (each such slice representing a 2d grid of 1d or 2d samples).
/// The per-pixel sequences are constructed by "offsetting" a base, common sequence using a set of per-pixel
/// N-dimensional shifts. The offsetting process is a regular Cranley-Patterson rotation, i.e. the <i>d</i>-th
/// coordinate of the final sequence for pixel <i>p</i> is given by:
///\n
///\code
///    sequence(p,d) = mod( base_sequence[d] + shift[p][d], 1 );
///\endcode
/// The sequence has a further <i>time</i> dimension that can be used for progressive rendering, which can
/// be advanced calling the TiledSequence::set_instance() method.
///\par
/// Notice this is a host container for a device-side sequence.
/// Once it is set up, the device-side view can be queried using the TiledSequence::view() method.
///
struct FERMAT_API TiledSequence
{
	TiledSequence() {}

	void setup(const uint32 _n_dimensions, const uint32 _tile_size, const bool _random = false);

	void set_instance(const uint32 instance);

	TiledSequenceView view()
	{
		TiledSequenceView r;
		r.n_dimensions	= n_dimensions;
		r.tile_size		= tile_size;
		r.samples		= m_samples.ptr();
		//r.shifts		= m_shifts.ptr();
		r.shifts		= m_tile_shifts.ptr();
		return r;
	}

	uint32 n_dimensions;
	uint32 tile_size;

	DomainBuffer<CUDA_BUFFER, float>	m_shifts;
	DomainBuffer<CUDA_BUFFER, float>	m_tile_shifts;
	DomainBuffer<CUDA_BUFFER, float>	m_samples;
	DomainBuffer<CUDA_BUFFER, float>	m_sequence;
};

///@} SamplingModule
///@} Fermat
