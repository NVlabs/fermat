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

#include <tiled_sequence.h>
#include <random_sequence.h>
#include <cugar/basic/cuda/arch.h>
#include <cugar/basic/numbers.h>
#include <cugar/sampling/latin_hypercube.h>
#include <cugar/sampling/multijitter.h>

#define N_SAMPLES (16*1024)

namespace detail {

__global__
void setup_samples_kernel(const uint32 instance, const uint32 X, const uint32 Y, const uint32 Z, const float* shifts, const float* sequence, float* samples)
{
	const uint32 pixel = threadIdx.x + blockIdx.x * blockDim.x;

	if (pixel < X*Y)
	{
		for (uint32 i = 0; i < Z; ++i)
		{
			const float s = sequence[i];

			samples[pixel + i*X*Y] = fmodf(s + shifts[pixel + i*X*Y], 1.0f);
			//samples[pixel + i*X*Y] = fmodf(s + shifts[pixel + (i % 3)*X*Y], 1.0f);	// use the same shifts for all layers
			//samples[pixel + i*X*Y] = s;												// do not shift the sequence
		}
	}
}

void setup_samples(const uint32 instance, const uint32 X, const uint32 Y, const uint32 Z, const float* shifts, const float* sequence, float* samples)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(X*Y, blockSize.x));
	setup_samples_kernel << < gridSize, blockSize >> > (instance, X, Y, Z, shifts, sequence, samples);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("setup samples") );
}

} // namespace detail

void TiledSequence::setup(const uint32 _n_dimensions, const uint32 _tile_size, const bool _random)
{
	n_dimensions = _n_dimensions;
	tile_size = _tile_size;

	// build the shifts
	if (_random)
	{
		// use random shifts
		m_shifts.alloc(tile_size * tile_size * n_dimensions);

		DeviceRandomSequence random_sequence;
		random_sequence.next(tile_size * tile_size * n_dimensions, m_shifts.ptr());
	}
	else
	{
		// build a 3d multi-jittered stack
		DomainBuffer<HOST_BUFFER, float> samples(tile_size * tile_size * n_dimensions);

		build_tiled_samples_3d(tile_size, tile_size, n_dimensions / 3, samples.ptr());

		// and replace as many layers as possible with blue-noise optimized samples
		load_samples("samples", tile_size, tile_size, n_dimensions / 3, samples.ptr());

		m_shifts = samples;
	}

	// allocate the samples buffer
	m_samples.alloc(tile_size * tile_size * n_dimensions);
	m_tile_shifts.alloc(tile_size * tile_size * n_dimensions);
	m_sequence.alloc(n_dimensions);

	// fill the tile shifts
	m_tile_shifts = m_shifts;

	m_sequence_samples.alloc(N_SAMPLES*n_dimensions);
}

void TiledSequence::set_instance(const uint32 instance)
{
	if ((instance % N_SAMPLES) == 0)
	{
		// setup the next batch of samples
		cugar::LHSampler sampler(instance);

		sampler.sample<false>(N_SAMPLES, n_dimensions, m_sequence_samples.ptr());
	}

	DomainBuffer<HOST_BUFFER, float> sequence(n_dimensions);

	for (uint32 d = 0; d < n_dimensions; ++d)
		sequence.ptr()[d] = m_sequence_samples[(instance % N_SAMPLES)*n_dimensions + d];

	m_sequence = sequence;

	detail::setup_samples(0, tile_size, tile_size, n_dimensions, m_shifts.ptr(), m_sequence.ptr(), m_samples.ptr());
}
