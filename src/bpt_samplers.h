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

#include <tiled_sequence.h>
#include <cugar/sampling/distributions.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

///
/// Primary coordinate sampler for light subpaths, based on a TiledSequence sampler
///
struct TiledLightSubpathPrimaryCoords
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	TiledLightSubpathPrimaryCoords(const TiledSequenceView _sequence) : sequence(_sequence) {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float sample(const uint32 idx, const uint32 vertex, const uint32 dim) const { return sequence.sample(idx, vertex * 3 + dim); }

	const TiledSequenceView sequence;
};

///
/// Primary coordinate sampler for eye subpaths, based on a TiledSequence sampler.
/// This sampler assumes that there is a one to one correspondence between paths and pixels,
/// i.e. that the path index IS the pixel index.
///
struct PerPixelEyeSubpathPrimaryCoords
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	PerPixelEyeSubpathPrimaryCoords(const TiledSequenceView _sequence, const uint32 _res_x, const uint32 _res_y) :
		sequence(_sequence), res_x(_res_x), res_y(_res_y) {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float sample(const uint32 idx, const uint32 vertex, const uint32 dim) const
	{
		const uint32 pixel_x = idx % res_x;
		const uint32 pixel_y = idx / res_x;

		if (vertex == 1 && dim < 2)
		{
			// use an optimized sampling pattern to rotate a Halton sequence
			return dim == 0 ?
				(pixel_x + sequence.sample_2d(pixel_x, pixel_y, dim)) / float(res_x) :
				(pixel_y + sequence.sample_2d(pixel_x, pixel_y, dim)) / float(res_y);
		}
		else
			return sequence.sample_2d(pixel_x, pixel_y, (vertex - 1) * 6 + dim);
	}

	const TiledSequenceView sequence;
	const uint32 res_x;
	const uint32 res_y;
};

///
/// A perturbation sampler
///
struct PerturbedPrimaryCoords
{
	enum Type
	{
		Null				= 0x0,
		CauchyPerturbation	= 0x1,
		IndependentSample	= 0x2
	};

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	uint32 chain_coordinate_index(const uint32 idx, const uint32 dim) const
	{
		return dim*n_chains + idx;
	}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	PerturbedPrimaryCoords(
		const uint32	_n_chains,
		float*			_path_u,
		const uint32	_path_vertex_offset,
		float*			_mut_u,
		const uint32	_mut_vertex_offset,
		const Type		_type,
		const float     _radius = 0.01f) :
		path_u(_path_u + _path_vertex_offset * 3 * _n_chains),
		mut_u(_mut_u + _mut_vertex_offset * 3 * _n_chains),
		n_chains(_n_chains),
		type(_type),
		radius(_radius)
	{}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float sample(const uint32 idx, const uint32 vertex, const uint32 dim) const { return perturbed_u(idx, vertex * 3 + dim); }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float u_m(const uint32 chain_id, const uint32 dim) const { return mut_u[chain_coordinate_index(chain_id, dim)]; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float& u(const uint32 chain_id, const uint32 dim) { return path_u[chain_coordinate_index(chain_id, dim)]; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	const float& u(const uint32 chain_id, const uint32 dim) const { return path_u[chain_coordinate_index(chain_id, dim)]; }

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float perturbed_u(const uint32 chain_id, const uint32 dim) const
	{
		if (type == CauchyPerturbation)
		{
			cugar::Cauchy_distribution cauchy(radius);

			return cugar::mod(u(chain_id, dim) + cauchy.map(u_m(chain_id, dim)), 1.0f);
		}
		else if (type == IndependentSample)
		{
			return u_m(chain_id, dim);
		}
		else
			return u(chain_id, dim);
	}

	float*	path_u;
	float*	mut_u;
	uint32	n_chains;
	Type	type;
	float   radius;
};

///@} BPTLib
///@} Fermat
