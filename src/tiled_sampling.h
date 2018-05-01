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
#include <algorithm>
#include <vector>

///@addtogroup Fermat
///@{

///@addtogroup SamplingModule
///@{

/// utility function for [0,1) randoms
///
inline float random()
{
	return rand() / float(RAND_MAX);
}

/// utility function for [0,N) integer randoms
///
inline uint32 irandom(const uint32 N)
{
	const float r = random();
	return min(uint32(r * N), N - 1);
}

struct strided_vec
{
	FERMAT_HOST_DEVICE strided_vec() {}
	FERMAT_HOST_DEVICE strided_vec(float* _ptr, const uint32 _off, const uint32 _stride) : base(_ptr), off(_off), stride(_stride) {}

	FERMAT_HOST_DEVICE float& operator[] (const uint32 i) { return base[off + i*stride]; }

	float*	base;
	uint32	off;
	uint32	stride;
};

//
// Represents a 3d lattice of 3d points.
// The samples are arranged in a SOA layout, so that the point (x,y,z).i coordinates are located at (x + y*X + z*X*Y*3 + i*X*Y)
// (i.e. if the set has Z slices of X*Y points, there will be Z*3 consecutive arrays of X*Y floats).
//
struct sample_set_3d
{
	sample_set_3d(const uint32 _x, const uint32 _y, float* _samples) :
		X(_x), Y(_y), samples(_samples) {}

	uint32 size() const { return X*Y; }

	strided_vec operator() (const uint32 x, const uint32 y, const uint32 z) { return strided_vec(samples, z*X*Y * 3 + X*y + x, X*Y); }

	uint32	X;
	uint32  Y;
	float*	samples;
};

///
/// build a 3d multi-jittering pattern
///
inline void mj_3d(uint32 X, uint32 Y, uint32 Z, sample_set_3d p, const uint32 slice)
{
#if 1
	for (uint32 k = 0; k < Z; ++k)
	{
		for (uint32 j = 0; j < Y; ++j)
		{
			for (uint32 i = 0; i < X; ++i)
			{
				p(i, j, k)[slice + 0] = (i + (j + (k + random()) / Z) / Y) / X;
				p(i, j, k)[slice + 1] = (j + (k + (i + random()) / X) / Z) / Y;
				p(i, j, k)[slice + 2] = (k + (i + (j + random()) / Y) / X) / Z;
			}
		}
	}

#if 1
	//
	// Exchange the points among Z slices
	//
	for (uint32 k = 0; k < Z; ++k)
	{
		for (uint32 j = 0; j < Y; ++j)
		{
			for (uint32 i = 0; i < X; ++i)
			{
				const uint32 r = k + irandom(Z - k);

				std::swap(p(i, j, k)[slice+0], p(i, j, r)[slice+0]);
				std::swap(p(i, j, k)[slice+1], p(i, j, r)[slice+1]);
				std::swap(p(i, j, k)[slice+2], p(i, j, r)[slice+2]);
			}
		}
	}
	//
	// Process Z slices
	//
	for (uint32 k = 0; k < Z; ++k)
	{
		//
		// Exchange the X components among Y columns (Z/Y/X)
		//
		for (uint32 j = 0; j < Y; ++j)
		{
			const uint32 r = j + irandom(Y - j); // use correlated sampling

			for (uint32 i = 0; i < X; ++i) // x is the fastest variable
			{
				//const uint32 r = j + irandom(Y - j);
				std::swap(p(i, j, k)[slice],
						  p(i, r, k)[slice]);
			}
		}

		//
		// Exchange the Y components among X rows (Z/Y/X)
		//
		for (uint32 i = 0; i < X; ++i)
		{
			const uint32 r = i + irandom(X - i); // use correlated sampling

			for (uint32 j = 0; j < Y; ++j) // y is the fastest variable
			{
				//const uint r = i + irandom(X - i);
				std::swap(p(i, j, k)[slice+1],
						  p(r, j, k)[slice+1]);
			}
		}
	}
#else
	//
	// Exchange the X components among Z slices (Z/Y/X)
	//
	for (uint32 k = 0; k < Z; ++k)
	{
		const uint32 r = k + irandom(Z - k);

		for (uint32 j = 0; j < Y; ++j)
		{
			//const uint32 r = k + irandom(Z - k);

			for (uint32 i = 0; i < X; ++i) // x is the fastest variable
			{
				//const uint32 r = k + irandom(Z - k);
				std::swap(p(i, j, k)[slice],
						  p(i, j, r)[slice]);
			}
		}
	}
	//
	// Exchange the Y components among X rows (X/Z/Y)
	//
	for (uint32 i = 0; i < X; ++i)
	{
		const uint32 r = i + irandom(X - i);

		for (uint32 k = 0; k < Z; ++k)
		{
			//const uint32 r = i + irandom(X - i);

			for (uint32 j = 0; j < Y; ++j) // y is the fastest variable
			{
				//const uint r = i + irandom(X - i);
				std::swap(p(i, j, k)[slice + 1],
						  p(r, j, k)[slice + 1]);
			}
		}
	}
	//
	// Exchange the Z components among Y columns (Y/X/Z)
	//
	for (uint32 j = 0; j < Y; ++j)
	{
		const uint32 r = j + irandom(Y - j);

		for (uint32 i = 0; i < X; ++i)
		{
			//const uint32 r = j + irandom(Y - j);

			for (uint32 k = 0; k < Z; ++k) // z is the fastest variable
			{
				//const uint32 r = j + irandom(Y - j);
				std::swap(p(i, j, k)[slice + 2],
						  p(i, r, k)[slice + 2]);
			}
		}
	}
#endif

#else
	//
	// Build Z independent 2d-multi-jittered grids of (x,y) points, together with some randomly shuffled z components
	//
	for (uint32 k = 0; k < Z; ++k)
	{
		for (uint32 j = 0; j < Y; ++j)
		{
			for (uint32 i = 0; i < X; ++i)
			{
				p(i, j, k)[slice + 0] = (i + (j + random()) / Y) / X;
				p(i, j, k)[slice + 1] = (j + (i + random()) / X) / Y;
				p(i, j, k)[slice + 2] = (i + (j + random()) / Y) / X;
			}
		}
	}
	//
	// Process Z slices
	//
	for (uint32 k = 0; k < Z; ++k)
	{
		//
		// Exchange the X components among Y columns (Z/Y/X)
		//
		for (uint32 j = 0; j < Y; ++j)
		{
			const uint32 r = j + irandom(Y - j); // use correlated sampling

			for (uint32 i = 0; i < X; ++i) // x is the fastest variable
			{
				//const uint32 r = j + irandom(Y - j);
				std::swap(p(i, j, k)[slice],
						  p(i, r, k)[slice]);
			}
		}
		//
		// Exchange the Y components among X rows (Z/Y/X)
		//
		for (uint32 i = 0; i < X; ++i)
		{
			const uint r = i + irandom(X - i); // use correlated sampling

			for (uint32 j = 0; j < Y; ++j) // y is the fastest variable
			{
				//const uint r = i + irandom(X - i);
				std::swap(p(i, j, k)[slice + 1],
						  p(r, j, k)[slice + 1]);
			}
		}
		// Exchange the Z components randomly
		for (uint32 n = 0; n < X*Y; ++n)
		{
			const uint r = n + irandom(X*Y - n);

			std::swap(p(n % X, n / X, k)[slice + 2],
					  p(r % X, r / X, k)[slice + 2]);
		}
	}
#endif
}

///
/// Build a 3d lattice of 3d points with low discrepancy.
/// The samples are arranged in a SOA layout, so that the point (x,y,z).i coordinates are located at (x + y*X + z*X*Y*3 + i*X*Y)
/// (i.e. if the set has Z slices of X*Y points, there will be Z*3 consecutive arrays of X*Y floats).
///
inline void build_tiled_samples_3d(const uint32 X, const uint32 Y, const uint32 Z, float* samples)
{
	// build the sample set
	sample_set_3d set( X, Y, samples );

	mj_3d( X, Y, Z, set, 0);

	// randomize the order of the samples within each 3d slice
	const uint32 SLICE_SIZE = X*Y;

	for (uint32 z = 0; z < Z; ++z)
	{
		for (uint32 i = 0; i < SLICE_SIZE; ++i)
		{
			const uint32 r = i + irandom(SLICE_SIZE - i);

			std::swap(samples[z * SLICE_SIZE * 3 + i + 0 * SLICE_SIZE], samples[z * SLICE_SIZE * 3 + r + 0 * SLICE_SIZE]);
			std::swap(samples[z * SLICE_SIZE * 3 + i + 1 * SLICE_SIZE], samples[z * SLICE_SIZE * 3 + r + 1 * SLICE_SIZE]);
			std::swap(samples[z * SLICE_SIZE * 3 + i + 2 * SLICE_SIZE], samples[z * SLICE_SIZE * 3 + r + 2 * SLICE_SIZE]);
		}
	}
}

/// load a set of 3d samples from a file
///
inline void load_samples(const char* nameradix, const uint32 X, const uint32 Y, const uint32 Z, float* samples)
{
	for (uint32 z = 0; z < Z; ++z)
	{
		char slicename[256];
		sprintf(slicename, "%s-%u.dat", nameradix, z);

		FILE* file = fopen(slicename, "rb");
		if (file == NULL)
			break;

		std::vector<float3> slice(X*Y);
		if (fread(&slice[0], sizeof(float3), X*Y, file) != X*Y)
			break;
		fclose(file);

		// convert from AoS to SoA
		for (uint32 i = 0; i < X*Y; ++i)
		{
			samples[z * X*Y * 3 + i + 0 * X*Y] = slice[i].x;
			samples[z * X*Y * 3 + i + 1 * X*Y] = slice[i].y;
			samples[z * X*Y * 3 + i + 2 * X*Y] = slice[i].z;
		}
		fprintf(stderr, "  loaded sample slice %u (%u x %u)\n", z, X, Y);
	}
}

///@} SamplingModule
///@} Fermat
