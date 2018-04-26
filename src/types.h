/*
 * Fermat
 *
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <cassert>

#ifdef __CUDACC__
#define FERMAT_HOST_DEVICE __host__ __device__
#define FERMAT_HOST   __host__
#define FERMAT_DEVICE __device__
#else
#define FERMAT_HOST_DEVICE 
#define FERMAT_HOST
#define FERMAT_DEVICE
#endif

#ifdef WIN32
#define FERMAT_FORCEINLINE __forceinline
#else
#define FERMAT_FORCEINLINE __inline__
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#define FERMAT_DEVICE_COMPILATION
#endif

#define ENABLE_ASSERTS

#if defined(ENABLE_ASSERTS)
#define FERMAT_ASSERT(x)	assert(x)
#else
#define FERMAT_ASSERT(x)
#endif

#define FERMAT_CUDA_TIMING

#if defined(FERMAT_CUDA_TIMING)
	#define FERMAT_CUDA_TIME(x)		x
#else
	#define FERMAT_CUDA_TIME(x)
#endif

#define CUDA_CHECKS

#if defined(CUDA_CHECKS)
	#define CUDA_CHECK(x)		x
#else
	#define CUDA_CHECK(x)
#endif

#define FERMAT_ALMOST_ONE_AS_INT	0x3F7FFFFFu

#if 0
typedef unsigned long long  uint64_t;
typedef unsigned int		uint32_t;
typedef unsigned short		uint16_t;
typedef unsigned char		uint8_t;
typedef long long			int64_t;
typedef int					int32_t;
typedef short				int16_t;
typedef char				int8_t;
#endif

typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t  uint8;
typedef int64_t int64;
typedef int32_t int32;
typedef int16_t int16;
typedef int8_t  int8;