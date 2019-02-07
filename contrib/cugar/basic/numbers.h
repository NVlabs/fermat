/*
 * CUGAR : Cuda Graphics Accelerator
 *
 * Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <limits>
#include <cugar/basic/types.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_fp16.h>

namespace cugar {

#if defined(WIN32)
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#ifndef M_TWO_PI
#define M_TWO_PI 6.28318530717958647693
#endif

#ifndef M_TWO_PIf
#define M_TWO_PIf 6.28318530717958647693f
#endif

CUGAR_HOST_DEVICE inline bool is_finite(const double x)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
	return isfinite(x) != 0;
  #else
	return _finite(x) != 0;
  #endif
}
CUGAR_HOST_DEVICE
inline bool is_nan(const double x)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
	return isnan(x) != 0;
  #else
	return _isnan(x) != 0;
  #endif
}
CUGAR_HOST_DEVICE inline bool is_finite(const float x)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
	return isfinite(x) != 0;
  #else
	return _finite(x) != 0;
  #endif
}
CUGAR_HOST_DEVICE inline bool is_nan(const float x)
{
  #if defined(CUGAR_DEVICE_COMPILATION)
	return isnan(x) != 0;
  #else
	return _isnan(x) != 0;
  #endif
}

#else

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_TWO_PI
#define M_TWO_PI 6.28318530717958647693
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#ifndef M_TWO_PIf
#define M_TWO_PIf 6.28318530717958647693f
#endif

#endif

#ifdef __CUDACC__

CUGAR_FORCEINLINE __device__ uint32 warp_tid() { return threadIdx.x & 31; }
CUGAR_FORCEINLINE __device__ uint32 warp_id()  { return threadIdx.x >> 5; }

#endif

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
float float_infinity() { return cugar::binary_cast<float>(0x7f800000u); }

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
double double_infinity() { return cugar::binary_cast<double>(0x7ff0000000000000ULL ); }

///\page utilities_page Utilities
///
/// CUGAR contains various convenience functions and functors needed for every day's work:
///
/// - \ref BasicUtils
/// - \ref BasicFunctors
/// - \ref BasicMetaFunctions
///

///@addtogroup Basic
///@{

///\defgroup BasicUtils Utilities
///
/// CUGAR's convenience functions and functors needed for every day's work...
///

/// \ingroup BasicUtils
/// return the bitmask with the lo N bits set
///
template <uint32 N>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 lo_bits() { return (1u << N) - 1u; }

/// \ingroup BasicUtils
/// return the bitmask with the hi N bits set
///
template <uint32 N>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 hi_bits() { return ~lo_bits<N>(); }

/// \ingroup BasicUtils
/// count the number of occurrences of a given value inside an array, up to a maximum value
///
template <typename Iterator, typename T>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 count_occurrences(const Iterator begin, uint32 size, const T val, const uint32 max_occ = uint32(-1))
{
    uint32 occ = 0u;
    for (uint32 i = 0; i < size; ++i)
    {
        if (begin[i] == val)
        {
            if (++occ >= max_occ)
                return occ;
        }
    }
    return occ;
}

/// \ingroup BasicUtils
/// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
///
template<typename L, typename R>
inline CUGAR_HOST_DEVICE L divide_ri(const L x, const R y)
{
    return L( (x + (y - 1)) / y );
}

/// \ingroup BasicUtils
/// x/y rounding towards zero for integers, used to determine # of blocks/warps etc.
///
template<typename L, typename R>
inline CUGAR_HOST_DEVICE L divide_rz(const L x, const R y)
{
    return L( x / y );
}

/// \ingroup BasicUtils
/// round x towards infinity to the next multiple of y
///
template<typename L, typename R>
inline CUGAR_HOST_DEVICE L round_i(const L x, const R y){ return L( y * divide_ri(x, y) ); }

/// \ingroup BasicUtils
/// round x towards zero to the next multiple of y
///
template<typename L, typename R>
inline CUGAR_HOST_DEVICE L round_z(const L x, const R y){ return L( y * divide_rz(x, y) ); }

/// \ingroup BasicUtils
/// round x towards to the closest multiple of x
///
template<typename L, typename R>
inline CUGAR_HOST_DEVICE L round(const L x, const R y)
{
    const L r = round_z( x, y );
    return R((x - r)*2) > y ? r+L(1) : r;
}

/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint8 comp(const uchar2 a, const char c)
{
    return (c == 0 ? a.x : a.y);
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE char comp(const char2 a, const char c)
{
    return (c == 0 ? a.x : a.y);
}

/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint8 comp(const uchar4 a, const char c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE char comp(const char4 a, const char c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}

/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 comp(const uint2 a, const uint32 c)
{
    return (c == 0 ? a.x : a.y);
}
/// set the c'th component of a
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE void set(uint2& a, const uint32 c, const uint32 v)
{
    if (c == 0) a.x = v;
    else        a.y = v;
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint64 comp(const ulonglong2 a, const uint32 c)
{
    return (c == 0 ? a.x : a.y);
}
/// set the c'th component of a
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE void set(ulonglong2& a, const uint32 c, const uint64 v)
{
    if (c == 0) a.x = v;
    else        a.y = v;
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE  int32 comp(const  int2 a, const uint32 c)
{
    return (c == 0 ? a.x : a.y);
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 comp(const uint4 a, const uint32 c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}
/// set the c'th component of a
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE void set(uint4& a, const uint32 c, const uint32 v)
{
    if (c == 0)         a.x = v;
    else if (c == 1)    a.y = v;
    else if (c == 2)    a.z = v;
    else                a.w = v;
}
/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE int32 comp(const int4 a, const uint32 c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}

/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint16 comp(const ushort4 a, const uint32 c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}

/// return the c'th component of a by value
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint64 comp(const ulonglong4 a, const uint32 c)
{
    return c <= 1 ?
        (c == 0 ? a.x : a.y) :
        (c == 2 ? a.z : a.w);
}
/// set the c'th component of a
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE void set(ulonglong4& a, const uint32 c, const uint64 v)
{
    if (c == 0)         a.x = v;
    else if (c == 1)    a.y = v;
    else if (c == 2)    a.z = v;
    else                a.w = v;
}

///@defgroup VectorTypesModule Vector Types
/// this module defines POD vector types as well as some reflection meta-functions
///@{

typedef uchar2 uint8_2;
typedef uchar3 uint8_3;
typedef uchar4 uint8_4;

typedef  char2  int8_2;
typedef  char3  int8_3;
typedef  char4  int8_4;

typedef ushort2 uint16_2;
typedef ushort3 uint16_3;
typedef ushort4 uint16_4;

typedef  short2  int16_2;
typedef  short3  int16_3;
typedef  short4  int16_4;

typedef uint2 uint32_2;
typedef uint3 uint32_3;
typedef uint4 uint32_4;

typedef  int2  int32_2;
typedef  int3  int32_3;
typedef  int4  int32_4;

typedef ulonglong2 uint64_2;
typedef ulonglong3 uint64_3;
typedef ulonglong4 uint64_4;

typedef  longlong2  int64_2;
typedef  longlong3  int64_3;
typedef  longlong4  int64_4;

template <typename T, uint32 DIM>
struct vector_type {};

template <typename T>
struct vector1_storage
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector1_storage() {}
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector1_storage(T _x) : x(_x) {}

	T x;
};
template <typename T>
struct vector2_storage
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector2_storage() {}
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector2_storage(T _x, T _y) : x(_x), y(_y) {}

	T x, y;
};
template <typename T>
struct vector3_storage
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector3_storage() {}
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector3_storage(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

	T x, y, z;
};
template <typename T>
struct vector4_storage
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector4_storage() {}
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE vector4_storage(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}

	T x, y, z, w;
};

template <typename T> struct vector_type<T,1> { typedef vector1_storage<T> type; static type make(const T i1) { return type(i1); } };
template <typename T> struct vector_type<T,2> { typedef vector2_storage<T> type; static type make(const T i1, const T i2) { return type(i1,i2); } };
template <typename T> struct vector_type<T,3> { typedef vector3_storage<T> type; static type make(const T i1, const T i2, const T i3) { return type(i1,i2,i3); } };
template <typename T> struct vector_type<T,4> { typedef vector4_storage<T> type; static type make(const T i1, const T i2, const T i3, const T i4) { return type(i1,i2,i3,4); } };

template <> struct vector_type<char,1> { typedef char  type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const char i1)                { return i1; } };
template <> struct vector_type<char,2> { typedef char2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const char i1, const char i2) { return make_char2(i1,i2); } };
template <> struct vector_type<char,3> { typedef char3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const char i1, const char i2, const char i3) { return make_char3(i1,i2,i3); }  };
template <> struct vector_type<char,4> { typedef char4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const char i1, const char i2, const char i3, const char i4) { return make_char4(i1,i2,i3,i4); }  };

template <> struct vector_type<unsigned char,1> { typedef unsigned char type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned char i1)                  { return i1; } };
template <> struct vector_type<unsigned char,2> { typedef uchar2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned char i1, const unsigned char i2) { return make_uchar2(i1,i2); } };
template <> struct vector_type<unsigned char,3> { typedef uchar3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned char i1, const unsigned char i2, const unsigned char i3) { return make_uchar3(i1,i2,i3); }  };
template <> struct vector_type<unsigned char,4> { typedef uchar4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned char i1, const unsigned char i2, const unsigned char i3, const unsigned char i4) { return make_uchar4(i1,i2,i3,i4); }  };

template <> struct vector_type<short,1> { typedef short  type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const short i1)                 { return i1; } };
template <> struct vector_type<short,2> { typedef short2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const short i1, const short i2) { return make_short2(i1,i2); } };
template <> struct vector_type<short,3> { typedef short3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const short i1, const short i2, const short i3) { return make_short3(i1,i2,i3); }  };
template <> struct vector_type<short,4> { typedef short4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const short i1, const short i2, const short i3, const short i4) { return make_short4(i1,i2,i3,i4); }  };

template <> struct vector_type<unsigned short,1> { typedef unsigned short type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned short i1)                   { return i1; } };
template <> struct vector_type<unsigned short,2> { typedef ushort2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned short i1, const unsigned short i2) { return make_ushort2(i1,i2); } };
template <> struct vector_type<unsigned short,3> { typedef ushort3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned short i1, const unsigned short i2, const unsigned short i3) { return make_ushort3(i1,i2,i3); }  };
template <> struct vector_type<unsigned short,4> { typedef ushort4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned short i1, const unsigned short i2, const unsigned short i3, const unsigned short i4) { return make_ushort4(i1,i2,i3,i4); }  };

template <> struct vector_type<int,1> { typedef int  type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int i1)               { return i1; } };
template <> struct vector_type<int,2> { typedef int2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int i1, const int i2) { return make_int2(i1,i2); } };
template <> struct vector_type<int,3> { typedef int3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int i1, const int i2, const int i3) { return make_int3(i1,i2,i3); } };
template <> struct vector_type<int,4> { typedef int4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int i1, const int i2, const int i3, const int i4) { return make_int4(i1,i2,i3,i4); } };

template <> struct vector_type<unsigned int,1> { typedef unsigned int type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned int i1)                 { return i1; } };
template <> struct vector_type<unsigned int,2> { typedef uint2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned int i1, const unsigned int i2) { return make_uint2(i1,i2); } };
template <> struct vector_type<unsigned int,3> { typedef uint3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned int i1, const unsigned int i2, const unsigned int i3) { return make_uint3(i1,i2,i3); } };
template <> struct vector_type<unsigned int,4> { typedef uint4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const unsigned int i1, const unsigned int i2, const unsigned int i3, const unsigned int i4) { return make_uint4(i1,i2,i3,i4); } };

template <> struct vector_type<int64,1>  { typedef  int64   type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int64 i1)                                                 { return i1; } };
template <> struct vector_type<int64,2>  { typedef  int64_2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int64 i1, const int64 i2)                                 {  int64_2 r; r.x = i1; r.y = i2; return r; } };
template <> struct vector_type<int64,3>  { typedef  int64_3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int64 i1, const int64 i2, const int64 i3)                 {  int64_3 r; r.x = i1; r.y = i2; r.z = i3; return r; } };
template <> struct vector_type<int64,4>  { typedef  int64_4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const int64 i1, const int64 i2, const int64 i3, const int64 i4) {  int64_4 r; r.x = i1; r.y = i2; r.z = i3, r.w = i4; return r; } };

template <> struct vector_type<uint64,1>  { typedef uint64   type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const uint64 i1)                                                    { return i1; } };
template <> struct vector_type<uint64,2>  { typedef uint64_2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const uint64 i1, const uint64 i2)                                   { uint64_2 r; r.x = i1; r.y = i2; return r; } };
template <> struct vector_type<uint64,3>  { typedef uint64_3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const uint64 i1, const uint64 i2, const uint64 i3)                  { uint64_3 r; r.x = i1; r.y = i2; r.z = i3; return r; } };
template <> struct vector_type<uint64,4>  { typedef uint64_4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const uint64 i1, const uint64 i2, const uint64 i3, const uint64 i4) { uint64_4 r; r.x = i1; r.y = i2; r.z = i3, r.w = i4; return r; } };

template <> struct vector_type<float,1> { typedef float  type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const float i1)                 { return i1; } };
template <> struct vector_type<float,2> { typedef float2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const float i1, const float i2) { return make_float2(i1,i2); } };
template <> struct vector_type<float,3> { typedef float3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const float i1, const float i2, const float i3) { return make_float3(i1,i2,i3); }  };
template <> struct vector_type<float,4> { typedef float4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const float i1, const float i2, const float i3, const float i4) { return make_float4(i1,i2,i3,i4); }  };

template <> struct vector_type<double,1> { typedef double  type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const double i1)                 { return i1; } };
template <> struct vector_type<double,2> { typedef double2 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const double i1, const double i2) { return make_double2(i1,i2); } };
template <> struct vector_type<double,3> { typedef double3 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const double i1, const double i2, const double i3) { return make_double3(i1,i2,i3); } };
template <> struct vector_type<double,4> { typedef double4 type; CUGAR_FORCEINLINE CUGAR_HOST_DEVICE static type make(const double i1, const double i2, const double i3, const double i4) { return make_double4(i1,i2,i3,i4); } };

template <typename T> CUGAR_FORCEINLINE CUGAR_HOST_DEVICE typename vector_type<T,1>::type make_vector(const T i1)                                     { return vector_type<T,1>::make( i1 ); }
template <typename T> CUGAR_FORCEINLINE CUGAR_HOST_DEVICE typename vector_type<T,2>::type make_vector(const T i1, const T i2)                         { return vector_type<T,2>::make( i1, i2 ); }
template <typename T> CUGAR_FORCEINLINE CUGAR_HOST_DEVICE typename vector_type<T,3>::type make_vector(const T i1, const T i2, const T i3)             { return vector_type<T,3>::make( i1, i2, i3 ); }
template <typename T> CUGAR_FORCEINLINE CUGAR_HOST_DEVICE typename vector_type<T,4>::type make_vector(const T i1, const T i2, const T i3, const T i4) { return vector_type<T,4>::make( i1, i2, i3, i4 ); }

template <typename T> struct vector_traits {};
template <>           struct vector_traits<char>           { typedef          char  value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<unsigned char>  { typedef unsigned char  value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<short>          { typedef          short value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<unsigned short> { typedef unsigned short value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<int>            { typedef          int   value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<unsigned int>   { typedef unsigned int   value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<int64>          { typedef          int64 value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<uint64>         { typedef         uint64 value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<float>          { typedef         float  value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<double>         { typedef         double value_type; const static uint32 DIM = 1; };
template <>           struct vector_traits<char2>   { typedef char value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<char3>   { typedef char value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<char4>   { typedef char value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<uchar2>  { typedef unsigned char value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<uchar3>  { typedef unsigned char value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<uchar4>  { typedef unsigned char value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<short2>  { typedef short value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<short3>  { typedef short value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<short4>  { typedef short value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<ushort2> { typedef unsigned short value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<ushort3> { typedef unsigned short value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<ushort4> { typedef unsigned short value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<int2>    { typedef int value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<int3>    { typedef int value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<int4>    { typedef int value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<uint2>   { typedef unsigned int value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<uint3>   { typedef unsigned int value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<uint4>   { typedef unsigned int value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<float2>  { typedef float value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<float3>  { typedef float value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<float4>  { typedef float value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<double2>  { typedef float value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<uint64_2> { typedef uint64 value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<uint64_3> { typedef uint64 value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<uint64_4> { typedef uint64 value_type; const static uint32 DIM = 4; };
template <>           struct vector_traits<int64_2>  { typedef int64  value_type; const static uint32 DIM = 2; };
template <>           struct vector_traits<int64_3>  { typedef int64  value_type; const static uint32 DIM = 3; };
template <>           struct vector_traits<int64_4>  { typedef int64  value_type; const static uint32 DIM = 4; };

///@} VectorTypesModule

/// sign function
///
template <typename T>
inline CUGAR_HOST_DEVICE T sgn(const T x) { return x > 0 ? T(1) : T(-1); }

/// round a floating point number
///
inline CUGAR_HOST_DEVICE float round(const float x)
{
	const int y = x > 0.0f ? int(x) : int(x)-1;
	return (x - float(y) > 0.5f) ? float(y)+1.0f : float(y);
}

/// absolute value
///
inline CUGAR_HOST_DEVICE int32 abs(const int32 a) { return a < 0 ? -a : a; }

/// absolute value
///
inline CUGAR_HOST_DEVICE int64 abs(const int64 a) { return a < 0 ? -a : a; }

/// absolute value
///
inline CUGAR_HOST_DEVICE float abs(const float a) { return fabsf(a); }

/// absolute value
///
inline CUGAR_HOST_DEVICE double abs(const double a) { return fabs(a); }

/// minimum of two floats
///
inline CUGAR_HOST_DEVICE float min(const float a, const float b) { return a < b ? a : b; }

/// maximum of two floats
///
inline CUGAR_HOST_DEVICE float max(const float a, const float b) { return a > b ? a : b; }

/// minimum of two int8
///
inline CUGAR_HOST_DEVICE int8 min(const int8 a, const int8 b) { return a < b ? a : b; }

/// maximum of two int8
///
inline CUGAR_HOST_DEVICE int8 max(const int8 a, const int8 b) { return a > b ? a : b; }

/// minimum of two uint8
///
inline CUGAR_HOST_DEVICE uint8 min(const uint8 a, const uint8 b) { return a < b ? a : b; }

/// maximum of two uint8
///
inline CUGAR_HOST_DEVICE uint8 max(const uint8 a, const uint8 b) { return a > b ? a : b; }

/// minimum of two uint16
///
inline CUGAR_HOST_DEVICE uint16 min(const uint16 a, const uint16 b) { return a < b ? a : b; }

/// maximum of two uint16
///
inline CUGAR_HOST_DEVICE uint16 max(const uint16 a, const uint16 b) { return a > b ? a : b; }

/// minimum of two int32
///
inline CUGAR_HOST_DEVICE int32 min(const int32 a, const int32 b) { return a < b ? a : b; }

/// maximum of two int32
///
inline CUGAR_HOST_DEVICE int32 max(const int32 a, const int32 b) { return a > b ? a : b; }

/// minimum of two uint32
///
inline CUGAR_HOST_DEVICE uint32 min(const uint32 a, const uint32 b) { return a < b ? a : b; }

/// maximum of two uint32
///
inline CUGAR_HOST_DEVICE uint32 max(const uint32 a, const uint32 b) { return a > b ? a : b; }

/// minimum of two int64
///
inline CUGAR_HOST_DEVICE int64 min(const int64 a, const int64 b) { return a < b ? a : b; }

/// maximum of two int64
///
inline CUGAR_HOST_DEVICE int64 max(const int64 a, const int64 b) { return a > b ? a : b; }

/// minimum of two uint64
///
inline CUGAR_HOST_DEVICE uint64 min(const uint64 a, const uint64 b) { return a < b ? a : b; }

/// maximum of two uint64
///
inline CUGAR_HOST_DEVICE uint64 max(const uint64 a, const uint64 b) { return a > b ? a : b; }

/// quantize the float x in [0,1] to an integer [0,...,n[
///
inline CUGAR_HOST_DEVICE uint32 quantize(const float x, const uint32 n)
{
	return (uint32)max( min( int32( x * float(n) ), int32(n-1) ), int32(0) );
}
/// compute the floating point module of a quantity with sign
///
inline float CUGAR_HOST_DEVICE mod(const float x, const float m) { return x > 0.0f ? fmodf( x, m ) : m - fmodf( -x, m ); }

/// compute the floating point square of a quantity
///
inline float CUGAR_HOST_DEVICE sqr(const float x) { return x*x; }

/// compute the floating point square of a quantity
///
inline double CUGAR_HOST_DEVICE sqr(const double x) { return x*x; }

/// compute the log base 2 of an integer
///
inline CUGAR_HOST_DEVICE uint32 log2(uint32 n)
{
    unsigned int c = 0;
    if (n & 0xffff0000u) { n >>= 16; c |= 16; }
    if (n & 0xff00) { n >>= 8; c |= 8; }
    if (n & 0xf0) { n >>= 4; c |= 4; }
    if (n & 0xc) { n >>= 2; c |= 2; }
    if (n & 0x2) c |= 1;
    return c;
/*    uint32 m = 0;
    while (n > 0)
    {
        n >>= 1;
        m++;
    }
    return m-1;*/
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float saturate(const float x)
{
#ifdef CUGAR_DEVICE_COMPILATION
	return ::saturate(x);
#else
	return max( min( x, 1.0f ), 0.0f );
#endif
}

/// compute a simple 32-bit hash
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint32 hash(uint32 a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

/// Thomas Wang's 32 bit Mix Function: http://www.cris.com/~Ttwang/tech/inthash.htm
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint32 hash2(uint32 key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}

/// Thomas Wang's 64 bit Mix Function: http://www.cris.com/~Ttwang/tech/inthash.htm
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint64 hash(uint64 key)
{
    key += ~(key << 32);
    key ^=  (key >> 22);
    key += ~(key << 13);
    key ^=  (key >> 8);
    key +=  (key << 3);
    key ^=  (key >> 15);
    key += ~(key << 27);
    key ^=  (key >> 31);
    return key;
}

/// simple 64-bit hash
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint64 hash2(uint64 key)
{
    return (key >> 32) ^ key;
}

/// elf 64-bit hash
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint64 hash3(uint64 key)
{
    uint32 hash = 0u;

    #if defined(__CUDA_ARCH__)
    #pragma unroll
    #endif
    for (uint32 i = 0; i < 8; ++i)
    {
        hash = (hash << 4) + ((key >> (i*8)) & 255u); // shift/mix

        // get high nybble
        const uint32 hi_bits = hash & 0xF0000000;
        if (hi_bits != 0u)
            hash ^= hi_bits >> 24; // xor high nybble with second nybble

        hash &= ~hi_bits; // clear high nybble
    }
    return hash;
}

#define CUGAR_RAND_A 1664525
#define CUGAR_RAND_C 1013904223

/// A very simple Linear Congruential Generator
///
struct LCG_random
{
    static const uint32 MAX = 0xFFFFFFFF;

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE LCG_random(const uint32 s = 0) : m_s(s) {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE uint32 next() { m_s = m_s*CUGAR_RAND_A + CUGAR_RAND_C; return m_s; }

    uint32 m_s;
};

/// A very simple Linear Congruential Generator
///
struct FLCG_random : LCG_random
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE FLCG_random(const uint32 s = 0) : LCG_random(s) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE float next() { return this->LCG_random::next() / float(LCG_random::MAX); }
};

/// an indexed random number generator (see "Correlated Multi-Jittered Sampling", by Andrew Kensler)
///
/// \param i		the index in the random stream
/// \param p		the seed
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float randfloat(unsigned i, unsigned p)
{
	i ^= p;
	i ^= i >> 17;
	i ^= i >> 10; i *= 0xb36534e5;
	i ^= i >> 12;
	i ^= i >> 21; i *= 0x93fc4795;
	i ^= 0xdf6e307f;
	i ^= i >> 17; i *= 1 | p >> 18;
	return i * (1.0f / 4294967808.0f);
}

/// reverse the bits of an integer n
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 reverse_bits(const uint32 n)
{
	uint32 bits = n;
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return bits;
}

/// return the radical inverse of an integer n
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float radical_inverse(unsigned int n)
{
#if 0
	double  result = 0.0;
	unsigned int  remainder;
	unsigned int  m, bj = 1;

	const unsigned int b = 2u;

	do
	{
		bj *= b;
		m = n;
		n /= b;

		remainder = m - n * b;

		result += double(remainder) / double(bj);
	} while (n > 0);

	return float(result);
#else
	uint32 bits = n;
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
#endif
};

/// Van der Corput radical inverse in base 2 with 52 bits precision.
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint64 radical_inverse(uint64 bits, const uint64 scramble)
{
	bits = (bits << 32) | (bits >> 32);
	bits = ((bits & 0x0000ffff0000ffffULL) << 16) |
		   ((bits & 0xffff0000ffff0000ULL) >> 16);
	bits = ((bits & 0x00ff00ff00ff00ffULL) << 8) |
		   ((bits & 0xff00ff00ff00ff00ULL) >> 8);
	bits = ((bits & 0x0f0f0f0f0f0f0f0fULL) << 4) |
		   ((bits & 0xf0f0f0f0f0f0f0f0ULL) >> 4);
	bits = ((bits & 0x3333333333333333ULL) << 2) |
		   ((bits & 0xccccccccccccccccULL) >> 2);
	bits = ((bits & 0x5555555555555555ULL) << 1) |
		   ((bits & 0xaaaaaaaaaaaaaaaaULL) >> 1);
	return (scramble ^ bits) >> (64 - 52); // Account for 52 bits precision.
}

/// A pseudorandom permutation function (see "Correlated Multi-Jittered Sampling", by Andrew Kensler)
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 permute(uint32 i, uint32 l, uint32 p) 
{
	uint32 w = l - 1;
	w |= w >> 1;
	w |= w >> 2;
	w |= w >> 4;
	w |= w >> 8;
	w |= w >> 16;
	do {
		i ^= p;					i *= 0xe170893d;
		i ^= p			>> 16;
		i ^= (i & w)	>> 4;
		i ^= p			>> 8;	i *= 0x0929eb3f;
		i ^= p			>> 23;
		i ^= (i & w)	>> 1;	i *= 1 | p >> 27;
								i *= 0x6935fa69;
		i ^= (i & w)	>> 11;	i *= 0x74dcb303;
		i ^= (i & w)	>> 2;	i *= 0x9e501cc3;
		i ^= (i & w)	>> 2;	i *= 0xc860a3df;
		i &= w;
		i ^= i			>> 5;
	} while (i >= l);

	return (i + p) % l;
}

#if defined(__CUDA_ARCH__)

CUGAR_FORCEINLINE CUGAR_DEVICE 
uint8 min3(const uint8 op1, const uint8 op2, const uint8 op3)
{
    uint32 r;
    asm( "  vmin.u32.u32.u32.min %0, %1, %2, %3;"               : "=r"(r)                              : "r"(uint32(op1)), "r"(uint32(op2)), "r"(uint32(op3)) );
    return r;
}

CUGAR_FORCEINLINE CUGAR_DEVICE 
uint32 min3(const uint32 op1, const uint32 op2, const uint32 op3)
{
    uint32 r;
    asm( "  vmin.u32.u32.u32.min %0, %1, %2, %3;"               : "=r"(r)                              : "r"(op1), "r"(op2), "r"(op3) );
    return r;
}

CUGAR_FORCEINLINE CUGAR_DEVICE 
uint32 max3(const uint32 op1, const uint32 op2, const uint32 op3)
{
    uint32 r;
    asm( "  vmax.u32.u32.u32.max %0, %1, %2, %3;"               : "=r"(r)                              : "r"(op1), "r"(op2), "r"(op3) );
    return r;
}

CUGAR_FORCEINLINE CUGAR_DEVICE 
int32 min3(const int32 op1, const int32 op2, const int32 op3)
{
    uint32 r;
    asm( "  vmin.s32.s32.s32.min %0, %1, %2, %3;"               : "=r"(r)                              : "r"(op1), "r"(op2), "r"(op3) );
    return r;
}

CUGAR_FORCEINLINE CUGAR_DEVICE 
int32 max3(const int32 op1, const int32 op2, const int32 op3)
{
    uint32 r;
    asm( "  vmax.s32.s32.s32.max %0, %1, %2, %3;"               : "=r"(r)                              : "r"(op1), "r"(op2), "r"(op3) );
    return r;
}

#else

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint8 min3(const uint8 op1, const uint8 op2, const uint8 op3)
{
    return cugar::min( op1, cugar::min( op2, op3 ) );
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint32 min3(const uint32 op1, const uint32 op2, const uint32 op3)
{
    return cugar::min( op1, cugar::min( op2, op3 ) );
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
uint32 max3(const uint32 op1, const uint32 op2, const uint32 op3)
{
    return cugar::max( op1, cugar::max( op2, op3 ) );
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
int32 min3(const int32 op1, const int32 op2, const int32 op3)
{
    return cugar::min( op1, cugar::min( op2, op3 ) );
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
int32 max3(const int32 op1, const int32 op2, const int32 op3)
{
    return cugar::max( op1, cugar::max( op2, op3 ) );
}

#endif

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
float min3(const float op1, const float op2, const float op3)
{
    return cugar::min( op1, cugar::min( op2, op3 ) );
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE 
float max3(const float op1, const float op2, const float op3)
{
    return cugar::max( op1, cugar::max( op2, op3 ) );
}

#ifdef __CUDA_ARCH__

inline CUGAR_DEVICE float fast_pow(const float a, const float b)
{
    return __powf(a,b);
}
inline CUGAR_DEVICE float fast_sin(const float x)
{
    return __sinf(x);
}
inline CUGAR_DEVICE float fast_cos(const float x)
{
    return __cosf(x);
}
inline CUGAR_DEVICE float fast_sqrt(const float x)
{
    return __fsqrt_rn(x);
}

#else

inline CUGAR_HOST_DEVICE float fast_pow(const float a, const float b)
{
    return ::powf(a,b);
}
inline CUGAR_HOST_DEVICE float fast_sin(const float x)
{
    return sinf(x);
}
inline CUGAR_HOST_DEVICE float fast_cos(const float x)
{
    return cosf(x);
}
inline CUGAR_HOST_DEVICE float fast_sqrt(const float x)
{
    return sqrtf(x);
}

#endif

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
void sincosf(float phi, float* s, float* c)
{
#if defined(CUGAR_DEVIE_COMPILATION)
	::sincosf(phi, s, c);
#else
	*s = sinf(phi);
	*c = cosf(phi);
#endif
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float rsqrtf(float x)
{
#if defined(CUGAR_DEVIE_COMPILATION)
	return ::rsqrtf(x);
#else
	return 1.0f / sqrtf(x);
#endif
}

#ifdef __CUDACC__
inline CUGAR_DEVICE uint16 float_to_half(const float x) { return __float2half_rn(x); }
inline CUGAR_DEVICE float  half_to_float(const uint32 h) { return __half2float(h); }
#endif

///
/// A generic class to represent traits of numeric types T.
/// Unlike STL's numeric_traits, field_traits<T>::min() and field_traits<T>::max() are signed.
///
template <typename T>
struct field_traits
{
#ifdef __CUDACC__
    /// return the minimum value of T
    ///
	CUGAR_HOST_DEVICE static T min() { return T(); }

    /// return the maximum value of T
    ///
    CUGAR_HOST_DEVICE static T max() { return T(); }
#else
    /// return the minimum value of T
    ///
	static T min()
    {
        return std::numeric_limits<T>::is_integer ?
             std::numeric_limits<T>::min() :
            -std::numeric_limits<T>::max();
    }
    /// return the maximum value of T
    ///
	static T max() { return std::numeric_limits<T>::max(); }
#endif
};

/// int8 specialization of field_traits
///
template <>
struct field_traits<int8>
{
	CUGAR_HOST_DEVICE static int8 min() { return -128; }
    CUGAR_HOST_DEVICE static int8 max() { return  127; }
};
/// int16 specialization of field_traits
///
template <>
struct field_traits<int16>
{
	CUGAR_HOST_DEVICE static int16 min() { return -32768; }
    CUGAR_HOST_DEVICE static int16 max() { return  32767; }
};
/// int32 specialization of field_traits
///
template <>
struct field_traits<int32>
{
	CUGAR_HOST_DEVICE static int32 min() { return -(1 << 30); }
    CUGAR_HOST_DEVICE static int32 max() { return  (1 << 30); }
};
/// int64 specialization of field_traits
///
template <>
struct field_traits<int64>
{
	CUGAR_HOST_DEVICE static int64 min() { return -(int64(1) << 62); }
    CUGAR_HOST_DEVICE static int64 max() { return  (int64(1) << 62); }
};

#ifdef __CUDACC__
/// float specialization of field_traits
///
template <>
struct field_traits<float>
{
	CUGAR_HOST_DEVICE static float min() { return -float(1.0e+30f); }
    CUGAR_HOST_DEVICE static float max() { return  float(1.0e+30f); }
};
/// double specialization of field_traits
///
template <>
struct field_traits<double>
{
	CUGAR_HOST_DEVICE static double min() { return -double(1.0e+30); }
    CUGAR_HOST_DEVICE static double max() { return  double(1.0e+30); }
};
/// uint32 specialization of field_traits
///
template <>
struct field_traits<uint32>
{
	CUGAR_HOST_DEVICE static uint32 min() { return 0; }
    CUGAR_HOST_DEVICE static uint32 max() { return uint32(-1); }
};
/// uint64 specialization of field_traits
///
template <>
struct field_traits<uint64>
{
	CUGAR_HOST_DEVICE static uint64 min() { return 0; }
    CUGAR_HOST_DEVICE static uint64 max() { return uint64(-1); }
};
#endif

using ::sinf;
using ::sin;
using ::cosf;
using ::cos;
using ::sqrtf;
using ::sqrt;
using ::expf;
using ::exp;
using ::logf;
using ::log;

///@} BasicUtils
///@} Basic

} // namespace cugar
