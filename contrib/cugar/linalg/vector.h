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
#include <cugar/basic/numbers.h>
#include <cmath>
#include <limits>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
 // suppress exception specification warnings
#pragma warning(push)
#pragma warning(disable:4068)
#endif

namespace cugar {

///@addtogroup LinalgModule
///@{

///@defgroup VectorsModule Vectors
/// defines linear vectors structs and functions
///@{

///
/// A generic small vector class with the dimension set at compile-time
///
template <typename T, uint32 DIM>
struct Vector
{
	typedef T value_type;

	static const uint32 DIMENSION = DIM;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector() {}
    
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector(const T v)
	{
		#pragma unroll
		for (uint32 d = 0; d < DIM; ++d)
			data[d] = v;
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector(const T* v)
	{
		#pragma unroll
		for (uint32 d = 0; d < DIM; ++d)
			data[d] = v[d];
	}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return DIM; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    const T& operator[] (const uint32 i) const { return data[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
          T& operator[] (const uint32 i)       { return data[i]; }

    T data[DIM];
};

///
/// A generic small vector class with the dimension set at compile-time
///
template <typename T>
struct Vector<T,1>
{
	typedef T									base_type;
	typedef T									value_type;

	static const uint32 DIMENSION = 1;

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,1>() {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,1>(const T v0)
    {
        data = v0;
    }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return 1; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    const T& operator[] (const uint32 i) const { return data; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
          T& operator[] (const uint32 i)       { return data; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    operator base_type() const { return data; }

    base_type data;
};

///
/// A generic small vector class with the dimension set at compile-time
///
template <typename T>
struct Vector<T, 2> : vector_type<T, 2>::type
{
    typedef typename vector_type<T,2>::type		base_type;
	typedef T									value_type;

	static const uint32 DIMENSION = 2;

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,2>() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T,2>(const T v)
	{
		base_type::x = base_type::y = v;
	}
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,2>(const T v0, const T v1)
    {
		base_type::x = v0;
		base_type::y = v1;
    }

	template <typename U>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Vector<T, 2>(const Vector<U, 2> op)
	{
		base_type::x = T(op[0]);
		base_type::y = T(op[1]);
	}
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2>(const base_type v) : base_type(v) {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return 2; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const T& operator[] (const uint32 i) const { return (&this->base_type::x)[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	T& operator[] (const uint32 i)       { return (&this->base_type::x)[i]; }

    //CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    //operator base_type() const { return data; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> xy() const { return Vector<T,2>(base_type::x, base_type::y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> yx() const { return Vector<T, 2>(base_type::y, base_type::x); }
};

///
/// A generic small vector class with the dimension set at compile-time
///
template <typename T>
struct Vector<T, 3> : vector_type<T, 3>::type
{
    typedef typename vector_type<T,3>::type     base_type;
	typedef T									value_type;

	static const uint32 DIMENSION = 3;

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,3>() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3>(const T v)
	{
		base_type::x = base_type::y = base_type::z = v;
	}
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,3>(const T v0, const T v1, const T v2)
    {
		base_type::x = v0;
		base_type::y = v1;
		base_type::z = v2;
    }

	template <typename U>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Vector<T, 3>(const Vector<U,3> op)
	{
		base_type::x = T(op[0]);
		base_type::y = T(op[1]);
		base_type::z = T(op[2]);
	}

	template <typename U>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Vector<T, 3>(const Vector<U, 2> op, const U v2 = 0.0f)
	{
		base_type::x = T(op[0]);
		base_type::y = T(op[1]);
		base_type::z = T(v2);
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3>(const base_type v) : base_type(v) {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return 3; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const T& operator[] (const uint32 i) const { return (&this->base_type::x)[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	T& operator[] (const uint32 i)       { return (&this->base_type::x)[i]; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T,3> xyz() const { return *this; }
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3> zxy() const { return Vector<T, 3>(base_type::z, base_type::x, base_type::y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3> yzx() const { return Vector<T, 3>(base_type::y, base_type::z, base_type::x); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> xy() const { return Vector<T,2>(base_type::x, base_type::y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> yx() const { return Vector<T, 2>(base_type::y, base_type::x); }

	//CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	//operator base_type() const { return *(base_type*)this; }
};

///
/// A generic small vector class with the dimension set at compile-time
///
template <typename T>
struct Vector<T, 4> : public vector_type<T, 4>::type
{
    typedef typename vector_type<T,4>::type     base_type;
	typedef T									value_type;

	static const uint32 DIMENSION = 4;

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,4>() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T,4>(const T v)
	{
		base_type::x = base_type::y = base_type::z = base_type::w = v;
	}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    Vector<T,4>(const T v0, const T v1, const T v2, const T v3)
    {
		base_type::x = v0;
		base_type::y = v1;
		base_type::z = v2;
		base_type::w = v3;
    }

	template <typename U>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Vector<T, 4>(const Vector<U, 4> op)
	{
		base_type::x = T(op[0]);
		base_type::y = T(op[1]);
		base_type::z = T(op[2]);
		base_type::w = T(op[3]);
	}
	
	template <typename U>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Vector<T, 4>(const Vector<U, 3> op, const T v3)
	{
		base_type::x = T(op[0]);
		base_type::y = T(op[1]);
		base_type::z = T(op[2]);
		base_type::w = v3;
	}
    
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 4>(const base_type v) : base_type(v) {}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return 4; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const T& operator[] (const uint32 i) const { return (&this->base_type::x)[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	T& operator[] (const uint32 i)       { return (&this->base_type::x)[i]; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3> xyz() const { return Vector<T, 3>(base_type::x, base_type::y, base_type::z); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3> zxy() const { return Vector<T, 3>(base_type::z, base_type::x, base_type::y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 3> yzx() const { return Vector<T, 3>(base_type::y, base_type::z, base_type::x); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> xy() const { return Vector<T, 2>(base_type::x, base_type::y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector<T, 2> yx() const { return Vector<T, 2>(base_type::y, base_type::x); }
	
	//CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	//operator base_type() const { return *(base_type*)this; }
};

template <typename T,uint32 DIM_T> struct vector_traits< Vector<T,DIM_T> > { typedef T value_type; const static uint32 DIM = DIM_T; };

/// \relates Vector
/// vector negation
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator- (const Vector<T, DIM>& op);

/// \relates Vector
/// vector addition
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator+ (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector addition
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator+= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector subtraction
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator- (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector subtraction
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator-= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector multiplication
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator* (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector multiplication
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator*= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector multiplication
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator* (const T op1, const Vector<T, DIM>& op2);

/// \relates Vector
/// vector multiplication by scalar
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator* (const Vector<T, DIM>& op1, const T op2);

/// \relates Vector
/// vector multiplication by scalar
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM>& operator*= (Vector<T, DIM>& op1, const T op2);

/// \relates Vector
/// vector division
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator/ (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector division
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator/= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector division by scalar
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator/ (const Vector<T, DIM>& op1, const T op2);

/// \relates Vector
/// vector division by scalar
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM>& operator/= (Vector<T, DIM>& op1, const T op2);

/// \relates Vector
/// binary component-wise vector min
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> min(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// binary component-wise vector max
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> max(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// component-wise vector min
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> min(const Vector<T,DIM>& op1, const T op2);

/// \relates Vector
/// component-wise vector max
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> max(const Vector<T,DIM>& op1, const T op2);

/// \relates Vector
/// component-wise vector abs
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> abs(const Vector<T,DIM>& op);

/// \relates Vector
/// return the maximum component
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T max_comp(const Vector<T, DIM>& op);

/// \relates Vector
/// return the minimum component
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T min_comp(const Vector<T, DIM>& op);

/// \relates Vector
/// return true if any of the components is non-zero
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool any(const Vector<T,DIM>& op);

/// \relates Vector
/// return true if all of the components are non-zero
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool all(const Vector<T,DIM>& op);

/// \relates Vector
/// vector equality test
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator== (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// vector inequality test
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator!= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// lexicographic vector less
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator< (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// lexicographic vector greater
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator> (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// lexicographic vector less or equal
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator<= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// lexicographic vector greater or equal
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator>= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2);

/// \relates Vector
/// dot product
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T dot(const Vector<T, DIM>& op1, const Vector<T, DIM>& op2);

/// \relates Vector
/// Euclidean length
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T length(const Vector<T, DIM>& op);

/// \relates Vector
/// Euclidean normalization
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> normalize(const Vector<T, DIM>& op);

/// \relates Vector
/// cross product
///
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3u> cross(const Vector<T, 3u>& op1, const Vector<T, 3u>& op2);

/// \relates Vector
/// reflect a vector against another
/// NOTE: this function assumes the vector I is incident into the surface, i.e. dot(I,N) < 0,
/// so that the reflected ray will point in the opposite direction, contrarily to the custom in BSDF
/// sampling literature of having bouth I and the reflected direction point outwards.
///
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> reflect(const Vector<T, 3> I, const Vector<T, 3> N);

/// \relates Vector
/// refract a vector against a normal
/// NOTE: this function assumes the vector I is incident into the surface, i.e. dot(I,N) < 0,
/// so that the refracted ray will point in the same direction, contrarily to the custom in BSDF
/// sampling literature of having I point outwards and the refracted direction inwards.
///
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> refract(const Vector<T, 3> I, const Vector<T, 3> N, const float eta);

/// \relates Vector
/// compute the normal responsible for refraction from I to T
/// NOTE: this function assumes the vector I is incident into the surface, i.e. dot(I,N) < 0,
/// and that the refracted ray points in the same direction, contrarily to the custom in BSDF
/// sampling literature of having I point outwards and T inwards.
///
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> refraction_normal(const Vector<T, 3> I, const Vector<T, 3> T, const float eta);

/// \relates Vector
/// given a vector N and an incident vector I, return sgn(dot(N,I)) * N
///
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> faceforward(const Vector<T, 3> N, const Vector<T, 3> I);

/// \relates Vector
/// return a vector orthogonal to a given vector
///
template <typename T>
CUGAR_HOST_DEVICE
Vector<T, 3> orthogonal(const Vector<T, 3> v);

/// \relates Vector
/// pack a normalized 2d vector using n bits per component
///
template <typename T>
CUGAR_HOST_DEVICE
uint32 pack_vector(const Vector<T, 2> v, const uint32 n_bits_comp);

/// \relates Vector
/// unpack a normalized 2d vector using n bits per component
///
template <typename T>
CUGAR_HOST_DEVICE
Vector<T, 2> unpack_vector(const uint32 u, const uint32 n_bits_comp);

/// \relates Vector
/// component-wise vector modulo
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> mod(const Vector<T, DIM>& op, const T m);

/// \relates Vector
/// component-wise vector sqrt
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> sqrt(const Vector<T, DIM>& op);

/// \relates Vector
/// return true iff all components are finite
///
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool is_finite(const Vector<T, DIM>& op);


typedef Vector<float, 2>	Vector2f;
typedef Vector<float, 3>	Vector3f;
typedef Vector<float, 4>	Vector4f;
typedef Vector<double, 2>	Vector2d;
typedef Vector<double, 3>	Vector3d;
typedef Vector<double, 4>	Vector4d;
typedef Vector<int, 2>		Vector2i;
typedef Vector<int, 3>		Vector3i;
typedef Vector<int, 4>		Vector4i;
typedef Vector<uint32, 2>	Vector2u;
typedef Vector<uint32, 3>	Vector3u;
typedef Vector<uint32, 4>	Vector4u;

///@} VectorsModule
///@} LinalgModule

} // namespace cugar

#include <cugar/linalg/vector_inl.h>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
  // suppress exception specification warnings
#pragma warning(pop)
#endif
