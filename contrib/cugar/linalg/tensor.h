/*
 * cugar
 * Copyright (c) 2011-2014, NVIDIA CORPORATION. All rights reserved.
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
#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <cmath>
#include <limits>

namespace cugar {

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <typename T, uint32 ORDER, uint32 N>
struct Tensor {};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <uint32 ORDER>
struct TensorIndex {};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <>
struct TensorIndex<0>
{
	uint32 dummy;
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <>
struct TensorIndex<1>
{
	TensorIndex<1>()  {};
	TensorIndex<1>(const uint32 _i) : x(_i)  {};

	operator uint32() const { return x; }

	uint32 x;
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <>
struct TensorIndex<2> : public uint2
{
	TensorIndex<2>()  {};
	TensorIndex<2>(const uint2 _i) : uint2(_i)  {};
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <>
struct TensorIndex<3> : public uint3
{
	TensorIndex<3>()  {};
	TensorIndex<3>(const uint3 _i) : uint3(_i)  {};
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <typename T, uint32 N>
struct Tensor<T,0,N>
{
	typedef T		value_type;
	typedef T		component_type;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Tensor(const value_type _d) : data(_d) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	operator value_type() const { return data; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type operator() () const { return data; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (TensorIndex<0> i) const { return data; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (TensorIndex<0> i) { return data; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 order() const { return 0; }

	value_type data;
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <typename T, uint32 N>
struct Tensor<T,1,N> : public Vector<T,N>
{
	typedef T				value_type;
	typedef Vector<T,N>		base_type;
	typedef T				component_type;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Tensor(const value_type _v) : base_type(_v) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor(const Vector<T,N>& _v) : base_type(_v) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor& operator=(const Vector<T,N>& _v) { *static_cast<base_type*>(this) = _v; return *this; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor& operator=(const Tensor<T,1,N>& _v) { *static_cast<base_type*>(this) = static_cast<base_type>(_v); return *this; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (TensorIndex<1> i) const { return base_type::operator[](i); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (TensorIndex<1> i) { return base_type::operator[](i); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 order() const { return 1; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return N; }
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <typename T, uint32 N>
struct Tensor<T,2,N> : public Matrix<T,N,N>
{
	typedef T				value_type;
	typedef Matrix<T,N,N>	base_type;
	typedef Tensor<T,1,N>	component_type;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Tensor(const value_type _v) : base_type(_v) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor(const base_type& _v) : base_type(_v) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor& operator=(const base_type& _v) { *static_cast<base_type*>(this) = _v; return *this; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor& operator=(const Tensor<T,2,N>& _v) { *static_cast<base_type*>(this) = static_cast<base_type>(_v); return *this; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (const TensorIndex<2> i) const { return base_type::operator()(i.x, i.y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (const TensorIndex<2> i) { return base_type::operator()(i.x, i.y); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (const uint32 i, const uint32 j) const { return base_type::operator()(i,j); }
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (const uint32 i, const uint32 j) { return base_type::operator()(i,j); }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const component_type& operator[] (const uint32 i) const { return reinterpret_cast<const component_type&>(base_type::operator[](i)); }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	component_type& operator[] (const uint32 i) { return reinterpret_cast<component_type&>(base_type::operator[](i)); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 order() const { return 2; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return N; }
};

///
/// A generic small N^D tensor class with the order set at compile-time
///
template <typename T, uint32 N>
struct Tensor<T,3,N>
{
	typedef T				value_type;
	typedef Tensor<T,2,N>	component_type;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor() {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	explicit Tensor(const value_type _v)
	{
		#pragma unroll
		for (uint32 i = 0; i < N; ++i)
			data[i] = component_type(_v);
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Tensor& operator=(const Tensor<T,3,N>& _v)
	{
		#pragma unroll
		for (uint32 i = 0; i < N; ++i)
			data[i] = _v.data[i];
		return *this;
	}

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const component_type& operator[] (const uint32 i) const { return data[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	component_type& operator[] (const uint32 i)				{ return data[i]; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (const uint32 i, const uint32 j, const uint32 k) const { return data[i](j,k); }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (const uint32 i, const uint32 j, const uint32 k)				{ return data[i](j,k); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	const value_type& operator() (const TensorIndex<3> i) const { return data[i.x](i.y, i.z); }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	value_type& operator() (const TensorIndex<3> i) { return data[i.x](i.y, i.z); }
	
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 order() const { return 3; }

    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
    uint32 dimension() const { return N; }

	component_type	 data[N];
};

/// Outer product of two tensors
///
CUGAR_API_CS CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
float outer_product(const float op1, const float op2) { return op1 * op2; }

/// Outer product of two tensors
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,0,N> outer_product(const Tensor<T, 0, N> op1, const Tensor<T, 0, N> op2)
{
	return Tensor<T,0,N>( op1 * op2 );
}

/// Outer product of two tensors
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,1,N> outer_product(const Tensor<T, 1, N> op1, const T op2)
{
	Tensor<T,1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] * op2;
	return r;
}

/// Outer product of two tensors
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,1,N> outer_product(const T op1, const Tensor<T, 1, N> op2)
{
	Tensor<T,1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1 * op2[i];
	return r;
}

/// Outer product of two tensors
///
template <typename T, uint32 D1, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D1,N> outer_product(const Tensor<T, D1, N> op1, const T op2)
{
	Tensor<T,D1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] * op2;
	return r;
}
/// Outer product of two tensors
///
template <typename T, uint32 D2, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D2,N> outer_product(const T op1, const Tensor<T, D2, N> op2)
{
	Tensor<T,D2,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1 * op2[i];
	return r;
}

/// Outer product of two tensors
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,2,N> outer_product(const Tensor<T, 1, N> op1, const Tensor<T, 1, N> op2)
{
	Tensor<T,2,N> r;
	for (uint32 i = 0; i < N; ++i)
		for (uint32 j = 0; j < N; ++j)
			r(i,j) = op1[i] * op2[j];
	return r;
}
/// Outer product of two tensors
///
template <typename T, uint32 D2, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D2+1,N> outer_product(const Tensor<T, 1, N> op1, const Tensor<T, D2, N> op2)
{
	Tensor<T,D2+1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = outer_product( op1[i], op2 );
	return r;
}
/// Outer product of two tensors
///
template <typename T, uint32 N, uint32 D1> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,3,N> outer_product(const Tensor<T, D1, N> op1, const Tensor<T, 1, N> op2)
{
	Tensor<T,D1+1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = outer_product( op1[i], op2 );
	return r;
}

/// Multiplication by a constant
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,1,N> operator* (const T op1, const Tensor<T, 1, N> op2)
{
	Tensor<T,1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1 * op2[i];
	return r;
}
/// Multiplication by a constant
///
template <typename T, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,1,N> operator* (const Tensor<T, 1, N> op1, const T op2)
{
	Tensor<T,1,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] * op2;
	return r;
}

/// Multiplication by a constant
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator* (const Tensor<T, D, N> op1, const T op2)
{
	Tensor<T,D,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] * op2;
	return r;
}
/// Multiplication by a constant
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator* (const T op1, const Tensor<T, D, N> op2)
{
	Tensor<T,D,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1 * op2[i];
	return r;
}

/// Multiplication by a constant
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator* (const Tensor<T, D, N> op1, const Tensor<T, 0, N> op2)
{
	return op1 * T(op2);
}
/// Multiplication by a constant
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator* (const Tensor<T, 0, N> op1, const Tensor<T, D, N> op2)
{
	return T(op1) * op2;
}

/// Sum of two tensors
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator+ (const Tensor<T, D, N> op1, const Tensor<T, D, N> op2)
{
	Tensor<T,D,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] + op2[i];
	return r;
}
/// Sum of two tensors
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator- (const Tensor<T, D, N> op1, const Tensor<T, D, N> op2)
{
	Tensor<T,D,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] - op2[i];
	return r;
}

/// Division by a constant
///
template <typename T, uint32 D, uint32 N> CUGAR_API_CS CUGAR_HOST_DEVICE
Tensor<T,D,N> operator/ (const Tensor<T, D, N> op1, const T op2)
{
	Tensor<T,D,N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = op1[i] / op2;
	return r;
}

} // namespace cugar
