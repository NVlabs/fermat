/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

/*! \file diff.h
 *   \brief Defines classes to perform automatic differentiation
 */

#pragma once

#include <cugar/linalg/matrix.h>
#include <cugar/linalg/tensor.h>
#include <stdio.h>


namespace cugar {

	using ::expf;
	using ::logf;
	using ::sinf;
	using ::cosf;
	using ::sqrtf;

	using ::exp;
	using ::log;
	using ::sin;
	using ::cos;
	using ::sqrt;

///@addtogroup Basic
///@{

///@defgroup AutoDiffModule	Automatic-Differentiation
/// This module defines classes to perform automatic differentiation
///@{

template <uint32 O1, uint32 O2>
struct static_min
{
	static const uint32 value = O1 < O2 ? O1 : O2;
};

///
/// This class allows to represent variables that carry associated differentials up to a specified compile-time order
/// with respect to a predefined set of variables
///
/// \tparam ValType     the value type of the expression
/// \tparam N			the number of variables the expression is differentiated against
/// \tparam O			the maximum order of differentiation, i.e:
///							0 = no derivatives,
///							1 = first order derivatives / gradient,
///							2 = second-order derivatives / Hessian,
///							3 = third-order derivatives,
///							...
///
template <typename ValType, uint32 N, uint32 O>
struct diff_var
{
	typedef ValType													value_type;
	typedef diff_var<ValType,N,O-1>									diff_component_type;
	typedef Vector<diff_component_type, N>							diff_type;

	value_type		u;
	diff_type		du;

	/// default constructor
	///
	CUGAR_HOST_DEVICE diff_var() : du(diff_component_type(value_type(0.0f))) {}

	/// copy constructor
	///
	CUGAR_HOST_DEVICE diff_var(const diff_var<ValType,N,O>& _other) : u(_other.u), du(_other.du) {}

	/// constructor
	///
	///\param _u		scalar value
	///\param _du		differential
	///
	CUGAR_HOST_DEVICE diff_var(const value_type _u, const diff_type _du) : u(_u), du(_du) {}

	/// constructor from scalar
	///
	///\param _u		scalar value
	///
	CUGAR_HOST_DEVICE explicit diff_var(value_type _u) : u(value_type(_u)), du(diff_component_type(0.0f)) {}

	/// conversion to scalar
	///
	CUGAR_HOST_DEVICE operator value_type() const { return u; }

	/// assignment to scalar
	///
	CUGAR_HOST_DEVICE diff_var& operator=(const value_type& _u) { u = _u; return *this; }

	/// return the first-order differential
	///
	CUGAR_HOST_DEVICE diff_type diff() const { return du; }
};

///
/// Specialization of diff_var to 0-order (i.e. a variable not carrying any differential)
/// This class allows to represent variables that carry associated differentials up to a specified compile-time order
/// with respect to a predefined set of variables
///
/// \tparam ValType     the value type of the expression
/// \tparam N			the number of variables the expression is differentiated against
/// \tparam O			the order of differentiation
///
template <typename ValType, uint32 N>
struct diff_var<ValType,N,0>
{
	typedef ValType												value_type;
	typedef ValType												diff_component_type;
	typedef Vector<diff_component_type, N>						diff_type;

	value_type		u;

	/// default constructor
	///
	CUGAR_HOST_DEVICE diff_var() {}

	/// copy constructor
	///
	CUGAR_HOST_DEVICE diff_var(const diff_var<ValType,N,0>& _other) : u(_other.u) {}

	/// constructor from scalar
	///
	///\param _u		scalar value
	///
	CUGAR_HOST_DEVICE explicit diff_var(value_type _u) : u(_u) {}

	/// conversion to scalar
	///
	CUGAR_HOST_DEVICE operator value_type() const { return u; }

	/// assignment to scalar
	///
	CUGAR_HOST_DEVICE diff_var& operator=(const value_type& _u) { u = _u; return *this; }

	/// return the first-order differential
	///
	CUGAR_HOST_DEVICE diff_type diff() const { return diff_type(ValType(0.0f)); }
};

///
/// Specialization of diff_var to scalar (i.e. 1-component) differentials
/// This class allows to represent variables that carry an associated first-order differential (or gradient)
/// with respect to a predefined set of variables.
///
/// \tparam ValType     the value type of the expression
/// \tparam DiffType    the type of differential / gradient of the expression
///
template <typename ValType, uint32 O>
struct diff_var<ValType, 1, O>
{
	typedef ValType													value_type;
	typedef diff_var<ValType,1,O-1>									diff_component_type;
	typedef diff_component_type										diff_type;

	value_type		u;
	diff_type		du;

	/// default constructor
	///
	CUGAR_HOST_DEVICE diff_var() : du(diff_component_type(value_type(0.0f))) {}

	/// copy constructor
	///
	CUGAR_HOST_DEVICE diff_var(const diff_var<ValType,1,O>& _other) : u(_other.u), du(_other.du) {}

	/// constructor
	///
	///\param _u		scalar value
	///\param _du		differential
	///
	CUGAR_HOST_DEVICE diff_var(const value_type _u, const diff_type _du) : u(_u), du(_du) {}

	/// constructor
	///
	///\param _u		scalar value
	///\param _du		differential
	///
	CUGAR_HOST_DEVICE diff_var(const value_type _u, const value_type _du) : u(_u), du(diff_type(_du)) {}

	/// constructor from scalar
	///
	///\param _u		scalar value
	///
	CUGAR_HOST_DEVICE explicit diff_var(value_type _u) : u(value_type(_u)), du(diff_component_type(value_type(0.0f))) {}

	/// conversion to scalar
	///
	CUGAR_HOST_DEVICE operator value_type() const { return u; }

	/// assignment to scalar
	///
	CUGAR_HOST_DEVICE diff_var& operator=(const value_type& _u) { u = _u; return *this; }

	/// return the first-order differential
	///
	CUGAR_HOST_DEVICE diff_type diff() const { return du; }

	// fake operator[] to pretend this is a 1-component vector
	CUGAR_HOST_DEVICE const diff_var<ValType,1,O>&	operator[] (const uint32 i) const { return *this; }
	CUGAR_HOST_DEVICE diff_var<ValType,1,O>&		operator[] (const uint32 i)			{ return *this; }
};

///
/// Specialization of diff_var to scalar (i.e. 1-component) zero-order differentials.
/// This class allows to represent variables that carry an associated first-order differential (or gradient)
/// with respect to a predefined set of variables.
///
/// \tparam ValType     the value type of the expression
/// \tparam DiffType    the type of differential / gradient of the expression
///
template <typename ValType>
struct diff_var<ValType, 1, 0>
{
	typedef ValType												value_type;
	typedef ValType												diff_component_type;
	typedef ValType												diff_type;

	value_type		u;

	/// default constructor
	///
	CUGAR_HOST_DEVICE diff_var() {}

	/// copy constructor
	///
	CUGAR_HOST_DEVICE diff_var(const diff_var<ValType,1,0>& _other) : u(_other.u) {}

	/// constructor from scalar
	///
	///\param _u		scalar value
	///
	CUGAR_HOST_DEVICE explicit diff_var(value_type _u) : u(_u) {}

	/// conversion to scalar
	///
	CUGAR_HOST_DEVICE operator value_type() const { return u; }

	/// assignment to scalar
	///
	CUGAR_HOST_DEVICE diff_var& operator=(const value_type& _u) { u = _u; return *this; }

	/// return the first-order differential
	///
	CUGAR_HOST_DEVICE diff_type diff() const { return diff_type(ValType(0.0f)); }

	// fake operator[] to pretend this is a 1-component vector
	CUGAR_HOST_DEVICE const diff_var<ValType,1,0>&	operator[] (const uint32 i) const { return *this; }
	CUGAR_HOST_DEVICE diff_var<ValType,1,0>&		operator[] (const uint32 i)			{ return *this; }
};

/// a utility function to mark a variable as "primary", i.e. to mark it as a variable we want to differentiate against:
/// this is exactly equivalent to setting its derivative to 1
///
template <typename VT, uint32 O>
CUGAR_HOST_DEVICE
inline void set_primary(diff_var<VT, 1, O>& var, const VT deriv = VT(1.0))
{
	var.du = deriv;
}

/// a utility function to mark a variable as the i-th "primary", i.e. to mark it as a variable we want to differentiate against:
/// this is exactly equivalent to setting the i-th component of its gradient to 1
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline void set_primary(const uint32 i, diff_var<VT, N, O>& var, const VT deriv = VT(1.0))
{
	var.du[i] = deriv;
}

/// given a variable with order 1 differentials, return a copy of order 0.
///
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT, N, 0> decrease_order(const diff_var<VT, N, 1>& op)
{
	diff_var<VT, N, 0> r;
	r.u  = op.u;
	return r;
}

/// given a variable with differentials up to order O, return a copy of order O-1.
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT, N, O-1> decrease_order(const diff_var<VT, N, O>& op)
{
	diff_var<VT, N, O-1> r;
	r.u  = op.u;
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = decrease_order(op.du[i]);
	return r;
}

/// given a variable with differentials up to order 0, return a copy of order 1.
///
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT, N, 1> raise_order(const diff_var<VT, N, 0>& op)
{
	diff_var<VT, N, 1> r;
	r.u  = op.u;
	return r;
}

/// given a variable with differentials up to order O, return a copy of order O+1.
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT, N, O+1> raise_order(const diff_var<VT, N, O>& op)
{
	diff_var<VT, N, O+1> r;
	r.u  = op.u;
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = raise_order(op.du[i]);
	return r;
}

/// given a tensor variable with differentials up to order O, return a copy of order O+1.
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Vector<diff_var<VT, N, O+1>, N> raise_order(const Vector<diff_var<VT, N, O>, N>& op)
{
	Vector<diff_var<VT, N, O+1>, N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = raise_order( op[i] );

	return r;
}
/// given a tensor variable with differentials up to order O, return a copy of order O+1.
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<diff_var<VT, N, O+1>, 1, N> raise_order(const Tensor<diff_var<VT, N, O>, 1, N>& op)
{
	Tensor<diff_var<VT, N, O+1>, 1, N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = raise_order( op[i] );

	return r;
}
/// given a tensor variable with differentials up to order O, return a copy of order O+1.
///
template <typename VT, uint32 N, uint32 O, uint32 TO>
CUGAR_HOST_DEVICE
inline Tensor<diff_var<VT, N, O+1>, TO, N> raise_order(const Tensor<diff_var<VT, N, O>, TO, N>& op)
{
	Tensor<diff_var<VT, N, O+1>, TO, N> r;
	for (uint32 i = 0; i < N; ++i)
		r[i] = raise_order( op[i] );

	return r;
}

/// return the differential of a given variable
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline typename diff_var<VT, N, O>::diff_type diff(const diff_var<VT, N, O> op)
{
	return op.diff();
}

/// return the differentiable Hessian of a given variable
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<diff_var<VT, N, O-2>, 2, N> diff_hessian(const diff_var<VT, N, O> op)
{
	Tensor<diff_var<VT, N, O-2>, 2, N> H;
	for (uint32 i = 0; i < N; ++i)
		for (uint32 j = 0; j < N; ++j)
			H(i,j) = diff( diff(op)[i] )[j];

	return H;
}

/// return the gradient of a given variable as a plain (i.e. non-differentiable) 1-tensor
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<VT, 1, N> gradient(const diff_var<VT, N, O> op)
{
	Tensor<VT, 1, N> J;
	for (uint32 i = 0; i < N; ++i)
		J[i] = diff(op)[i];

	return J;
}

/// return the Jacobian of a given variable as a plain (i.e. non-differentiable) 1-tensor
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<VT, 1, N> jacobian(const diff_var<VT, N, O> op)
{
	Tensor<VT, 1, N> J;
	for (uint32 i = 0; i < N; ++i)
		J[i] = diff(op)[i];

	return J;
}

/// return the Hessian of a given variable as a plain (i.e. non-differentiable) 2-tensor
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<VT, 2, N> hessian(const diff_var<VT, N, O> op)
{
	Tensor<VT, 2, N> H;
	for (uint32 i = 0; i < N; ++i)
		for (uint32 j = 0; j < N; ++j)
			H(i,j) = diff( diff(op)[i] )[j];

	return H;
}

namespace detail {

template <uint32 ORDER> struct dispatch_diff_tensor {};

template <>
struct dispatch_diff_tensor<1>
{
	template <typename VT, uint32 N, uint32 O>
	CUGAR_HOST_DEVICE
	static inline Tensor<VT, 1, N> apply(const diff_var<VT, N, O> op)
	{
		Tensor<VT, 1, N> T;
		for (uint32 i = 0; i < N; ++i)
			for (uint32 j = 0; j < N; ++j)
				T(i) = diff(op)[i];

		return T;
	}
};

template <>
struct dispatch_diff_tensor<2>
{
	template <typename VT, uint32 N, uint32 O>
	CUGAR_HOST_DEVICE
	static inline Tensor<VT, 2, N> apply(const diff_var<VT, N, O> op)
	{
		Tensor<VT, 2, N> T;
		for (uint32 i = 0; i < N; ++i)
			for (uint32 j = 0; j < N; ++j)
				T(i,j) = diff( diff(op)[i] )[j];

		return T;
	}
};

template <>
struct dispatch_diff_tensor<3>
{
	template <typename VT, uint32 N, uint32 O>
	CUGAR_HOST_DEVICE
	static inline Tensor<VT, 3, N> apply(const diff_var<VT, N, O> op)
	{
		Tensor<VT, 3, N> T;
		for (uint32 i = 0; i < N; ++i)
			for (uint32 j = 0; j < N; ++j)
				for (uint32 k = 0; k < N; ++k)
					T(i,j,k) = diff( diff( diff(op)[i] )[j] )[k];

		return T;
	}
};

} // namespace detail

/// return the differential tensor of order M of a given variable as a plain (i.e. non-differentiable) M-tensor
///
///\tparam ORDER	order of the differential
///
template <uint32 ORDER, typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline Tensor<VT, ORDER, N> diff_tensor(const diff_var<VT, N, O> op)
{
	return detail::dispatch_diff_tensor<ORDER>::apply(op);
}

// special-case negation for zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator-(const diff_var<VT,N,0> a)
{
	return diff_var<VT,N,0>( -a.u );
}

// special-case subtraction for zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator-(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a.u - b.u );
}

// special-case subtraction for zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator-(const diff_var<VT,N,0> a, const VT b)
{
	return diff_var<VT,N,0>( a.u - b );
}

// special-case subtraction for zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator-(const VT a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a - b.u );
}


// addition of zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator+(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a.u + b.u );
}

// addition of zero-order var with scalar
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator+(const diff_var<VT,N,0> a, const VT b)
{
	return diff_var<VT,N,0>( a.u + b );
}

// addition of zero-order var with scalar
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator+(const VT a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a + b.u );
}

// multiplication of zero-order var by scalar
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator*(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a.u * b.u );
}

// multiplication of zero-order var by scalar
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator*(const diff_var<VT,N,0> a, const VT b)
{
	return diff_var<VT,N,0>( a.u * b );
}

// multiplication of zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator*(const VT a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a * b.u );
}

// division of zero-order vars
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator/(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
{
	return diff_var<VT,N,0>( a.u / b.u );
}

// division of a zero-order var by scalar
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> operator/(const diff_var<VT,N,0> a, const VT b)
{
	return diff_var<VT,N,0>( a.u / b );
}

namespace detail {

// addition of arbitrary order vars
template <typename VT, uint32 N, uint32 O1, uint32 O2>
struct dispatch_sum
{
	static const uint32 RO = static_min<O1,O2>::value;

	// fully general case
	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,RO> apply(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
	{
		diff_var<VT,N,RO> r;
	
		r.u = a.u + b.u;

		// component-wise expression
		for (uint32 i = 0; i < N; ++i)
			r.du[i] = a.du[i] + b.du[i];

		return r;
	}
};

// addition of zero-order vars
template <typename VT, uint32 N>
struct dispatch_sum<VT,N,0,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u + b.u );
	}
};

// right-addition with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_sum<VT,N,O,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,O> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u + b.u );
	}
};

// left-addition with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_sum<VT,N,0,O>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,O> b)
	{
		return diff_var<VT,N,0>( a.u + b.u );
	}
};

// multiplication of arbitrary order vars
template <typename VT, uint32 N, uint32 O1, uint32 O2>
struct dispatch_mul
{
	static const uint32 RO = static_min<O1,O2>::value;

	// fully general case
	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,RO> apply(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
	{
		diff_var<VT,N,RO> r;
	
		r.u = a.u * b.u;

		// component-wise expression
		for (uint32 i = 0; i < N; ++i)
			r.du[i] = a.du[i] * b + a * b.du[i];

		return r;
	}
};

// addition of zero-order vars
template <typename VT, uint32 N>
struct dispatch_mul<VT,N,0,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u * b.u );
	}
};

// right-addition with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_mul<VT,N,O,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,O> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u * b.u );
	}
};

// left-addition with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_mul<VT,N,0,O>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,O> b)
	{
		return diff_var<VT,N,0>( a.u * b.u );
	}
};


// division of arbitrary order vars
template <typename VT, uint32 N, uint32 O1, uint32 O2>
struct dispatch_div
{
	static const uint32 RO = static_min<O1,O2>::value;

	// fully general case
	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,RO> apply(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
	{
		diff_var<VT,N,RO> r;
	
		r.u = a.u / b.u;

		// component-wise expression
		for (uint32 i = 0; i < N; ++i)
			r.du[i] = (a.du[i] * b - a * b.du[i]) / (b * b);

		return r;
	}
};

// division of zero-order vars
template <typename VT, uint32 N>
struct dispatch_div<VT,N,0,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u / b.u );
	}
};

// right-division with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_div<VT,N,O,0>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,O> a, const diff_var<VT,N,0> b)
	{
		return diff_var<VT,N,0>( a.u / b.u );
	}
};

// left-division with a zero-order var
template <typename VT, uint32 N, uint32 O>
struct dispatch_div<VT,N,0,O>
{
	static const uint32 RO = 0;

	CUGAR_HOST_DEVICE
	static inline diff_var<VT,N,0> apply(const diff_var<VT,N,0> a, const diff_var<VT,N,O> b)
	{
		return diff_var<VT,N,0>( a.u / b.u );
	}
};

} // namespace detail

/// addition of arbitrary order vars
///
template <typename VT, uint32 N, uint32 O1, uint32 O2>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,detail::dispatch_sum<VT,N,O1,O2>::RO> operator+(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
{
	return detail::dispatch_sum<VT,N,O1,O2>::apply(a,b);
}

/// right-addition with a scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator+(const diff_var<VT,N,O> a, const VT b)
{
	return diff_var<VT,N,O>( a.u + b, a.du );
}

/// left-addition with a scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator+(const VT a, const diff_var<VT,N,O> b)
{
	return diff_var<VT,N,O>( a + b.du, b.du );
}

/// negation
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator-(const diff_var<VT,N,O> a)
{
	return diff_var<VT,N,O>( -a.u, -a.du );
}

/// right-subtraction of scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator-(const diff_var<VT,N,O> a, const VT b)
{
	return diff_var<VT,N,O>( a.u - b, a.du );
}

/// left-subtraction of scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator-(const VT a, const diff_var<VT,N,O> b)
{
	return diff_var<VT,N,O>( a - b.u, -b.du );
}

/// subtraction of arbitrary order vars
///
template <typename VT, uint32 N, uint32 O1, uint32 O2>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,detail::dispatch_sum<VT,N,O1,O2>::RO> operator-(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
{
	return detail::dispatch_sum<VT,N,O1,O2>::apply(a,-b);
}

/// multiplication of arbitrary order vars
///
template <typename VT, uint32 N, uint32 O1, uint32 O2>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,detail::dispatch_mul<VT,N,O1,O2>::RO> operator*(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
{
	return detail::dispatch_mul<VT,N,O1,O2>::apply(a,b);
}

/// right-multiplication by scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator*(const diff_var<VT,N,O> a, const VT b)
{
	diff_var<VT,N,O> r;
	r.u = a.u * b;

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = a.du[i] * b;

	return r;
}
/// left-multiplication by scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator*(const VT a, const diff_var<VT,N,O> b)
{
	diff_var<VT,N,O> r;
	r.u = a * b.u;

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = a * b.du[i];

	return r;
}

/// division of arbitrary order vars
///
template <typename VT, uint32 N, uint32 O1, uint32 O2>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,detail::dispatch_div<VT,N,O1,O2>::RO> operator/(const diff_var<VT,N,O1> a, const diff_var<VT,N,O2> b)
{
	return detail::dispatch_div<VT,N,O1,O2>::apply(a,b);
}

/// division by scalar
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator/(const diff_var<VT,N,O> a, const VT b)
{
	return diff_var<VT,N,O>( a.u / b, a.du / b );
}

/// scalar divided by arbitrary order
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> operator/(const VT a, const diff_var<VT,N,O> b)
{
	return detail::dispatch_div<VT,N,O,O>::apply(diff_var<VT,N,O>(a),b);
}

/// += operator
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O>& operator+=(diff_var<VT,N,O>& a, const diff_var<VT,N,O> b)
{
	a = a + b;
	return a;
}
/// -= operator
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O>& operator-=(diff_var<VT,N,O>& a, const diff_var<VT,N,O> b)
{
	a = a - b;
	return a;
}
/// *= operator
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O>& operator*=(diff_var<VT,N,O>& a, const diff_var<VT,N,O> b)
{
	a = a * b;
	return a;
}
/// /= operator
///
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O>& operator/=(diff_var<VT,N,O>& a, const diff_var<VT,N,O> b)
{
	a = a / b;
	return a;
}


// zero-order specialization of sin
template <uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<float,N,0> sin(const diff_var<float,N,0> a)
{
	return diff_var<float,N,0>( ::sinf(a.u) );
}
// zero-order specialization of cos
template <uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<float,N,0> cos(const diff_var<float,N,0> a)
{
	return diff_var<float,N,0>( ::cosf(a.u) );
}
// zero-order specialization of log
template <uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<float,N,0> log(const diff_var<float,N,0> a)
{
	return diff_var<float,N,0>( ::logf(a.u) );
}
// zero-order specialization of exp
template <uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<float,N,0> exp(const diff_var<float,N,0> a)
{
	return diff_var<float,N,0>( ::expf(a.u) );
}
// zero-order specialization of sqrt
template <uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<float,N,0> sqrt(const diff_var<float,N,0> a)
{
	return diff_var<float,N,0>( ::sqrtf(a.u) );
}

// arbitrary-order specialization of sin
template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> sin(const diff_var<float,N,O> a)
{
	diff_var<float,N,O> r;
	r.u = ::sinf(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = cos( decrease_order( a ) ) * a.du[i];

	return r;
}
// arbitrary-order specialization of cos
template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> cos(const diff_var<float,N,O> a)
{
	diff_var<float,N,O> r;
	r.u = ::cosf(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = -sin( decrease_order( a ) ) * a.du[i];

	return r;
}

// arbitrary-order specialization of log
template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> log(const diff_var<float,N,O> a)
{
	diff_var<float,N,O> r;
	r.u = ::logf(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = a.du[i] / a;

	return r;
}

// arbitrary-order specialization of exp
template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> exp(const diff_var<float,N,O> a)
{
	diff_var<float,N,O> r;
	r.u = ::expf(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = exp( decrease_order( a ) ) * a.du[i];

	return r;
}

// arbitrary-order specialization of sqrt
template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> sqrt(const diff_var<float,N,O> a)
{
	diff_var<float,N,O> r;
	r.u = ::sqrtf(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = a.du[i] * 0.5f / sqrt(decrease_order(a));

	return r;
}

template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> sinf(const diff_var<float,N,O> a) { return sin(a); }

template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> cosf(const diff_var<float,N,O> a) { return cos(a); }

template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> logf(const diff_var<float,N,O> a) { return log(a); }

template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> expf(const diff_var<float,N,O> a) { return exp(a); }

template <uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<float,N,O> sqrtf(const diff_var<float,N,O> a) { return sqrt(a); }

// zero-order specialization of sin
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> sin(const diff_var<VT,N,0> a)
{
	return diff_var<VT,N,0>( sin(a.u) );
}
// zero-order specialization of cos
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> cos(const diff_var<VT,N,0> a)
{
	return diff_var<VT,N,0>( cos(a.u) );
}
// zero-order specialization of log
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> log(const diff_var<VT,N,0> a)
{
	return diff_var<VT,N,0>( log(a.u) );
}
// zero-order specialization of exp
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> exp(const diff_var<VT,N,0> a)
{
	return diff_var<float,N,0>( exp(a.u) );
}
// zero-order specialization of sqrt
template <typename VT, uint32 N>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,0> sqrt(const diff_var<VT,N,0> a)
{
	return diff_var<VT,N,0>( sqrt(a.u) );
}

// arbitrary-order specialization of sin
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> sin(const diff_var<VT,N,O> a)
{
	diff_var<VT,N,O> r;
	r.u = sin(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = cos( decrease_order( a ) ) * a.du[i];

	return r;
}
// arbitrary-order specialization of cos
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> cos(const diff_var<VT,N,O> a)
{
	diff_var<VT,N,O> r;
	r.u = cos(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = -sin( decrease_order( a ) ) * a.du[i];

	return r;
}

// arbitrary-order specialization of log
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> log(const diff_var<VT,N,O> a)
{
	diff_var<VT,N,O> r;
	r.u = log(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = a.du[i] / a;

	return r;
}

// arbitrary-order specialization of exp
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> exp(const diff_var<VT,N,O> a)
{
	diff_var<VT,N,O> r;
	r.u = exp(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = exp( decrease_order( a ) ) * a.du[i];

	return r;
}

// arbitrary-order specialization of sqrt
template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> sqrt(const diff_var<VT,N,O> a)
{
	diff_var<VT,N,O> r;
	r.u = sqrt(a.u);

	// component-wise expression
	for (uint32 i = 0; i < N; ++i)
		r.du[i] = (a.du[i] * VT(0.5f)) / sqrt(decrease_order(a));

	return r;
}

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> max(const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u > b.u ? a : b; }

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline diff_var<VT,N,O> min(const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u < b.u ? a : b; }

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline bool operator< (const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u < b.u; }

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline bool operator<= (const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u <= b.u; }

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline bool operator> (const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u > b.u; }

template <typename VT, uint32 N, uint32 O>
CUGAR_HOST_DEVICE
inline bool operator>= (const diff_var<VT,N,O> a, const diff_var<VT,N,O> b) { return a.u >= b.u; }

///@} // end of the AutoDiffModule group
///@} // end of the Basic group

} // namespace cugar
