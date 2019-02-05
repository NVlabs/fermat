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

#pragma once

#include <cugar/linalg/vector.h>
#include <cmath>

namespace cugar {

/// \page linalg_page Linear Algebra Module
///
/// This \ref LinalgModule "module" implements various linear-algebra classes and functions
///
/// - Vector
/// - Matrix
/// - Bbox
///

///@defgroup LinalgModule Linear Algebra
/// This module defines linear algebra objects and functions
///@{

///@addtogroup MatricesModule Matrices
///@{

///
/// A dense N x M matrix class over a templated type T.
///
template <typename T, int N, int M> struct CUGAR_API_CS Matrix
{
public:
	typedef T value_type;
	typedef T Field_type;

    typedef Vector<T,M> row_vector;
    typedef Vector<T,N> column_vector;

public:
CUGAR_HOST_DEVICE inline                      Matrix     ();
CUGAR_HOST_DEVICE inline        explicit      Matrix     (const T s);
CUGAR_HOST_DEVICE inline        explicit      Matrix     (const Vector<T,M>& v);
CUGAR_HOST_DEVICE inline                      Matrix     (const Matrix<T,N,M>&);
CUGAR_HOST_DEVICE inline                      Matrix     (const Vector<T,M> *v);
CUGAR_HOST_DEVICE inline                      Matrix     (const T *v);
CUGAR_HOST_DEVICE inline                      Matrix     (const T **v);
//inline                      Matrix     (const T v[N][M]);

CUGAR_HOST_DEVICE inline        Matrix<T,N,M>&    operator  = (const Matrix<T,N,M>&);
CUGAR_HOST_DEVICE inline        Matrix<T,N,M>&    operator += (const Matrix<T,N,M>&);
CUGAR_HOST_DEVICE inline        Matrix<T,N,M>&    operator -= (const Matrix<T,N,M>&);
CUGAR_HOST_DEVICE inline        Matrix<T,N,M>&    operator *= (T);
CUGAR_HOST_DEVICE inline        Matrix<T,N,M>&    operator /= (T);

CUGAR_HOST_DEVICE inline        const Vector<T,M>& operator [] (int) const;
CUGAR_HOST_DEVICE inline        Vector<T,M>&       operator [] (int);
CUGAR_HOST_DEVICE inline        const Vector<T,M>& get (int) const;
CUGAR_HOST_DEVICE inline        void               set (int, const Vector<T,M>&);

CUGAR_HOST_DEVICE inline        const T&    operator () (int i, int j) const;
CUGAR_HOST_DEVICE inline        T&          operator () (int i, int j);

CUGAR_HOST_DEVICE inline        T           det() const;

CUGAR_HOST_DEVICE static inline Matrix<T,N,M> one();

friend CUGAR_HOST_DEVICE int            operator == <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE int            operator != <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  - <T,N,M> (const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  + <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  - <T,N,M> (const Matrix<T,N,M>&,  const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  * <T,N,M> (const Matrix<T,N,M>&,  T);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  * <T,N,M> (T,                     const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Vector<T,M>    operator  * <T,N,M> (const Vector<T,N>&,    const Matrix<T,N,M>&);
friend CUGAR_HOST_DEVICE Vector<T,N>    operator  * <T,N>   (const Vector<T,N>&,    const Matrix<T,N,N>&);
friend CUGAR_HOST_DEVICE Vector<T,N>    operator  * <T,N,M> (const Matrix<T,N,M>&,  const Vector<T,M>&);
friend CUGAR_HOST_DEVICE Vector<T,N>    operator  * <T,N>   (const Matrix<T,N,N>&,  const Vector<T,N>&);
friend CUGAR_HOST_DEVICE Matrix<T,N,M>  operator  / <T,N,M> (const Matrix<T,N,M>&,  T);

public:
	Vector<T,M> r[N];
};

typedef Matrix<float,2,2>  Matrix2x2f;
typedef Matrix<double,2,2> Matrix2x2d;
typedef Matrix<float,3,3>  Matrix3x3f;
typedef Matrix<double,3,3> Matrix3x3d;
typedef Matrix<float,4,4>  Matrix4x4f;
typedef Matrix<double,4,4> Matrix4x4d;
typedef Matrix<float,2,3>  Matrix2x3f;
typedef Matrix<float,3,2>  Matrix3x2f;
typedef Matrix<double,2,3>  Matrix2x3d;
typedef Matrix<double,3,2>  Matrix3x2d;

template <typename T, int N, int M, int Q> CUGAR_HOST_DEVICE Matrix<T,N,Q>& multiply   (const Matrix<T,N,M>&,  const Matrix<T,M,Q>&,  Matrix<T,N,Q>&);
template <typename T, int N, int M, int Q> CUGAR_HOST_DEVICE Matrix<T,N,Q>  operator * (const Matrix<T,N,M>&,  const Matrix<T,M,Q>&);
template <typename T, int N, int M> CUGAR_HOST_DEVICE Vector<T,M>&     multiply    (const Vector<T,N>&,    const Matrix<T,N,M>&,  Vector<T,M>&);
template <typename T, int N, int M> CUGAR_HOST_DEVICE Vector<T,N>&     multiply    (const Matrix<T,N,M>&,  const Vector<T,M>&,    Vector<T,N>&);
template <typename T, int N, int M> CUGAR_HOST_DEVICE Matrix<T,M,N>    transpose   (const Matrix<T,N,M>&);
template <typename T, int N, int M> CUGAR_HOST_DEVICE Matrix<T,M,N>&   transpose   (const Matrix<T,N,M>&,  Matrix<T,M,N>&);
template <typename T, int N, int M> CUGAR_HOST_DEVICE bool             invert      (const Matrix<T,N,M>&,  Matrix<T,M,N>&); // gives inv(A^t * A)*A^t
template <typename T, int N, int M> CUGAR_HOST_DEVICE T                det         (const Matrix<T,N,M>&);
template <typename T>               CUGAR_HOST_DEVICE void             cholesky    (const Matrix<T,2,2>&, Matrix<T,2,2>&);

/// Outer product of two vectors
///
template <typename T, uint32 N, uint32 M> CUGAR_API_CS CUGAR_HOST_DEVICE Matrix<T,N,M> outer_product(const Vector<T,N> op1, const Vector<T,M> op2);

/// build a 3d translation matrix
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> translate(const Vector<T,3>& vec);

/// build a 3d scaling matrix
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> scale(const Vector<T,3>& vec);

/// build a 3d perspective matrix
///
template <typename T>
Matrix<T,4,4> perspective(T fovy, T aspect, T zNear, T zFar);

/// build a 3d look at matrix
///
template <typename T>
Matrix<T,4,4> look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up, bool flip_sign = false);

/// build the inverse of a 3d look at matrix
///
template <typename T>
Matrix<T,4,4> inverse_look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up, bool flip_sign = false);

/// build a 3d rotation around the X axis
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_X(const T q);

/// build a 3d rotation around the Y axis
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_Y(const T q);

/// build a 3d rotation around the Z axis
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_Z(const T q);

/// build a 3d rotation around an arbitrary axis
///
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_axis(const T q, const Vector3f& axis);

/// transform a 3d point with a perspective transform
///
CUGAR_HOST_DEVICE inline Vector3f ptrans(const Matrix4x4f& m, const Vector3f& v);

/// transform a 3d vector with a perspective transform
///
CUGAR_HOST_DEVICE inline Vector3f vtrans(const Matrix4x4f& m, const Vector3f& v);

/// get the eigenvalues of a matrix
///
CUGAR_HOST_DEVICE inline Vector2f eigen_values(const Matrix2x2f& m);

/// get the singular values of a matrix
///
CUGAR_HOST_DEVICE inline Vector2f singular_values(const Matrix2x2f& m);

/// get the singular value decomposition of a matrix
///
CUGAR_HOST_DEVICE inline void svd(
	const Matrix2x2f&	m,
	Matrix2x2f&			u,
	Vector2f&			s,
	Matrix2x2f&			v);

/// a generic outer product functor:
/// this class is not an STL binary functor, in the sense it does not define
/// its argument and result types
///
struct GenericOuterProduct
{
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	float operator() (const float op1, const float op2) const { return op1 * op2; }

	template <typename T, uint32 N, uint32 M> 
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	Matrix<T,N,M> operator() (const Vector<T,N> op1, const Vector<T,M> op2) const
	{
		return outer_product( op1, op2 );
	}
};

/// an outer product functor
///
template <typename T, uint32 N, uint32 M> 
struct OuterProduct
{
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	Matrix<T,N,M> operator() (const Vector<T,N> op1, const Vector<T,M> op2) const
	{
		return outer_product( op1, op2 );
	}
};

/// outer product functor specialization
///
template <typename T> 
struct OuterProduct<T,1,1>
{
	CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
	T operator() (const T op1, const T op2) const { return op1 * op2; }
};

typedef OuterProduct<float,2,2> OuterProduct2x2f;
typedef OuterProduct<float,3,3> OuterProduct3x3f;
typedef OuterProduct<float,4,4> OuterProduct4x4f;

typedef OuterProduct<float,2,3> OuterProduct2x3f;
typedef OuterProduct<float,3,2> OuterProduct3x2f;

typedef OuterProduct<double,2,2> OuterProduct2x2d;
typedef OuterProduct<double,3,3> OuterProduct3x3d;
typedef OuterProduct<double,4,4> OuterProduct4x4d;

typedef OuterProduct<double,2,3> OuterProduct2x3d;
typedef OuterProduct<double,3,2> OuterProduct3x2d;

///@} MatricesModule
///@} LinalgModule

} // namespace cugar

#include <cugar/linalg/matrix_inline.h>
