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

//
// I M P L E M E N T A T I O N
//

#ifdef _V
#undef _V
#endif
#define _V(v, i)   (((Vector<T,M>&)v).x[(i)])

#ifdef _M
#undef _M
#endif
#define _M(m, i, j)   (((Matrix<T,N,M>&)m).r[(i)].x[(j)])

#ifdef _CM
#undef _CM
#endif
#define _CM(m, i, j)   (((const Matrix<T,N,M>&)m).r[(i)].x[(j)])


//
// Matrix inline methods
//

template <typename T, int N, int M> Matrix<T,N,M>::Matrix() { }

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T s)
{
	for (int i = 0; i < N; i++)
        r[i] = Vector<T,M>(s);
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] = m.r[i];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const Vector<T,M> *v)
{
	for (int i = 0; i < N; i++)
		r[i] = v[i];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T *v)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i*M + j];
}

template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T **v)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i][j];
}
/*
template <typename T, int N, int M> Matrix<T,N,M>::Matrix(const T v[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = v[i][j];
}
*/
template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator  = (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] = m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator += (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] += m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator -= (const Matrix<T,N,M>& m)
{
	for (int i = 0; i < N; i++)
		r[i] -= m.r[i];
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator *= (T k)
{
	for (int i = 0; i < N; i++)
		r[i] *= k;
	return *this;
}

template <typename T, int N, int M> Matrix<T,N,M>& Matrix<T,N,M>::operator /= (T k)
{
	for (int i = 0; i < N; i++)
		r[i] /= k;
	return *this;
}

template <typename T, int N, int M> const Vector<T,M>& Matrix<T,N,M>::operator [] (int i) const
{
	return r[i];
}
template <typename T, int N, int M> Vector<T,M>& Matrix<T,N,M>::operator [] (int i)
{
	return r[i];
}

template <typename T, int N, int M> const Vector<T,M>& Matrix<T,N,M>::get(int i) const
{
	return r[i];
}
template <typename T, int N, int M> void Matrix<T,N,M>::set(int i, const Vector<T,M>& v)
{
	r[i] = v;
}

template <typename T, int N, int M> T Matrix<T,N,M>::operator () (int i, int j) const
{
	return r[i][j];
}
template <typename T, int N, int M> T& Matrix<T,N,M>::operator () (int i, int j)
{
	return r[i][j];
}

template <typename T, int N, int M> T Matrix<T,N,M>::det() const
{
	return det(*this);
}

template <typename T, int N, int M>
Matrix<T, N, M> Matrix<T,N,M>::one()
{
	Matrix<T,N,M> r;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[i][j] = (i == j) ? 1.0f : 0.0f;

	return r;
}


template <typename T, int N, int M, int Q> CUGAR_HOST_DEVICE Matrix<T,N,Q>& multiply(const Matrix<T,N,M>& a, const Matrix<T,M,Q>& b, Matrix<T,N,Q>& r)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < Q; j++)
		{
			r[i][j] = 0.0;
			for (int k = 0; k < M; k++)
				r[i][j] += a[i][k]*b[k][j];
		}
	}
	return r;
}

// OPTIMAL
template <typename T, int N, int M> CUGAR_HOST_DEVICE Vector<T,M>& multiply(const Vector<T,N>& v, const Matrix<T,N,M>& m, Vector<T,M>& r)
{
	for (int i = 0; i < M; i++)
	{
		r[i] = 0.0;
		for (int j = 0; j < N; j++)
			r[i] += v[j]*m(j,i);
	}
	return r;
}
// OPTIMAL
template <typename T, int N, int M> CUGAR_HOST_DEVICE Vector<T,N>& multiply(const Matrix<T,N,M>& m, const Vector<T,M>& v, Vector<T,N>& r)
{
	for (int i = 0; i < N; i++)
	{
		r[i] = 0.0;
		for (int j = 0; j < M; j++)
			r[i] += m(i,j)*v[j];
	}
	return r;
}

template <typename T, int N, int M> CUGAR_HOST_DEVICE Matrix<T,M,N> transpose(const Matrix<T,N,M>& m)
{
	Matrix<T,M,N> r;

	return transpose(m, r);
}

template <typename T, int N, int M> CUGAR_HOST_DEVICE Matrix<T,M,N>& transpose(const Matrix<T,N,M>& m, Matrix<T,M,N>& r)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			r[j][i] = m[i][j];

	return r;
}

template <typename T, int N, int M> bool invert(const Matrix<T,N,M>& a, Matrix<T,M,N>& r)
{
    T d = a.det();
	if (d < 1.0e-10f) return false;

	return true;
}

template <typename T>
CUGAR_HOST_DEVICE 
bool invert(const Matrix<T,2,2>& a, Matrix<T,2,2>& r)
{
    const T det = cugar::det(a);
	if (fabsf(det) < 1.0e-10f) return false;

    const T invdet = T(1.0) / det;

    r(0,0) =  a(1,1) * invdet;
    r(0,1) = -a(0,1) * invdet;
    r(1,0) = -a(1,0) * invdet;
    r(1,1) =  a(0,0) * invdet;
	return true;
}

template <typename T>
CUGAR_HOST_DEVICE 
bool invert(const Matrix<T,3,3>& a, Matrix<T,3,3>& r)
{
    const T det = cugar::det(a);
	if (fabsf(det) < 1.0e-10f) return false;

    const T invdet = T(1.0) / det;

    r(0,0) = (a(1,1) * a(2,2) - a(2,1) * a(1,2)) * invdet;
    r(0,1) = (a(0,2) * a(2,1) - a(0,1) * a(2,2)) * invdet;
    r(0,2) = (a(0,1) * a(1,2) - a(0,2) * a(1,1)) * invdet;
    r(1,0) = (a(1,2) * a(2,0) - a(1,0) * a(2,2)) * invdet;
    r(1,1) = (a(0,0) * a(2,2) - a(0,2) * a(2,0)) * invdet;
    r(1,2) = (a(1,0) * a(0,2) - a(0,0) * a(1,2)) * invdet;
    r(2,0) = (a(1,0) * a(2,1) - a(2,0) * a(1,1)) * invdet;
    r(2,1) = (a(2,0) * a(0,1) - a(0,0) * a(2,1)) * invdet;
    r(2,2) = (a(0,0) * a(1,1) - a(1,0) * a(0,1)) * invdet;
    return true;
}

template <typename T>
CUGAR_HOST_DEVICE 
bool invert(const Matrix<T,4,4>& a, Matrix<T,4,4>& r)
{
	T t1, t2, t3, t4, t5, t6;

	// We calculate the adjoint matrix of a, then we divide it by the determinant of a
	// We indicate with Aij the cofactor of a[i][j]:   Aij = (-1)^(i+j)*Det(Tij), where
	// Tij is the minor complementary of a[i][j]

	// First block ([0,0] - [3,1])

	t1 = a(2,2)*a(3,3) - a(2,3)*a(3,2);
	t2 = a(2,1)*a(3,3) - a(2,3)*a(3,1);
	t3 = a(2,1)*a(3,2) - a(2,2)*a(3,1);
	t4 = a(2,0)*a(3,3) - a(2,3)*a(3,0);
	t5 = a(2,0)*a(3,2) - a(2,2)*a(3,0);
	t6 = a(2,0)*a(3,1) - a(2,1)*a(3,0);

	r[0][0] =   a(1,1) * t1 - a(1,2) * t2 + a(1,3) * t3;      // A00
	r[0][1] = -(a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3);     // A10
	r[1][0] = -(a(1,0) * t1 - a(1,2) * t4 + a(1,3) * t5);     // A01
	r[1][1] =   a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5;      // A11
	r[2][0] =   a(1,0) * t2 - a(1,1) * t4 + a(1,3) * t6;      // A02
	r[2][1] = -(a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6);     // A21
	r[3][0] = -(a(1,0) * t3 - a(1,1) * t5 + a(1,2) * t6);     // A03
	r[3][1] =   a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6;      // A13

	// Second block ([0,2] - [3,2])

	t1 = a(1,2)*a(3,3) - a(1,3)*a(3,2);
	t2 = a(1,1)*a(3,3) - a(1,3)*a(3,1);
	t3 = a(1,1)*a(3,2) - a(1,2)*a(3,1);
	t4 = a(1,0)*a(3,3) - a(1,3)*a(3,0);
	t5 = a(1,0)*a(3,2) - a(1,2)*a(3,0);
	t6 = a(1,0)*a(3,1) - a(1,1)*a(3,0);

	r[0][2] =   a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3;      // A20
	r[1][2] = -(a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5);     // A21
	r[2][2] =   a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6;      // A22
	r[3][2] = -(a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6);     // A23

	// Third block ([0,3] - [3,3])

	t1 = a(1,2)*a(2,3) - a(1,3)*a(2,2);
	t2 = a(1,1)*a(2,3) - a(1,3)*a(2,1);
	t3 = a(1,1)*a(2,2) - a(1,2)*a(2,1);
	t4 = a(1,0)*a(2,3) - a(1,3)*a(2,0);
	t5 = a(1,0)*a(2,2) - a(1,2)*a(2,0);
	t6 = a(1,0)*a(2,1) - a(1,1)*a(2,0);

	r[0][3] = -(a(0,1) * t1 - a(0,2) * t2 + a(0,3) * t3);     // A30
	r[1][3] =   a(0,0) * t1 - a(0,2) * t4 + a(0,3) * t5;      // A31
	r[2][3] = -(a(0,0) * t2 - a(0,1) * t4 + a(0,3) * t6);     // A32
	r[3][3] =   a(0,0) * t3 - a(0,1) * t5 + a(0,2) * t6;      // A33

	// We save some time calculating Det(a) this way (now r is adjoint of a)
	// Det(a) = a00 * A00 + a01 * A01 + a02 * A02 + a03 * A03
	T d = a(0,0)*r[0][0] + a(0,1)*r[1][0] + a(0,2)*r[2][0] + a(0,3)*r[3][0];

	if (d == T(0.0)) return false; // Singular matrix => no inverse

	d = T(1.0)/d;

	r *= d;
	return true;
}

template <typename T, int N, int M>
CUGAR_HOST_DEVICE
T det(const Matrix<T,N,M>& m)
{
	return 0.0f;
}

template <typename T>
CUGAR_HOST_DEVICE 
T det(const Matrix<T,2,2>& m)
{
	return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

template <typename T>
CUGAR_HOST_DEVICE 
T det(const Matrix<T,3,3>& m)
{

	return m(0,0) * (m(1,1)*m(2,2) - m(1,2)*m(2,1))
         - m(0,1) * (m(1,0)*m(2,2) - m(1,2)*m(2,0))
         + m(0,2) * (m(1,0)*m(2,1) - m(1,1)*m(2,0));
}

template <typename T>
CUGAR_HOST_DEVICE 
void cholesky(const Matrix<T,2,2>& m, Matrix<T,2,2>& r)
{
    const T a = sqrt(m(0,0));
	const T b = m(0, 1) / a;
    const T c = sqrt(m(1,1) - b*b);
    r(0,0) = a;
    r(0,1) = 0;
    r(1,0) = b;
    r(1,1) = c;
}

//
// Matrix<T,N,M> template <typename T, int N, int M> functions (not members)
//

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE int operator == (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	for (int i = 0; i < N; i++)
	{
		if (a[i] != b[i])
			return 0;
	}
	return 1;
}

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE int operator != (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	return !(a == b);
}

template <typename T, int N, int M> CUGAR_API_CS Matrix<T,N,M> CUGAR_HOST_DEVICE operator  - (const Matrix<T,N,M>& a)
{
	return (Matrix<T,N,M>(a) *= -1.0);
}

template <typename T, int N, int M> CUGAR_API_CS Matrix<T,N,M> CUGAR_HOST_DEVICE operator  + (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r(i,j) = a(i,j) + b(i,j);

	return r;
}

template <typename T, int N, int M> CUGAR_API_CS Matrix<T,N,M> CUGAR_HOST_DEVICE operator  - (const Matrix<T,N,M>& a, const Matrix<T,N,M>& b)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r(i,j) = a(i,j) - b(i,j);

	return r;
}

template <typename T, int N, int M, int Q> CUGAR_API_CS CUGAR_HOST_DEVICE Matrix<T,N,Q> operator * (const Matrix<T,N,M>& a, const Matrix<T,M,Q>& b)
{
	Matrix<T,N,Q> r;

	return multiply(a, b, r);
}

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE Matrix<T,N,M> operator  * (const Matrix<T,N,M>& a, T k)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r(i,j) = a(i,j) * k;

	return r;
}

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE Matrix<T,N,M> operator  * (T k, const Matrix<T,N,M>& a)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r(i,j) = a(i,j) * k;

	return r;
}

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE Vector<T,M> operator * (const Vector<T,N>& v, const Matrix<T,N,M>& m)
{
	Vector<T,M> r;

	return multiply(v, m, r);
}
template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE Vector<T,N> operator * (const Matrix<T,N,M>& m, const Vector<T,M>& v)
{
	Vector<T,N> r;

    return multiply(m, v, r);
}

template <typename T, int N, int M> CUGAR_API_CS CUGAR_HOST_DEVICE Matrix<T,N,M> operator  / (const Matrix<T,N,M>& a, T k)
{
	Matrix<T,N,M> r;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r(i,j) = a(i,j) / k;

	return r;
}

template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> translate(const Vector<T,3>& vec)
{
    Matrix<T,4,4> m( T(0) );

    m(0,0) = m(1,1) = m(2,2) = m(3,3) = 1.0f;

    m(0,3) = vec.x;
    m(1,3) = vec.y;
    m(2,3) = vec.z;

    return m;
}

/// build a 3d scaling matrix
template <typename T>
CUGAR_HOST_DEVICE Matrix<T, 4, 4> scale(const Vector<T, 3>& vec)
{
	Matrix<T,4,4> m( T(0) );

	m(0,0) = vec[0];
	m(1,1) = vec[1];
	m(2,2) = vec[2];
	m(3,3) = 1.0f;

	return m;
}

template <typename T>
Matrix<T,4,4> perspective(T fovy, T aspect, T zNear, T zFar)
{
    Matrix<T,4,4> m( T(0) );

    T f = T(1) / std::tan(fovy / T(2));

    m(0,0) = f / aspect;
    m(1,1) = f;
    m(2,2) = (zFar + zNear) / (zNear - zFar);
    m(2,3) = T(2) * zFar * zNear / (zNear - zFar);
    m(3,2) = T(-1);
    m(3,3) = T(0);

    return m;
}
template <typename T>
Matrix<T,4,4> look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up, bool flip_sign)
{
    Vector<T,3> f = normalize(center - eye);
    Vector<T,3> r = normalize(cross(f, up));
    Vector<T,3> u = cross(r, f);

    Matrix<T,4,4> m(T(0));
    m(0,0) = r.x;
    m(0,1) = r.y;
    m(0,2) = r.z;
    m(1,0) = u.x;
    m(1,1) = u.y;
    m(1,2) = u.z;
    m(2,0) = (flip_sign ? -1.0f : 1.0f) * f.x;
    m(2,1) = (flip_sign ? -1.0f : 1.0f) * f.y;
    m(2,2) = (flip_sign ? -1.0f : 1.0f) * f.z;
	m(3,0) = eye.x;
	m(3,1) = eye.y;
	m(3,2) = eye.z;
	m(3,3) = 1.0f;

    return m;
}
template <typename T>
Matrix<T,4,4> inverse_look_at(const Vector<T,3>& eye, const Vector<T,3>& center, const Vector<T,3>& up, bool flip_sign)
{
	Vector<T,3> f = normalize(center - eye);
	Vector<T,3> r = normalize(cross(f, up));
	Vector<T,3> u = cross(r, f);

	Matrix<T,4,4> m(T(0));
	m(0,0) = r.x;
	m(1,0) = r.y;
	m(2,0) = r.z;
	m(0,1) = u.x;
	m(1,1) = u.y;
	m(2,1) = u.z;
	m(0,2) = (flip_sign ? -1.0f : 1.0f) * f.x;
	m(1,2) = (flip_sign ? -1.0f : 1.0f) * f.y;
	m(2,2) = (flip_sign ? -1.0f : 1.0f) * f.z;
	m(3,0) = -eye.x;
	m(3,1) = -eye.y;
	m(3,2) = -eye.z;
	m(3,3) = 1.0f;

	return m;
}
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_X(const T q)
{
    Matrix<T,4,4> m;

	const float sin_q = sin(q);
	m[1][1] = m[2][2] = cos(q);
	m[1][2] = -sin_q;
	m[2][1] =  sin_q;
	m[0][0] = m[3][3] = T(1.0f);
	m[0][1] =
	m[0][2] =
	m[0][3] =
	m[1][0] =
	m[1][3] =
	m[2][0] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_Y(const T q)
{
    Matrix<T,4,4> m;

    const float sin_q = sin(q);
	m[0][0] = m[2][2] = cos(q);
	m[2][0] = -sin_q;
	m[0][2] =  sin_q;
	m[1][1] = m[3][3] = T(1.0f);
	m[0][1] =
	m[0][3] =
	m[1][0] =
	m[1][2] =
	m[1][3] =
	m[2][1] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
template <typename T>
CUGAR_HOST_DEVICE Matrix<T,4,4> rotation_around_Z(const T q)
{
    Matrix<T,4,4> m;

    const float sin_q = sin(q);
	m[0][0] = m[1][1] = cos(q);
	m[1][0] =  sin_q;
	m[0][1] = -sin_q;
	m[2][2] = m[3][3] = T(1.0f);
	m[0][2] =
	m[0][3] =
	m[1][2] =
	m[1][3] =
	m[2][0] =
	m[2][1] =
	m[2][3] =
	m[3][0] =
	m[3][1] =
	m[3][2] = T(0.0f);
	return m;
}
// build a 3d rotation around an arbitrary axis
template <typename T>
CUGAR_HOST_DEVICE Matrix<T, 4, 4> rotation_around_axis(const T q, const Vector3f& axis)
{
	const Vector3f tangent  = orthogonal( axis );
	const Vector3f binormal = cross( axis, binormal );

	Matrix4x4f basis_change;
	basis_change[0] = Vector4f( tangent, 0.0f );
	basis_change[1] = Vector4f( binormal, 0.0f );
	basis_change[2] = Vector4f( axis, 0.0f );
	basis_change[3] = Vector4f( 0.0f, 0.0f, 0.0f, 1.0f );

	Matrix4x4f inv_basis_change = transpose( basis_change );
	//invert( basis_change, inv_basis_change );

	const Matrix4x4f rot = rotation_around_Z( q );

	return basis_change * rot * inv_basis_change;
}
CUGAR_HOST_DEVICE inline Vector3f ptrans(const Matrix4x4f& m, const Vector3f& v)
{
	const Vector4f r = m * Vector4f(v,1.0f);
	return Vector3f( r[0], r[1], r[2] );
}
CUGAR_HOST_DEVICE inline Vector3f vtrans(const Matrix4x4f& m, const Vector3f& v)
{
	const Vector4f r = m * Vector4f(v,0.0f);
	return Vector3f( r[0], r[1], r[2] );
}

// get the eigenvalues of a matrix
//
CUGAR_HOST_DEVICE inline Vector2f eigen_values(const Matrix2x2f& m)
{
	const float T = m(0,0) + m(1,1);
	const float D = m(0,0) * m(1,1) - m(1,0) * m(0,1);
	const float S = sqrtf(T*T*0.25f - D);
	const float l1 = T*0.5f + S;
	const float l2 = T*0.5f - S;
	return Vector2f(l1,l2);
}

// get the singular values of a matrix
//
CUGAR_HOST_DEVICE inline Vector2f singular_values(const Matrix2x2f& m)
{
	const float a = m(0,0);
	const float b = m(0,1);
	const float c = m(1,0);
	const float d = m(1,1);

  #if 0
	const float S1 = a*a + b*b + c*c + d*d;
	const float S2 = sqrtf( sqr(a*a + b*b - c*c - d*d) + 4*sqr(a*c + b*d) );
	return Vector2f(
		sqrtf((S1 + S2)*0.5f),
		sqrtf((S1 - S2)*0.5f));
  #else
	const float s00 = sqrtf( sqr(a - d) + sqr(b + c) );
	const float s01 = sqrtf( sqr(a + d) + sqr(b - c) );

	const float s0 = (s00 + s01) / 2;
	const float s1 = fabsf(s0 - s00);

    return Vector2f(s0,s1);
  #endif
}

// get the singular value decomposition of a matrix
//
CUGAR_HOST_DEVICE inline void svd(
	const Matrix2x2f&	m,
	Matrix2x2f&			u,
	Vector2f&			s,
	Matrix2x2f&			v)
{
	const float a = m(0,0);
	const float b = m(0,1);
	const float c = m(1,0);
	const float d = m(1,1);

	s = singular_values(m);

	v(1,0) = (s[0] > s[1]) ? sinf((atan2f(2 * (a*b + c*d), a*a - b*b + c*c - d*d)) / 2) : 0;
    v(0,0) = sqrtf(1 - v(1,0) * v(1,0));
    v(0,1) = -v(1,0);
    v(1,1) =  v(0,0);

	u(0,0) = (s[0] != 0) ? (a * v(0,0) + b * v(1,0)) / s[0] : 1;
    u(1,1) = (s[0] != 0) ? (c * v(0,0) + d * v(1,0)) / s[0] : 0;
    u(0,1) = (s[1] != 0) ? (a * v(0,1) + b * v(1,1)) / s[1] : -u(1,1);
    u(1,1) = (s[1] != 0) ? (c * v(0,1) + d * v(1,1)) / s[1] :  u(0,0);
}

#undef _V
#undef _M
#undef _CM

} // namespace cugar
