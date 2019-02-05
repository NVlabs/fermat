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

#include <cugar/linalg/matrix.h>

namespace cugar {

///@addtogroup LinalgModule
///@{

///@addtogroup MatricesModule Matrices
///@{

/// Return the scale factor for a small linear system
///
template <typename T, int N>
Vector<T,N> scale_factor(const Matrix<T,N,N>& A)
{
	int i,j;
    Vector<T,N> S;
	
	for (i = 0; i < N; ++i)
	{
		S[i] = fabs(A[i][0]);
		for (j = 1; j < N; ++j)
		{
            if (S[i] < fabs(A[i][j]))
                S[i] = fabs(A[i][j]);
		}
	}
	return S;
}

/// Partial pivoting for a small linear system
///
template <typename T, int N>
bool pivot_partial(Matrix<T,N,N>& A, const Vector<T,N>& S, Vector<T,N>& B)
{
	int i,j;
	T temp;
	for(j = 0; j < N; ++j)
	{
		for(i = j + 1; i < N; ++i)
		{
			if (S[i] == 0)
			{
				if(B[i] == 0)
                    return false; // System doesn´t have a unique solution
				else 
                    return false; // System is incosistent
			}
			if (fabs(A[i][j]/S[i]) > fabs(A[j][j]/S[j]))
			{
                std::swap( A[i], A[j] );
				temp = B[i];
				B[i] = B[j];
				B[j] = temp;
			}
		}
		
		if (A[j][j] == 0)
            return false; // System is singular
	}
	return true;
}

/// Forward elimination for a small linear system
///
template <typename T, int N>
bool forward_elimination(Matrix<T,N,N>& A, Vector<T,N>& B)
{
	int i,j,k;
	T m;
	
	for (k = 0; k < N-1; ++k)
	{
		for (i = k + 1; i < N; ++i)
		{
			m = A[i][k] / A[k][k];
			for (j = k + 1; j < N; ++j)
			{
				A[i][j] -= m * A[k][j];
				if (i == j && A[i][j] == 0)
                    return false; // Singular system
			}
			B[i] -= m * B[k];
		}
	}
	return true;		
}

/// Back substitution for a small linear system
///
template <typename T, int N>
Vector<T,N> back_substitution(Matrix<T,N,N>& A, Vector<T,N>& B)
{
	int i,j;
	T sum;
	Vector<T,N> X;
	X[N-1] = B[N-1]/A[N-1][N-1];
	for (i = N - 2; i >= 0; --i)
	{
		sum = 0;
		for (j = i + 1; j < N; ++j)
	        sum += A[i][j] * X[j];
		X[i] = (B[i] - sum) / A[i][i];
	}
	return X;
}

/// Gaussian elimination of a small linear system
///
template <typename T, int N>
Vector<T,N> gaussian_elimination(Matrix<T,N,N> A, Vector<T,N> B)
{
	const Vector<T,N> S = scale_factor(A);
	pivot_partial(A,S,B);
	forward_elimination(A,B);
	return back_substitution(A,B);
}

///@} MatricesModule
///@} LinalgModule

} // namespace cugar
