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

namespace cugar {

#if 0
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>::Vector(const T* v)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        this->operator[](d) = v[d];
}
#endif

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator- (const Vector<T, DIM>& op)
{
	Vector<T, DIM> r;
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = -op[d];
	return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator+ (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = op1[d] + op2[d];
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator+= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1[d] += op2[d];
    return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator- (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = op1[d] - op2[d];
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator-= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1[d] -= op2[d];
    return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator* (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = op1[d] * op2[d];
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator*= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1[d] = op1[d] * op2[d];
    return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator* (const T op1, const Vector<T, DIM>& op2)
{
	Vector<T, DIM> r;
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = op1 * op2[d];
	return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator* (const Vector<T, DIM>& op1, const T op2)
{
	Vector<T, DIM> r;
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = op1[d] * op2;
	return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM>& operator*= (Vector<T, DIM>& op1, const T op2)
{
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		op1[d] = op1[d] * op2;
	return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> operator/ (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = op1[d] / op2[d];
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM>& operator/= (Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1[d] = op1[d] / op2[d];
    return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> operator/ (const Vector<T, DIM>& op1, const T op2)
{
	Vector<T, DIM> r;
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = op1[d] / op2;
	return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM>& operator/= (Vector<T, DIM>& op1, const T op2)
{
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		op1[d] = op1[d] / op2;
	return op1;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> min(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = cugar::min( op1[d], op2[d] );
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> max(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    Vector<T,DIM> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r[d] = cugar::max( op1[d], op2[d] );
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> min(const Vector<T,DIM>& op1, const T op2)
{
	Vector<T,DIM> r;
#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = cugar::min( op1[d], op2 );
	return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T,DIM> max(const Vector<T,DIM>& op1, const T op2)
{
	Vector<T,DIM> r;
#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = cugar::max( op1[d], op2 );
	return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool any(const Vector<T,DIM>& op)
{
    bool r = false;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r || (op[d] != 0);
    return r;
}
template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool all(const Vector<T,DIM>& op)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op[d] != 0);
    return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator== (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op1[d] == op2[d]);
    return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator!= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = false;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r || (op1[d] != op2[d]);
    return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator< (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op1[d] < op2[d]);
    return r;
}


template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator<= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op1[d] <= op2[d]);
    return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator> (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op1[d] > op2[d]);
    return r;
}


template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool operator>= (const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
    bool r = true;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op1[d] >= op2[d]);
    return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T dot(const Vector<T, DIM>& op1, const Vector<T, DIM>& op2)
{
	T r = 0.0f;
	for (uint32 d = 0; d < DIM; ++d)
		r += op1[d] * op2[d];
	return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T square_length(const Vector<T, DIM>& op)
{
	return dot(op, op);
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T length(const Vector<T, DIM>& op)
{
	return sqrt( dot(op, op) );
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> normalize(const Vector<T, DIM>& op)
{
	const float l = length(op);
	return l > T(0) ? op / l : op;
}

template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3u> cross(const Vector<T, 3u>& op1, const Vector<T, 3u>& op2)
{
	return Vector<T, 3u>(
		op1[1] * op2[2] - op1[2] * op2[1],
		op1[2] * op2[0] - op1[0] * op2[2],
		op1[0] * op2[1] - op1[1] * op2[0]);
}
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> reflect(const Vector<T, 3> I, const Vector<T, 3> N)
{
	return I - T(2.0)*dot(I, N)*N;
}
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> refract(const Vector<T, 3> I, const Vector<T, 3> N, const float eta)
{
	const float N_dot_I = dot(N, I);
	const float cos_theta_t2 = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
	if (cos_theta_t2 < 0.f)
		return reflect(I,N); // total internal reflection
	else
		return eta * I - (eta * N_dot_I + sqrtf(cos_theta_t2)) * N;
}
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> refraction_normal(const Vector<T, 3> I, const Vector<T, 3> T, const float eta)
{
	return normalize(T - I * eta);
}
template <typename T>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
Vector<T, 3> faceforward(const Vector<T, 3> N, const Vector<T, 3> I)
{
	return dot(I,N) > 0.0f ? N : -N;
}
template <typename T>
CUGAR_HOST_DEVICE
Vector<T, 3> orthogonal(const Vector<T, 3> v)
{
	if (v[0] * v[0] < v[1] * v[1])
	{
		if (v[0] * v[0] < v[2] * v[2])
		{
			// r = -cross( v, (1,0,0) )
			return Vector<T, 3>(0.0f, -v[2], v[1]);
		}
		else
		{
			// r = -cross( v, (0,0,1) )
			return Vector<T, 3>(-v[1], v[0], 0.0);
		}
	}
	else
	{
		if (v[1] * v[1] < v[2] * v[2])
		{
			// r = -cross( v, (0,1,0) )
			return Vector<T, 3>(v[2], 0.0, -v[0]);
		}
		else
		{
			// r = -cross( v, (0,0,1) )
			return Vector<T, 3>(-v[1], v[0], 0.0);
		}
	}
}

template <typename T>
CUGAR_HOST_DEVICE
uint32 pack_vector(const Vector<T, 2> v, const uint32 n_bits_comp)
{
	const uint32 MAX = (1u << n_bits_comp) - 1u;
	const uint32 x = cugar::quantize(v[0], MAX);
	const uint32 y = cugar::quantize(v[1], MAX);
	return x | (y << n_bits_comp);
}

template <typename T>
CUGAR_HOST_DEVICE
Vector<T, 2> unpack_vector(const uint32 u, const uint32 n_bits_comp)
{
	const uint32 MAX = (1u << n_bits_comp) - 1u;
	return Vector<T, 2>(
		T(u & MAX)			/ T(MAX),
		T(u >> n_bits_comp) / T(MAX) );
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector<T, DIM> mod(const Vector<T, DIM>& op, const T m)
{
	Vector<T, DIM> r;
	#pragma unroll
	for (uint32 d = 0; d < DIM; ++d)
		r[d] = mod(op[d],m);
	return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T max_comp(const Vector<T, DIM>& op)
{
	T r = op[0];
	#pragma unroll
	for (uint32 d = 1; d < DIM; ++d)
		r = max(r, op[d]);
	return r;
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
T min_comp(const Vector<T, DIM>& op)
{
	T r = op[0];
#pragma unroll
	for (uint32 d = 1; d < DIM; ++d)
		r = min(r, op[d]);
	return r;
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float max_comp(const Vector3f& op)		// Vector3f specialization
{
	return max3(op.x, op.y, op.z);
}
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float min_comp(const Vector3f& op)		// Vector3f specialization
{
	return min3(op.x, op.y, op.z);
}

template <typename T, uint32 DIM>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool is_finite(const Vector<T, DIM>& op)
{
	for (uint32 i = 0; i < DIM; ++i)
	{
		if (is_finite(op[i]) == false)
			return false;
	}
	return true;
}

CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool is_finite(const Vector3f& op) { return is_finite(op.x) && is_finite(op.y) && is_finite(op.z); }

} // namespace cugar
