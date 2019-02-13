/*
 * Copyright (c) 2019, NVIDIA Corporation
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

 /*! \file blend.h
 *   \brief Defines a weighted blending between two BSDFs
 *
 * This module provides a weighted blending between two BSDFs
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///\par
/// Evaluate the Fresnel reflection coefficient for dielectrics
///
template <typename VectorType>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
VectorType fresnel_dielectric(
	const float cos_theta_i,
	const float cos_theta_t,
	const VectorType eta)
{
	if (eta == VectorType(1.0f))
		return VectorType(0.0f);

	const VectorType Rs =
		(cos_theta_i - eta * cos_theta_t) /
		(cos_theta_i + eta * cos_theta_t);

	const VectorType Rp =
		(eta * cos_theta_i - cos_theta_t) /
		(eta * cos_theta_i + cos_theta_t);

	return 0.5f * (Rs * Rs + Rp * Rp);
}

///\par
/// Evaluate the Fresnel reflection coefficient for dielectric/conductor interfaces
///
template <typename VectorType>
CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
VectorType fresnel_dielectric_conductor(
	const float cos_theta,
	const VectorType eta,
	const VectorType eta_k)
{
    const float cos_theta2 = cos_theta * cos_theta;
    const VectorType two_eta_cos_theta = 2 * eta * cos_theta;

    const VectorType t0 = eta * eta + eta_k * eta_k;
    const VectorType t1 = t0 * cos_theta2;
    const VectorType Rs = (t0 - two_eta_cos_theta + cos_theta2) / (t0 + two_eta_cos_theta + cos_theta2);
    const VectorType Rp = (t1 - two_eta_cos_theta +          1) / (t1 + two_eta_cos_theta +          1);

    return 0.5* (Rp + Rs);
}

CUGAR_HOST_DEVICE CUGAR_FORCEINLINE
float pow5(const float x)
{
	const float x2 = (x*x);
	return x2 * x2 * x;
}

///\par
/// Evaluate the Fresnel reflection coefficient using Schlick's approximation
///
/// \param eta			incoming IOR / outgoing IOR
///
template <typename VectorType>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
VectorType fresnel_schlick(
		  float			cos_theta_i,
	const float			eta,
	const VectorType	fresnel_base)
{
				cos_theta_i  = saturate( fabsf(cos_theta_i) );	// be paranoid about overflows
	const float cos_theta_t2 = saturate( 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i) );
	if (cos_theta_t2 < 0.0f)
	{
		// TIR
		return 1.0f;
	}
	else
	{
		const float cos_theta = eta > 1.0f ? sqrtf(cos_theta_t2) : cos_theta_i;

		const float Fc = pow5(1 - cos_theta);										// 1 sub, 3 mul
		//const float Fc = exp2( (-5.55473 * cos_theta - 6.98316) * cos_theta );	// 1 mad, 1 mul, 1 exp

		return VectorType(Fc) + (1 - Fc) * fresnel_base;							// 1 add, 3 mad
	}
}

///\par
/// Evaluate the cosine of the refraction angle with the normal - cos(theta_t)
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool refraction_angle(
	const Vector3f	w_i,			// incident vector (= w_i - pointing outwards!)
	const Vector3f	N,				// normal
	const float		eta,			// the relative index of refraction
	const float		cos_theta_i,	// cos(theta_i) = dot(w_i,N)
		  float*	cos_theta_t)	// cos(theta_t) = dot(w_t,N)
{
	if (eta == 1.0f)
	{
		*cos_theta_t = -cos_theta_i;
		return true;
	}

	const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
	if (cos_theta_t2 < 0.0f)
		return false;			// total internal reflection

	*cos_theta_t = (cos_theta_i >= 0.0f ? -1.0f : 1.0f) * sqrtf(cos_theta_t2);
	return true;
}

///\par
/// Evaluate refraction; this function returns false upon Total Internal Reflection (TIR).
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
bool refract(
	const Vector3f	w_i,			// incident vector (= w_i - pointing outwards!)
	const Vector3f	N,				// normal
	const float		cos_theta_i,	// cos(theta_i) = dot(w_i,N)
	const float		eta,			// relative ior
	Vector3f*		out,			// output refraction direction (= w_o)
	float*			F)				// output Fresnel reflection weight
{
	if (eta == 1.0f)
	{
		*out = -w_i;
		*F   = 0.0f;
		return true;
	}

	const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
	if (cos_theta_t2 < 0.0f)
		return false;			// total internal reflection

	const float cos_theta_t = (cos_theta_i >= 0.0f ? -1.0f : 1.0f) * sqrtf(cos_theta_t2);

	*F = fresnel_dielectric( fabsf(cos_theta_i), fabsf(cos_theta_t), eta );

	// refract
	*out = (eta * cos_theta_i + cos_theta_t) * N - eta * w_i;
	return true;
}

///\par
/// Evaluate refraction given the incident and transmitted angles with the normal
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector3f refract(
	const Vector3f	w_i,			// incident vector (= w_i - pointing outwards!)
	const Vector3f	N,				// normal
	const float		eta,			// relative ior
	const float		cos_theta_i,	// cos(theta_i) = dot(w_i,N)
	const float		cos_theta_t)	// cos(theta_t) = dot(w_t,N)
{
	// refract
	return (eta * cos_theta_i + cos_theta_t) * N - eta * w_i;
}

/*! \}
 */

} // namespace cugar
