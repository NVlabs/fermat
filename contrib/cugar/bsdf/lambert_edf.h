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

/*! \file lambert.h
*   \brief Defines the Lambert BSDF
*
* This module provides a Lambert BSDF implementation
*/

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/spherical/mappings.h>
#include <cugar/bsdf/differential_geometry.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
*  \{
*/

///
/// Implements a Lambertian edf.
///
struct LambertEdf
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	LambertEdf(const Vector3f _color) :
		color(_color) {}

	/// evaluate the EDF f(in,out)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f in, const Vector3f out) const
	{
		const float NoL = dot(geometry.normal_s, out);

		return NoL > 0.0f ? color : Vector3f(0.0f);
	}

	/// evaluate the EDF/pdf ratio f(in,out)/p(in,out) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f in, const Vector3f out) const
	{
		const float NoL = dot(geometry.normal_s, out);

		return NoL > 0.0f ? color * float(M_PI) : Vector3f(0.0f);
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f in, const Vector3f out, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		return (measure == kProjectedSolidAngle) ?
			1.0f / float(M_PI) :
			fabsf(dot(geometry.normal_s, out)) / float(M_PI);
	}

	/// sample out given in and return both the pdf p and the value g = f/p, wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void sample(
		const Vector2f  u,
		const DifferentialGeometry& geometry,
		const Vector3f  in,
		Vector3f&		out,
		Vector3f&		g,
		float&			p,
		float&			p_proj) const
	{
		Vector3f local_dir = square_to_cosine_hemisphere(u);

		out = local_dir[0] * geometry.tangent +
			  local_dir[1] * geometry.binormal +
			  local_dir[2] * geometry.normal_s;
		g = color * float(M_PI);
		p = local_dir[2] / float(M_PI);
		p_proj = 1.0f / float(M_PI);
	}

	/// given V and L, invert the sampling functions used to generate L from V
	///
	template <typename RandomGeneratorT>
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	bool invert(
		const DifferentialGeometry& geometry,
		const Vector3f				V,
		const Vector3f				L,
		RandomGeneratorT&			random,
		Vector3f&					z,
		float&						p,
		float&						p_proj) const
	{
		const Vector3f local_L(
			dot(L, geometry.tangent),
			dot(L, geometry.binormal),
			dot(L, geometry.normal_s));

		const float LN = local_L[2];

		if (LN < 0.0f)
			return false;

		p = float(M_PI) / cugar::max( LN, 1.0e-8f );
		p_proj = float(M_PI);

		const Vector2f u = cosine_hemisphere_to_square(local_L);
		z = Vector3f(u.x, u.y, random.next());
		return true;
	}

	/// given V and L and u, compute the probability of sampling u by inversion of V and L
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void inverse_pdf(
		const DifferentialGeometry& geometry,
		const Vector3f				V,
		const Vector3f				L,
		const Vector3f				u,
		float&						p,
		float&						p_proj) const
	{
		const Vector3f N = geometry.normal_s;

		const float LN = dot(L, N);

		if (LN < 0.0f)
			p = p_proj = 0.0f;
		else
		{
			p = float(M_PI) / cugar::max( LN, 1.0e-8f );
			p_proj = float(M_PI);
		}
	}

public:
	Vector3f	color;
};

/*! \}
*/

} // namespace cugar
