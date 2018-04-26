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
 *   \brief Defines the Lambert transmission BSDF
 *
 * This module provides a Lambert transmission BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/spherical/mappings.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///
/// Implements a Lambertian transmitter bsdf.
///
struct LambertTransBsdf
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	LambertTransBsdf(const Vector3f _color) :
		color(_color) {}

	/// evaluate the BRDF f(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoL	= dot(N,L);
		const float NoV	= dot(N,V);

		return NoL * NoV < 0.0f ? color : Vector3f(0.0f);
	}

	/// evaluate the BRDF/pdf ratio f(V,L)/p(V,L) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N,L);
		const float NoV = dot(N,V);

		return NoL * NoV < 0.0f ? color * M_PIf : Vector3f(0.0f);
	}

	/// evaluate the BRDF and the pdf in a single call
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void f_and_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, Vector3f& f, float& p, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		f = NoL * NoV < 0.0f ? color : Vector3f(0.0f);
		p = NoL * NoV < 0.0f ? 1.0f / M_PIf : 0.0f;

		if (measure == kSolidAngle)
			p *= fabsf(NoL);
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		const float p = NoL * NoV < 0.0f ? 1.0f / M_PIf : 0.0f;

		return (measure == kSolidAngle) ? p * fabsf(NoL) : p;
	}

	/// sample L given V and return both the pdf p and the value g = f/p, wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void sample(
		const Vector3f				u,
		const DifferentialGeometry& geometry, 
		const Vector3f				V,
			  Vector3f&				L,
			  Vector3f&				g,
			  float&				p,
			  float&				p_proj) const
	{
		const Vector3f N = geometry.normal_s;

		Vector3f local_L = square_to_cosine_hemisphere(u.xy());
		if (dot(V, N) > 0.0f)
			local_L[2] = -local_L[2];

		L = local_L[0] * geometry.tangent +
			local_L[1] * geometry.binormal +
			local_L[2] * geometry.normal_s;

		g      = color * M_PIf;
		p      = fabsf( local_L[2] ) / M_PIf;
		p_proj = 1.0f / M_PIf;
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
		const Vector3f N = geometry.normal_s;

		if (dot(V, N) * dot(L, N) > 0.0f)
			return false;

		const Vector3f local_L(
			dot(L, geometry.tangent ),
			dot(L, geometry.binormal ),
			fabsf(dot(L, geometry.normal_s )));

		p      = M_PIf / cugar::max( local_L[2], 1.0e-8f );
		p_proj = M_PIf;

		const Vector2f u = cosine_hemisphere_to_square( local_L );
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

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		if (NoV * NoL > 0.0f)
			p = p_proj = 0.0f;
		else
		{
			p      = M_PIf / cugar::max(fabsf(NoL), 1.0e-8f);
			p_proj = M_PIf;
		}
	}

public:
	Vector3f	color;
};

/*! \}
 */

} // namespace cugar
