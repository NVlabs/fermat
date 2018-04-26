/*
 * Copyright (c) 2010-2016, NVIDIA Corporation
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
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/spherical/mappings.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///
/// Implements a weighted blending between two BSDFs.
///
template <
	typename TBsdf1,
	typename TBsdf2>
struct BlendBsdf
{
	/// constructor: this method accepts weights _w1 and _w2 for the corresponding BSDF components;
	/// the weights must be such that their sum is equal to or less than 1. If it is less, some energy
	/// will be absorbed.
	///
	/// \param _w1			the weight of the first BSDF
	/// \param _w2			the weight of the second BSDF
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	BlendBsdf(
		const float  _w1    = 0.5f,
		const float  _w2    = 0.5f,
		const TBsdf1 _bsdf1 = TBsdf1(),
		const TBsdf2 _bsdf2 = TBsdf2()) :
		w1(_w1),
		w2(_w2),
		bsdf1(_bsdf1),
		bsdf2(_bsdf2)
	{}

	/// evaluate the BSDF f(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		return
			w1 * bsdf1.f(geometry,V,L) +
			w2 * bsdf2.f(geometry,V,L);
	}

	/// evaluate the BSDF/pdf ratio f(V,L)/p(V,L) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		return
			w1 * bsdf1.f_over_p(geometry,V,L) +
			w2 * bsdf2.f_over_p(geometry,V,L);
	}

	/// evaluate the BSDF and the pdf in a single call
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void f_and_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, Vector3f& f, float& p, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		Vector3f f1, f2;
		float    p1, p2;

		bsdf1.f_over_p( geometry, V, L, f1, p1, measure );
		bsdf2.f_over_p( geometry, V, L, f2, p2, measure );

		f = w1 * f1 + w2 * f2;
		p = w1 * p1 + w2 * p2;
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		return
			w1 * bsdf1.p(geometry,V,L) +
			w2 * bsdf2.p(geometry,V,L);
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
		if (u.z < w1)
		{
			// sample the first component
			bsdf1.sample( Vector3f(u.x,u.y,u.z/w1), geometry, V, L, g, p, p_proj );

			g      /= w1;
			p      *= w1;
			p_proj *= w1;
		}
		else if (u.z < w1 + w2)
		{
			// sample the second component
			bsdf2.sample( Vector3f(u.x,u.y,(u.z-w1)/w2), geometry, V, L, g, p, p_proj );

			g      /= w2;
			p      *= w2;
			p_proj *= w2;
		}
		else
		{
			// absorption
			g      = Vector3f(0.0f);
			p      = (1 - w1 - w2);
			p_proj = (1 - w1 - w2);
		}
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
		// TODO
		return false;
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
		p_proj = 1.0f / this->p( geometry, V, L, u, kProjectedSolidAngle );
		p      = fabsf(dot(L, geometry.normal_s)) * p_proj;
	}

public:
	float	w1, w2;
	TBsdf1	bsdf1;
	TBsdf2	bsdf2;
};

/*! \}
 */

} // namespace cugar
