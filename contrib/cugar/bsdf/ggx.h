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

/*! \file ggx.h
 *   \brief Defines the GGX BSDF
 *
 * This module provides a GGX BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/bsdf/ggx_common.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
*  \{
*/

///
/// Implements the GGX microfacet distribution with the V-cavities shadow-masking model, using the importance sampling scheme
/// described in <i>Algorithm 3</i> of:
///
///   https://hal.inria.fr/hal-00996995v1/document
///
struct GGXVCavityMicrofacetDistribution
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	GGXVCavityMicrofacetDistribution(const float _roughness) :
		roughness(_roughness),
		inv_roughness(1.0f / _roughness) {}

	/// evaluate the pdf of sampling H given V, p(H|V) = p(V,H)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f H) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		const float VoH = fabsf( dot(V, H) );
		const float NoV = fabsf( dot(N, V) );
		const float NoH = fabsf( dot(N, H) );

		const float G1 = fminf(1, 2 * NoH * NoV / VoH);

		const float D = hvd_ggx_eval(inv_alpha, NoH, dot(geometry.tangent, H), dot(geometry.binormal, H));
			  float p = D * G1 * VoH / NoV;
			// NOTE: if reflection of V against H was used to generate a ray, the Jacobian 1 / (4*VoH) would elide
			// the VoH at the numerator, leaving: D * G1 / (4 * NoV) as a solid angle probability, or D * G1 / (4 * NoV * NoL)
			// as a projected solid angle probability

		return (!isfinite(p) || isnan(p)) ? 1.0e8f : p;
	}

	/// sample H given V and return the pdf p
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void sample(
		const Vector3f				u,
		const DifferentialGeometry& geometry,
		const Vector3f				V,
		Vector3f&					H,
		float&						p) const
	{
		//const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);
		const float inv_alpha = inv_roughness;

		const Vector3f N = geometry.normal_s;

		H = hvd_ggx_sample( make_float2(u.x,u.y), inv_alpha );

		H =
			H[0] * geometry.tangent +
			H[1] * geometry.binormal +
			H[2] * geometry.normal_s;

		const Vector3f H_prime = reflect(H, N);

		float VoH       = dot(V, H);
		float VoH_prime = dot(V, H_prime);

		if (u[2] < VoH_prime / (VoH + VoH_prime))
		{
			H   = H_prime;
			VoH = VoH_prime;
		}

		p = this->p(geometry, V, H);
	}

public:
	float roughness;
	float inv_roughness;
};

///
/// Implements the GGX bsdf with the V-cavities shadow-masking model, using the importance sampling scheme
/// described in <i>Algorithm 3</i> of:
///
///   https://hal.inria.fr/hal-00996995v1/document
///
struct GGXBsdf
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	GGXBsdf(const float _roughness) :
		roughness(_roughness),
		inv_roughness(1.0f / _roughness) {}

	/// evaluate the BRDF f(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f H = cugar::normalize(V + L);
		const Vector3f N = geometry.normal_s;

		if (dot(N,L)*dot(N,V) <= 0.0f)
			return cugar::Vector3f(0.0f);

		const float VoH = dot(V, H);
		const float NoL = dot(N, L);
		const float NoV = dot(N, V);
		const float NoH = dot(N, H);

		if (NoL * NoV <= 0.0f || NoH == 0.0f)
			return 0.0f;

		const float D     = hvd_ggx_eval(inv_alpha, NoH, dot(geometry.tangent, H), dot(geometry.binormal, H));
		const float G2    = fminf(1, fminf(2 * NoH * NoV / VoH, 2 * NoH * NoL / VoH));
		const float denom = (4 * NoV * NoL /** NoH*/);
		const float p     = D * G2 / denom;

		return (!isfinite(p) || isnan(p)) ? 1.0e8f : p;
	}

	/// evaluate the BRDF/pdf ratio f(V,L)/p(V,L) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f H = cugar::normalize(V + L);
		const Vector3f N = geometry.normal_s;

		if (dot(N, L)*dot(N, V) <= 0.0f)
			return cugar::Vector3f(0.0f);

		const float VoH = dot(V, H);
		const float NoL = dot(N, L);
		const float NoV = dot(N, V);
		const float NoH = dot(N, H);

		const float G2  = fminf(1, fminf(2 * NoH * NoV / VoH, 2 * NoH * NoL / VoH));
		const float G1  = fminf(1, 2 * NoH * NoV / VoH);

		return cugar::Vector3f(G2 / G1);
	}

	/// evaluate the BRDF and the pdf in a single call
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void f_and_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, Vector3f& f, float& p, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f H = cugar::normalize(V + L);
		const Vector3f N = geometry.normal_s;

		const float VoH = dot(V, H);
		const float NoL = dot(N, L);
		const float NoV = dot(N, V);
		const float NoH = dot(N, H);

		const float D     = hvd_ggx_eval(inv_alpha, NoH, dot(geometry.tangent, H), dot(geometry.binormal, H));
		const float G2    = fminf(1, fminf(2 * NoH * NoV / VoH, 2 * NoH * NoL / VoH));
		const float G1    = fminf(1, 2 * NoH * NoV / VoH);
		const float denom = (4 * NoV * NoL /** NoH*/);
		
		p = D * G1 / denom;
		p = (!isfinite(p) || isnan(p)) ? 1.0e8f : p;

		float f_s = D * G2 / denom;
			  f_s = (!isfinite(f_s) || isnan(f_s)) ? 1.0e8f : f_s;

		f = (NoL * NoV <= 0.0f || NoH == 0.0f) ? Vector3f(0.0f) : Vector3f(f_s);

		if (measure == kSolidAngle)
			p *= fabsf( NoL );
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f H = cugar::normalize(V + L);
		const Vector3f N = geometry.normal_s;

		const float VoH = dot(V, H);
		const float NoL = dot(N, L);
		const float NoV = dot(N, V);
		const float NoH = dot(N, H);

		const float D     = hvd_ggx_eval(inv_alpha, NoH, dot(geometry.tangent, H), dot(geometry.binormal, H));
		const float G1    = fminf(1, 2 * NoH * NoV / VoH);
		const float denom = (4 * NoV * NoL /** NoH*/);
			  float p     = D * G1 / denom;

		if (measure == kSolidAngle)
			p *= fabsf(NoL);

		return (!isfinite(p) || isnan(p)) ? 1.0e8f : p;
	}

	/// sample L given V and return both the pdf p and the value g = f/p
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void sample(
		const Vector3f				u,
		const DifferentialGeometry& geometry,
		const Vector3f				V,
		Vector3f&					L,
		Vector3f&					g,
		float&						p,
		float&						p_proj) const
	{
		//const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);
		const float inv_alpha = inv_roughness;

		const Vector3f N = geometry.normal_s;

		Vector3f H = hvd_ggx_sample( make_float2(u.x,u.y), inv_alpha );

		H =
			H[0] * geometry.tangent +
			H[1] * geometry.binormal +
			H[2] * geometry.normal_s;

		const Vector3f H_prime = reflect(H, N);

		float VoH       = dot(V, H);
		float VoH_prime = dot(V, H_prime);

		if (u[2] < VoH_prime / (VoH + VoH_prime))
		{
			H   = H_prime;
			VoH = VoH_prime;
		}

		L = 2 * dot(V, H) * H - V;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);
		const float NoH = dot(N, H);

		const float D     = hvd_ggx_eval(make_float2(inv_alpha,inv_alpha), NoH, dot(geometry.tangent, H), dot(geometry.binormal, H));
		const float G2    = fminf(1, fminf(2 * NoH * NoV / VoH, 2 * NoH * NoL / VoH));
		const float G1    = fminf(1, 2 * NoH * NoV / VoH);
		const float denom = (4 * NoV * NoL /** NoH*/);
			  
		g      = cugar::Vector3f(1.0f) * (dot(L,N)*dot(V,N) > 0.0f ? (G2 / G1) : 0.0f);
		p_proj = D * G1 / denom;
		p      = p_proj * fabsf(dot(N, L));

		if (!isfinite(p) || isnan(p)) p = 1.0e8f;
		if (!isfinite(p_proj) || isnan(p_proj)) p_proj = 1.0e8f;
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
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		//if (dot(N, L)*dot(N, V) <= 0.0f)
		//	return false;

		const Vector3f H     = cugar::normalize(V + L);
		const Vector3f H_alt = reflect(H, N);
		//printf("\n  H: %f, %f, %f\n", H.x, H.y, H.z);
		//printf("  H_alt: %f, %f, %f\n", H_alt.x, H_alt.y, H_alt.z);

		// Now, there's two possibilities. Either:
		//  1) H was originally sampled, and L directly generated from it, or
		//  2) H_alt was originally sampled, and H was obtained as H = H_alt' = reflect(H_alt,N);
		//
		// 1) can only happen if H.N > 0, whereas 2) happens otherwise.
		//
		// If the case had been 1, the probability of staying with H was V.H / (V.H + V.H_alt)
		// If the case had been 2, the probability of switching from H_alt to H was, again, V.H / (V.H + V.H_alt)
		// These are needed to compute the random number that makes sure the eventual switch happens appropriately.

		float VoH     = dot(V, H);
		float VoH_alt = dot(V, H_alt);

		//printf("  rel probs: %f, %f : (%f, %f)\n", VoH / (VoH_alt + VoH), VoH_alt / (VoH_alt + VoH), VoH, VoH_alt);

		const float NoH = dot(H, N); // If NoH < 0.0f, the only possibility is that H_alt must have been sampled
		//printf("  NoH: %f\n", NoH);

		const cugar::Vector3f H_orig = NoH > 0.0f ? H : H_alt;
		const float2          H_prob = NoH > 0.0f ?
			make_float2( fmaxf( 0.0f, fminf( 1.0f, VoH_alt / (VoH_alt + VoH) ) ), 1.0f ) :
			make_float2( 0.0f, fmaxf( 0.0f, fminf( 1.0f, VoH / (VoH_alt + VoH) ) ) );
		
		const Vector3f local_H(
			dot(H_orig, geometry.tangent),
			dot(H_orig, geometry.binormal),
			dot(H_orig, geometry.normal_s));
		//printf("  local_H: %f, %f, %f - H_prob: [%f,%f]\n", local_H.x, local_H.y, local_H.z, H_prob.x, H_prob.y);

		const float2 u = hvd_ggx_invert( local_H, inv_alpha );
		//const Vector3f local_H2 = hvd_ggx_sample(u, inv_alpha);
		//printf("  local_H2: %f, %f, %f - u(%f,%f)\n", local_H2.x, local_H2.y, local_H2.z, u.x, u.y);

		p_proj = 1.0f / cugar::max(this->p(geometry, V, L, kProjectedSolidAngle), 1.0e-8f);
		p      = p_proj / cugar::max(fabsf(dot(L, N)), 1.0e-8f);

		z = Vector3f( u.x, u.y, H_prob.x + random.next() * (H_prob.y - H_prob.x) );
		return H_prob.y - H_prob.x > 0.0f ? true : false; // NOTE: inversion actually works, but the resulting pdf contains a Dirac delta
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

		p_proj = 1.0f / cugar::max(this->p(geometry, V, L, kProjectedSolidAngle), 1.0e-8f);
		p      = p_proj / cugar::max(fabsf(dot(L, N)), 1.0e-8f);
	}

public:
	float roughness;
	float inv_roughness;
};

/*! \}
 */

} // namespace cugar
