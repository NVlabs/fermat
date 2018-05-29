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
 *   \brief Defines the GGXSmith BSDF
 *
 * This module provides a GGXSmith BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/bsdf/ggx_common.h>

#define USE_APPROX_SMITH

namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///
/// Implements the GGX microfacet distribution with the Smith shadow-masking model, as described in:
///
///   https://hal.archives-ouvertes.fr/hal-01509746/document
///
struct GGXSmithMicrofacetDistribution
{
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	GGXSmithMicrofacetDistribution(const float _roughness) :
		roughness(_roughness),
		inv_roughness(1.0f / _roughness) {}

	// Smith G1(V) term
	//
#if 1
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float SmithG1V(const float NoV) const
	{
		const float a = roughness;
		const float a2 = a*a;
		const float G_V = NoV + sqrtf((NoV - NoV * a2) * NoV + a2);
		return 2.0f * NoV / G_V;
	}
#else
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float SmithG1V(const float NoV) const
	{
		const float CosThetaV2 = NoV*NoV;
		const float SinThetaV2 = 1 - CosThetaV2;
		const float TanThetaV2 = fabsf(SinThetaV2 <= 0.0f ? 0.0f : SinThetaV2 / CosThetaV2);
		// Perpendicular incidence -- no shadowing/masking
		if (TanThetaV2 == 0.0f)
			return 1.0f;

		const float alpha = roughness /** roughness*/;
		const float R2 = alpha * alpha * TanThetaV2;
		return 2.0f / (1.0f + sqrtf(1.0f + R2));
	}
#endif

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

		const float G1 = SmithG1V(NoV);

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
		const float2 alpha     = make_float2(roughness, roughness);
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f V_local(
			dot(V, geometry.tangent),
			dot(V, geometry.binormal),
			dot(V, geometry.normal_s) );

		H = vndf_ggx_smith_sample( make_float2(u.x,u.y), alpha, V_local);

		H =
			H.x * geometry.tangent +
			H.y * geometry.binormal +
			H.z * geometry.normal_s;

		p = this->p(geometry, V, H);
	}

public:
	float roughness;
	float inv_roughness;
};

///
/// Implements the GGX bsdf with the Smith height-correlated shadow-masking model,
/// as described in:
///
///   https://hal.archives-ouvertes.fr/hal-01509746/document
///
struct GGXSmithBsdf
{
	/// Constructor
	///
	/// \param _roughness		the surface roughness
	/// \param _transmission	whether this is a transmissive or reflective BSDF
	/// \param _int_ior			the "interior" ior, on the opposite side of the surface normal
	/// \param _ext_ior			the "exterior" ior, on the same side of the surface normal
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	GGXSmithBsdf(const float _roughness, bool _transmission = false, float _int_ior = 1.0f, float _ext_ior = 1.0f) :
		roughness(_roughness),
		inv_roughness(1.0f / _roughness),
		int_ior(_transmission ? _int_ior : 0.0f),
		ext_ior(_transmission ? _ext_ior : 0.0f) {}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	bool is_transmissive() const { return int_ior != 0.0f; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	bool is_reflective() const { return int_ior == 0.0f; }

	/// fetch the exterior/interior IOR ratio
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float get_eta(const float NoV) const { return NoV >= 0.0f ? ext_ior / int_ior : int_ior / ext_ior; }

	/// fetch the interior/exterior IOR ratio
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float get_inv_eta(const float NoV) const { return NoV >= 0.0f ? int_ior / ext_ior : ext_ior / int_ior; }

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static float clamp_inf(const float p) { return (!isfinite(p) || isnan(p)) ? 1.0e8f : p; }

	// Appoximation of joint Smith term for GGX, pre-divided by the BRDF denominator (4 * NoV * NoL)
	// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float PredividedSmithJointApprox(const float NoV, const float NoL) const
	{
		const float a = roughness /** roughness*/;
		const float Vis_SmithV = NoL * (NoV * (1 - a) + a);
		const float Vis_SmithL = NoV * (NoL * (1 - a) + a);
		// Note: will generate NaNs with Roughness = 0.  MinRoughness is used to prevent this
		return 0.5f * 1.0f / (Vis_SmithV + Vis_SmithL);
	}
	// Height-Correlated Smith Joint Masking-Shadowing Function for GGX, pre-divided by the BRDF denominator (4 * NoV * NoL)
	// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float PredividedSmithJoint(const float NoV, const float NoL) const
	{
		const float a = roughness /** roughness*/;
		const float a2 = a*a;
		const float G_V = NoV + sqrtf((NoV - NoV * a2) * NoV + a2);
		const float G_L = NoL + sqrtf((NoL - NoL * a2) * NoL + a2);
		return 1.0f / (G_V * G_L);
	}

	// Smith G1(V) term, pre-divided by the BRDF denominator (4 * NoV * NoL)
	//
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float PredividedSmithG1V(const float NoV, const float NoL) const
	{
		const float a = roughness /** roughness*/;
		const float a2 = a*a;
		const float G_V = NoV + sqrtf((NoV - NoV * a2) * NoV + a2);
		return 0.5f / (G_V * NoL);
	}

	// Height-Correlated Smith Joint Masking-Shadowing Function for GGX
	// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float SmithJoint(const float NoV, const float NoL) const
	{
		// masking
		//const float a_V = inv_roughness / tanf(acosf(NoV));in
		//const float LambdaV = (NoV < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_V / a_V)) : 0.0f;
		const float CosThetaV2 = NoV*NoV;
		const float SinThetaV2 = 1 - CosThetaV2;
		const float TanThetaV2 = fabsf(SinThetaV2 <= 0.0f ? 0.0f : SinThetaV2 / CosThetaV2);
		const float a_V2 = inv_roughness * inv_roughness / TanThetaV2;
		const float LambdaV = (NoV < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_V2)) : 0.0f;

		// shadowing
		//const float a_L = inv_roughness / tanf(acosf(NoL));
		//const float LambdaL = (NoL < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_L / a_L)) : 0.0f;
		const float CosThetaL2 = NoL*NoL;
		const float SinThetaL2 = 1 - CosThetaL2;
		const float TanThetaL2 = fabsf(SinThetaL2 <= 0.0f ? 0.0f : SinThetaL2 / CosThetaL2);
		const float a_L2 = inv_roughness * inv_roughness / TanThetaL2;
		const float LambdaL = (NoL < 1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a_L2)) : 0.0f;

		// height-correlated Smith shadow-masking term (http://jcgt.org/published/0003/02/03/paper.pdf, page 91)
		return 1.0f / (1.0f + LambdaV + LambdaL);
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float SmithG1(const float NoV) const
	{
		const float CosThetaV2 = NoV*NoV;
		const float SinThetaV2 = 1 - CosThetaV2;
		const float TanThetaV2 = fabsf(SinThetaV2 <= 0.0f ? 0.0f : SinThetaV2 / CosThetaV2);
		// Perpendicular incidence -- no shadowing/masking
		if (TanThetaV2 == 0.0f)
			return 1.0f;

		const float alpha = roughness /** roughness*/;
		const float R2 = alpha * alpha * TanThetaV2;
		return 2.0f / (1.0f + sqrtf(1.0f + R2));
	}

	/// evaluate the BRDF f(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		const float eta     = get_eta(NoV);
		const float inv_eta = get_inv_eta(NoV);

		const Vector3f H = microfacet(V,L,N,inv_eta);

		const float NoH = dot(N, H);

		const float transmission_sign = is_transmissive() ? -1.0f : 1.0f;

		// check whether there is energy exchange between different sides of the surface
		if (transmission_sign * NoL * NoV <= 0.0f || NoH == 0.0f)
			return cugar::Vector3f(0.0f);

		const float D = hvd_ggx_eval(inv_alpha, fabsf( NoH ), dot(geometry.tangent, H), dot(geometry.binormal, H));
		//const float denom = (4 * NoV * NoL);

      #if defined(USE_APPROX_SMITH)
		const float G = PredividedSmithJointApprox(fabsf(NoV), fabsf(NoL));
	  #else
		const float G = PredividedSmithJoint(fabsf(NoV), fabsf(NoL));
	  #endif

		if (!is_transmissive())
			return clamp_inf( G * D );
		else
		{
			// compute whether this is a ray for which we have total internal reflection,
			// i.e. if cos_theta_t2 <= 0, and in that case return zero.
			const float VoH = dot(V, H);
			const float LoH = dot(L, H);
			const float cos_theta_i = fabsf(VoH);
			const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
			if (cos_theta_t2 < 0.0f)
				return 0.0f;

			// compute dwo_dh: equation 17 in Microfacet Models for Refraction through Rough Surfaces, Walter et al.
			const float sqrtDenom = VoH + inv_eta * LoH;
			const float factor = 4 * inv_eta * inv_eta
				* VoH * LoH /
				(sqrtDenom * sqrtDenom);
				// NOTE: the actual formula has no 4 factor at the numerator, but it contains NoV * NoL at
				// the denominator, which is already included in our G term above. The additional multiplication
				// by 4 is needed to account for the predivision.
			return clamp_inf(factor * G * D);
		}
	}

	/// evaluate the BRDF/pdf ratio f(V,L)/p(V,L) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		const float inv_eta = get_inv_eta(NoV);

		const Vector3f H = microfacet(V,L,N,inv_eta);

		const float NoH = dot(N, H);

		const float transmission_sign = is_transmissive() ? -1.0f : 1.0f;

		// check whether there is energy exchange between different sides of the surface
		if (transmission_sign * NoL * NoV <= 0.0f || NoH == 0.0f)
			return cugar::Vector3f(0.0f);

		const float G1 = PredividedSmithG1V(fabsf(NoV), fabsf(NoL));

	  #if defined(USE_APPROX_SMITH)
		const float G = PredividedSmithJointApprox(fabsf(NoV), fabsf(NoL));
	  #else
		const float G  = PredividedSmithJoint(fabsf(NoV), fabsf(NoL));
	  #endif

		return cugar::Vector3f(clamp_inf( G / G1 ));
	}

	/// evaluate the BRDF and the pdf in a single call
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void f_and_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, Vector3f& f, float& p, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		// fetch the interior/exterior IOR ratio
		const float eta     = get_eta(NoV);
		const float inv_eta = get_inv_eta(NoV);

		const Vector3f H = microfacet(V,L,N,inv_eta);

		const float NoH = dot(N, H);

		const float transmission_sign = is_transmissive() ? -1.0f : 1.0f;

		// check whether there is energy exchange between different sides of the surface
		if (transmission_sign * NoL * NoV <= 0.0f || NoH == 0.0f)
		{
			p = 0.0f;
			f = Vector3f(0.0f);
			return;
		}

		const float D = hvd_ggx_eval(inv_alpha, fabsf( NoH ), dot(geometry.tangent, H), dot(geometry.binormal, H));

	  #if defined(USE_APPROX_SMITH)
		const float G = PredividedSmithJointApprox(fabsf(NoV), fabsf(NoL));
	  #else
		const float G = PredividedSmithJoint(fabsf(NoV), fabsf(NoL));
	  #endif

		const float G1 = PredividedSmithG1V(fabsf(NoV), fabsf(NoL));

		f = clamp_inf( G * D );
		p = clamp_inf( G1 * D );

		if (is_transmissive())
		{
			// compute whether this is a ray for which we have total internal reflection,
			// i.e. if cos_theta_t2 <= 0, and in that case return zero.
			const float VoH = dot(V, H);
			const float LoH = dot(L, H);
			const float cos_theta_i = fabsf(VoH);
			const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
			if (cos_theta_t2 < 0.0f)
			{
				p = 0.0f;
				f = Vector3f(0.0f);
				return;
			}

			// compute dwo_dh: equation 17 in Microfacet Models for Refraction through Rough Surfaces, Walter et al.
			const float sqrtDenom = VoH + inv_eta * LoH;
			const float factor = 4 * inv_eta * inv_eta
				* VoH * LoH /
				(sqrtDenom * sqrtDenom);
				// NOTE: the actual formula has no 4 factor at the numerator, but it contains NoV * NoL at
				// the denominator, which is already included in our G term above. The additional multiplication
				// by 4 is needed to account for the predivision.
			f *= clamp_inf(factor);
			p *= clamp_inf(factor);
		}

		if (measure == kSolidAngle)
			p *= fabsf( NoL );
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		const float NoL = dot(N, L);
		const float NoV = dot(N, V);

		const float eta     = get_eta(NoV);
		const float inv_eta = get_inv_eta(NoV);

		const Vector3f H = microfacet(V,L,N,inv_eta);

		const float NoH = dot(N, H);

		const float transmission_sign = is_transmissive() ? -1.0f : 1.0f;

		// check whether there is energy exchange between different sides of the surface
		if (transmission_sign * NoL * NoV <= 0.0f || NoH == 0.0f)
			return 0.0f;

		const float D = hvd_ggx_eval(inv_alpha, fabsf( NoH ), dot(geometry.tangent, H), dot(geometry.binormal, H));
		const float G1 = PredividedSmithG1V(fabsf(NoV), fabsf(NoL));
		
		float p = clamp_inf( G1 * D );

		if (is_transmissive())
		{
			// compute whether this is a ray for which we have total internal reflection,
			// i.e. if cos_theta_t2 <= 0, and in that case return zero.
			const float VoH = dot(V, H);
			const float LoH = dot(L, H);
			const float cos_theta_i = fabsf(VoH);
			const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
			if (cos_theta_t2 < 0.0f)
				return 0.0f;

			// compute dwo_dh: equation 17 in Microfacet Models for Refraction through Rough Surfaces, Walter et al.
			const float sqrtDenom = VoH + inv_eta * LoH;
			const float factor = 4 * inv_eta * inv_eta
				* VoH * LoH /
				(sqrtDenom * sqrtDenom);
				// NOTE: the actual formula has no 4 factor at the numerator, but it contains NoV * NoL at
				// the denominator, which is already included in our G term above. The additional multiplication
				// by 4 is needed to account for the predivision.
			p *= clamp_inf(factor);
		}

		if (measure == kSolidAngle)
			p *= fabsf( NoL );

		return p;
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
		const float2 alpha     = make_float2(roughness, roughness);
		const float2 inv_alpha = make_float2(inv_roughness, inv_roughness);

		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);
		const float eta = get_eta(NoV);

		const float sgn_V = NoV > 0.0f ? 1.0f : -1.0f;

		const Vector3f V_local(
			dot(V, geometry.tangent),
			dot(V, geometry.binormal),
			sgn_V * NoV );

		Vector3f H = vndf_ggx_smith_sample( make_float2(u.x,u.y), alpha, V_local);

		H =
			H.x * geometry.tangent +
			H.y * geometry.binormal +
			sgn_V * H.z * geometry.normal_s;

		if (!is_transmissive())
		{
			// reflect
			L = 2 * dot(V, H) * H - V;
		}
		else
		{
			// compute whether this is a ray for which we have total internal reflection,
			// i.e. if cos_theta_t2 <= 0, and in that case return zero.
			const float VoH = dot(V, H);
			const float cos_theta_i  = VoH;
			const float cos_theta_t2 = 1.f - eta * eta * (1.f - cos_theta_i * cos_theta_i);
			if (cos_theta_t2 < 0.0f)
			{
				p		= 0.0f;
				p_proj	= 0.0f;
				g		= Vector3f(0.0f);
				return;
			}
			const float cos_theta_t = -(cos_theta_i >= 0.0f ? 1.0f : -1.0f) * sqrtf(cos_theta_t2);

			// refract
			L = (eta * cos_theta_i + cos_theta_t) * H - eta * V;
		}

		const float NoL = dot(N, L);
		const float NoH = dot(N, H);

		const float transmission_sign = is_transmissive() ? -1.0f : 1.0f;

		// check whether there is energy exchange between different sides of the surface
		if (transmission_sign * NoL * NoV <= 0.0f || NoH == 0.0f)
		{
			p		= 0.0f;
			p_proj	= 0.0f;
			g		= Vector3f(0.0f);
		}
		else
		{
			const float D = hvd_ggx_eval(inv_alpha, fabsf( NoH ), dot(geometry.tangent, H), dot(geometry.binormal, H));

		  #if defined(USE_APPROX_SMITH)
			const float G = PredividedSmithJointApprox(fabsf(NoV), fabsf(NoL));
		  #else
			const float G = PredividedSmithJoint(fabsf(NoV), fabsf(NoL));
		  #endif

			const float G1 = PredividedSmithG1V(fabsf(NoV), fabsf(NoL));

			p_proj	= clamp_inf( G1 * D );
			p		= p_proj * fabsf(NoL);
			g		= clamp_inf( G / G1 );
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
		// TODO!
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
		const Vector3f N = geometry.normal_s;

		p_proj = 1.0f / cugar::max(this->p(geometry, V, L, kProjectedSolidAngle), 1.0e-8f);
		p      = p_proj / cugar::max(fabsf(dot(L, N)), 1.0e-8f);
	}

public:
	float roughness;
	float inv_roughness;
	float int_ior;
	float ext_ior;
};

/*! \}
 */

} // namespace cugar
