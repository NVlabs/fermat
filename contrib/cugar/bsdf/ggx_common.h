/*
 * Copyright (c) 2010-2019, NVIDIA Corporation
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

/*! \file ggx_common.h
 *   \brief Defines the GGX BSDF
 *
 * This module provides a GGX BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/bsdf/differential_geometry.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

/// return the microfacet normal for a given pair of incident and outgoing direction vectors
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector3f microfacet(const Vector3f V, const Vector3f L, const Vector3f N, const float inv_eta)
{
	Vector3f H = (dot(V, N)*dot(L, N) >= 0.0f) ?
		V + L :
		V + L*inv_eta;

	// when in doubt, return N
	if (dot(H,H) == 0.0f)
		return N;

	// make sure H points in the same direction as N
	if (dot(N, H) < 0.0f)
		H = -H;

	return normalize(H);
}

/// return the microfacet normal for a given pair of incident and outgoing direction vectors
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
Vector3f vndf_microfacet(const Vector3f V, const Vector3f L, const Vector3f N, const float inv_eta)
{
	Vector3f H = (dot(V, N)*dot(L, N) >= 0.0f) ?
		V + L :
		V + L*inv_eta;

	// when in doubt, return N
	if (dot(H,H) < 1.0e-12f)
		return N;

	// make sure H points in the same direction as V
	if (dot(V, H) < 0.0f)
		H = -H;

	return normalize(H);
}

/// evaluate anisotropic GGX / Trowbridge-Reitz distribution on the non-projected hemisphere
///
/// [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float hvd_ggx_eval(
	const float2 &inv_alpha,
	const float nh,     // dot(shading_normal, h)
	const float ht,     // dot(x_axis, h)
	const float hb)     // dot(z_axis, h)
{
	const float x = ht * inv_alpha.x;
	const float y = hb * inv_alpha.y;
	const float aniso = x * x + y * y;

	/*original:
	const float nh2 = nh * nh;
	const float f = aniso / nh2 + 1.0f;
	return (float)(1.0 / M_PI) * inv_alpha.x * inv_alpha.y / (f * f *
	nh2 * nh);
	*/
	const float f = aniso + nh * nh;
	return (1.0f / M_PIf) * inv_alpha.x * inv_alpha.y /** nh*/ / (f * f);
}

/// sample isotropic GGX distribution on the non-projected hemisphere
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float3 hvd_ggx_sample(
	const float2 &samples,
	const float   inv_alpha)
{
	// optimized version, any further optimizations introduce severe precision problems
	const float phi = (float)(2.0 * M_PI) * samples.x;
	float cp, sp;
	cugar::sincosf(phi, &sp, &cp);

	const float ia2 = inv_alpha * inv_alpha;

	const float sp2 = cp;
	const float cp2 = sp;

	const float tantheta2 = samples.y / ((1.0f - samples.y) * ia2);
	const float sintheta = ::sqrtf(tantheta2 / (1.0f + tantheta2));

	return make_float3(
		cp2 * sintheta,
		sp2 * sintheta,
		cugar::rsqrtf(1.0f + tantheta2)); // also okay on CPU so far
}

/// sample the half-vector anisotropic GGX distribution on the non-projected hemisphere
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float3 hvd_ggx_sample(
	const float2 &samples,
	const float2 &inv_alpha)
{
	// optimized version, any further optimizations introduce severe precision problems
	const float phi = (float)(2.0 * M_PI) * samples.x;
	float cp, sp;
	sincosf(phi, &sp, &cp);

	const float iax2 = inv_alpha.x*inv_alpha.x;
	const float iay2 = inv_alpha.y*inv_alpha.y;

	const float is = rsqrtf(iax2 * cp*cp + iay2 * sp*sp); // also okay on CPU so far

	const float sp2 = inv_alpha.x * cp*is;
	const float cp2 = inv_alpha.y * sp*is;

	const float tantheta2 = samples.y / ((1.0f - samples.y) * (cp2 * cp2 * iax2 + sp2 * sp2 * iay2));
	const float sintheta = sqrtf(tantheta2 / (1.0f + tantheta2));

	return make_float3(
		cp2 * sintheta,
		sp2 * sintheta,
		rsqrtf(1.0f + tantheta2)); // also okay on CPU so far
}

namespace ggx {

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector2f Fu(const float u, const float2 ia, const float2 b)
	{
		const float iax2 = ia.x*ia.x;
		const float iay2 = ia.y*ia.y;

		const float2 p = make_float2(sinf(u), cosf(u));

		return Vector2f(
			sqrtf(iax2 * p.y*p.y + iay2 * p.x*p.x) * b.x - p.y,
			sqrtf(iax2 * p.y*p.y + iay2 * p.x*p.x) * b.y - p.x);
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector2f Ju(const float u, const float2 ia, const float2 b)
	{
		const float iax2 = ia.x*ia.x;
		const float iay2 = ia.y*ia.y;

		const float2 p = make_float2(sinf(u), cosf(u));

		Vector2f r;
		r.x = (-2.0f * iax2 * p.y * p.x + 2.0f * iay2 * p.x * p.y) * b.x * 0.5f / (iax2 * p.y*p.y + iay2 * p.x*p.x) + p.x;						// d/du f1
		r.y = (-2.0f * iax2 * p.y * p.x + 2.0f * iay2 * p.x * p.y) * b.x * 0.5f / (iax2 * p.y*p.y + iay2 * p.x*p.x) - p.y;						// d/du f2
		return r;
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	bool refine_solution(float& u, const float2 ia, const float2 b)
	{
		//const float r_old = length(Fu(u, ia, b));
		for (uint32 i = 0; i < 100; ++i)
		{
			const Vector2f f = Fu(u, ia, b);
			if (fabsf(f.x) < 1.0e-5f &&
				fabsf(f.y) < 1.0e-5f)
				return true;

			const Vector2f J = Ju(u, ia, b);

			const Vector2f inv_J = J / dot(J, J);

			u = u - dot(inv_J, f);
		}
		//const float r_new = length(Fu(u, ia, b));
		//printf("refined %f!\n", r_new / r_old);
		//printf("inversion failed, %f, %f!\n", r_old, r_new);
		return false;
	}

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	bool invert(float& u, const float2 ia, const float2 b)
	{
		// easy analytical inversion
		const float cp = b.x;
		const float sp = b.y;

		//const float phi = atan2(sp, cp);
		const float phi = atan2f( sp, cp );
		u = phi < 0.0f ? phi + 2.0f*float(M_PI) : phi;
		//refine_solution(u, ia, b);
		return true;
	}

} // namespace ggx

/// invert the the half-vector anisotropic GGX distribution sampling function
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float2 hvd_ggx_invert(
	const float3 H,
	const float2 inv_alpha)
{
	const float tantheta2 = 1.0f / (H.z * H.z) - 1.0f;
	const float sintheta = sqrtf(tantheta2 / (1.0f + tantheta2));
	const float cp2 = H.x / sintheta;
	const float sp2 = H.y / sintheta;

	const float iax2 = inv_alpha.x*inv_alpha.x;
	const float iay2 = inv_alpha.y*inv_alpha.y;

	const float v = tantheta2 * (cp2 * cp2 * iax2 + sp2 * sp2 * iay2) / (1.0f + tantheta2 * (cp2 * cp2 * iax2 + sp2 * sp2 * iay2));

	float u(0.5f);

	ggx::invert(u, inv_alpha, make_float2(sp2 / inv_alpha.x, cp2 / inv_alpha.y));

	u /= (2.0f * M_PIf);

	return make_float2(u, v);
}

/// A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals
/// https://hal.archives-ouvertes.fr/hal-01509746
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float3 vndf_ggx_smith_sample(
	const float2	samples,
	const float2	alpha,
	const Vector3f	_V)
{
	// stretch view
	Vector3f V = normalize(Vector3f(alpha.x * _V.x, alpha.y * _V.y, _V.z));

	// orthonormal basis
	Vector3f T1 = (V.z < 0.9999f) ? normalize(cross(V, Vector3f(0,0,1))) : Vector3f(1,0,0);
	Vector3f T2 = cross(T1, V);

	// sample point with polar coordinates (r, phi)
	float a = 1.0f / (1.0f + V.z);
	float r = sqrtf(samples.x);
	float phi = (samples.y < a) ? samples.y / a * M_PIf : M_PIf + (samples.y - a) / (1.0f - a) * M_PIf;
	float P1 = r*cosf(phi);
	float P2 = r*sinf(phi) * ((samples.y < a) ? 1.0f : V.z);

	// compute normal
	Vector3f N = P1*T1 + P2*T2 + sqrtf(max(0.0f, 1.0f - P1*P1 - P2*P2))*V;

	// unstretch
	N = normalize(Vector3f(alpha.x*N.x, alpha.y*N.y, max(0.0f, N.z)));
	return N;
}

/// Inversion function for "A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals"
/// https://hal.archives-ouvertes.fr/hal-01509746
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
float2 vndf_ggx_smith_invert(
	const Vector3f	_N,
	const float2	alpha,
	const Vector3f	_V)
{
	// stretch view
	Vector3f V = normalize(Vector3f(alpha.x * _V.x, alpha.y * _V.y, _V.z));

	// orthonormal basis
	Vector3f T1 = (V.z < 0.9999f) ? normalize(cross(V, Vector3f(0,0,1))) : Vector3f(1,0,0);
	Vector3f T2 = cross(T1, V);

	// solve for (X,Y,Z) the system:
	//
	// N = normalize(Vector3f(ax*X, ay*Y, max(0.0f, Z)));
	//
	// <=> N = (ax*X, ay*Y, max(0.0f, Z) / sqrt(ax*X, ay*Y, max(0.0f, Z)^2
	//
	// <=> Nx^2 = (ax*X)^2 / (ax*X, ay*Y, max(0.0f, Z)^2
	//     Ny^2 = (ay*Y)^2 / (ax*X, ay*Y, max(0.0f, Z)^2
	//     Nz^2 = Z^2 / (ax*X, ay*Y, max(0.0f, Z)^2
	//
	// <=> a0 * XX + b0 * YY + c0 * ZZ = 0
	//     a1 * XX + b1 * YY + c1 * ZZ = 0
	//     a2 * XX + b2 * YY + c2 * ZZ = 0
	// 
	// => a linear system in XX, YY, ZZ
	// => solve for (XX,YY,ZZ)
	//
	// M * (XX,YY,ZZ) = 0 with the constraint (XX + YY + ZZ) = 1
	//
	// => find the solution space kernel(M), and find a point of that space satisfying the constraint
	//
	// Geometrically: given the 1d vector space spanned by < N > (i.e. the kernel of the normalization),
	// we need to find the point in that space with (ax X)^2 + (ax Y)^2 + Z^2 = 1.
	// In other words, we need to intersect the ray < N > with an ellipsoid.
	//
	// Given M = (ax, ay, 1) * I and setting v' = M^(-1) v, we need to solve for t^2 |v'| ^2 = 1,
	// obtaining t = |v'|^2.
	//
	Vector3f N = normalize( Vector3f( _N.x / alpha.x, _N.y / alpha.y, _N.z ) );

	// solve for (P1, P2)
	const float P1 = dot( N, T1 );
	const float P2 = dot( N, T2 );

	const float a = 1.0f / (1.0f + V.z);

	float2 samples;

	// two cases:
	// 1. N projects along V to the tangent half disk (on the tangent plane)   (green disk in the paper)
	// 2. N projects along V to the half disk orthogonal to V				   (blue disk in the paper)
	Vector3f PN = N - dot(N,V) * V;

	if (PN.z > 0.0f)
	{	// case 2: samples.y < a	(blue disk :  phi in [0,PI))
		const float r2 = min( P1*P1 + P2*P2, 1.0f );
		const float r  = sqrtf(r2);

		float phi = r > 1.0e-6f ? atan2f( P2 / r, P1 / r ) : 0.0f;
		if (phi < 0.0f)
			phi += M_TWO_PIf;

		// solve for samples.x : r = sqrtf(samples.x);
		samples.x = r2;

		// solve for samples.y : phi = samples.y / a * M_PIf;
		samples.y = phi * a / M_PIf;
		if (phi < M_PIf)
		{
			assert(is_finite(samples.x) && is_finite(samples.y));
			assert(samples.y >= 0.0f && samples.y <= 1.0f);
			return samples;
		}
	}
	//else // stay on the safe side: if the first case didn't work, try the second
	{
		// case 1: samples.y >= a	(green disk :  phi in [PI,2*PI))
		const float r2 = min( P1*P1 + P2*P2 / (V.z*V.z), 1.0f );
		const float r  = sqrtf(r2);

		float phi = r > 1.0e-6f && V.z > 1.0e-6f ? atan2f( P2 / (r * V.z), P1 / r ) : 0.0f;
		if (phi < 0.0f)
			phi += M_TWO_PIf;

		// solve for samples.x : r = sqrtf(samples.x);
		samples.x = r2;

		// solve for samples.y : phi = M_PIf + (samples.y - a) / (1.0f - a) * M_PIf;
		// => phi - M_PIf = (samples.y - a) / (1.0f - a) * M_PIf;
		// => (phi - M_PIf) * (1.0f - a) / M_PIf; = (samples.y - a);
		samples.y = (phi - M_PIf) * (1.0f - a) / M_PIf + a;
		assert(is_finite(samples.x) && is_finite(samples.y));
		assert(samples.y >= 0.0f && samples.y <= 1.0f);
	}
	return samples;
}

/*! \}
 */

} // namespace cugar
