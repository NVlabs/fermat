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

/*! \file ltc.h
 *   \brief Defines the LTC BSDF
 *
 * This module provides a ltc BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/linalg/matrix.h>
#include <cugar/basic/numbers.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/spherical/mappings.h>

#ifndef ONE_OVER_PIf
#define ONE_OVER_PIf		0.31830988618f
#endif

#ifndef ONE_OVER_TWO_PIf
#define ONE_OVER_TWO_PIf	0.15915494309f
#endif

#ifndef HALF_PIf
#define HALF_PIf			1.57079632679f
#endif

#ifndef ONE_OVER_HALF_PIf
#define ONE_OVER_HALF_PIf	0.63661977236f
#endif

#define LTC_USE_INVDET

namespace cugar {

/// clip a quad to the plane z = 0
///
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 clip_quad_to_horizon(cugar::Vector3f L[5]);

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///
/// Implements the a bsdf based on a single Linearly Transformed Cosine (LTC).
/// LTC can be used to approximate many isotropic BSDF types.
///
/// see:
///>    Real-Time Polygonal-Light Shading with Linearly Transformed Cosines,
///>    Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
///
struct LTCBsdf
{
	inline
	static void preprocess(const uint32 size, const Matrix3x3f* tabM, float4* tab, float4* tab_inv)
	{
		for (uint32 j = 0; j < size; ++j)
			for (uint32 i = 0; i < size; ++i)
			{
				Matrix3x3f M = transpose(tabM[i + j*size]);

				float a = M[0][0];
				float b = M[2][0];
				float c = M[1][1];
				float d = M[0][2];

				tab[i + j*size] = make_float4(a, b, c, d);

				// Rescaled inverse of m:
				// a 0 b   inverse   1      0      -b
				// 0 c 0     ==>     0 (a - b*d)/c  0
				// d 0 1            -d      0       a

				// Store the variable terms
				tab_inv[i + j*size] = make_float4(
					a,
					-b,
					(a - b*d) / c,
					-d);
			}
	}

	struct LTC
	{
		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		LTC(const float cosTheta, const float4* tabM, const float4* tabMinv, const float* tabA, const uint32 size)
		{
			const float theta = acos(fabsf(cosTheta));

			const float theta_unit	= theta * ONE_OVER_HALF_PIf * size;
			const float theta_floor	= floorf(theta_unit);				// NOTE:
			const float theta_c1	= theta_unit - theta_floor;			// tex fetch logic that
			const float theta_c0	= 1.0f - theta_c1;					// can be done in HW

			const int t1 = max(0, min((int)size - 1, (int)theta_floor));
			const int t2 = min((int)size - 1, t1 + 1);

			amplitude = tabA[t1*size] * theta_c0 +						// 1 TEX fetch
						tabA[t2*size] * theta_c1;						// with interpolation

			m = Vector4f(tabM[t1*size]) * theta_c0 +
				Vector4f(tabM[t2*size]) * theta_c1;

			//M[0][0] = m.x;  M[1][0] = 0;    M[2][0] = m.y;
			//M[0][1] = 0;    M[1][1] = m.z;  M[2][1] = 0;
			//M[0][2] = m.w;  M[1][2] = 0;    M[2][2] = 1;

		  #if 1
			m_inv = Vector4f(tabMinv[t1*size]) * theta_c0 +				// 1 TEX fetch
				    Vector4f(tabMinv[t2*size]) * theta_c1;				// with interpolation
		  #else
			const float a = m.x;
			const float b = m.y;
			const float c = m.z;
			const float d = m.w;

			// Rescaled inverse of m:
			// a 0 b   inverse   1      0      -b
			// 0 c 0     ==>     0 (a - b*d)/c  0
			// d 0 1            -d      0       a

			// Store the variable terms
			m_inv = Vector4f(
				a,
				-b,
				(a - b*d) / c,
				-d);
		  #endif

			//invM[0][0] = 1;			invM[1][0] = 0;			invM[2][0] = m_inv.y;
			//invM[0][1] = 0;			invM[1][1] = m_inv.z;	invM[2][1] = 0;
			//invM[0][2] = m_inv.w;		invM[1][2] = 0;			invM[2][2] = m_inv.x;

		  #if defined(LTC_USE_INVDET)
			//detMinv = fabsf(cugar::det(invM));
			detMinv = fabsf(m_inv.x * m_inv.z - m_inv.w * m_inv.z*m_inv.y);		// 2 MULS + 1 MAD
		  #else
			//detM = fabsf(cugar::det(M));
			detM = fabsf(m.x * m.z - m.w * m.z*m.y);							// 2 MULS + 1 MAD
		  #endif
		}

		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		Vector3f transform(const Vector3f L) const
		{
			//return M * L;
			return Vector3f(
				m.x * L.x + m.w * L.z,									// 1 MUL + 1 MAD
				m.z * L.y,												// 1 MUL
				m.y * L.x + L.z);										// 1 MAD
		}

		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		Vector3f inv_transform(const Vector3f L) const
		{
			//return invM * L;
			return Vector3f(
				L.x + m_inv.w * L.z,									// 1 MAD
				m_inv.z * L.y,											// 1 MUL
				m_inv.y * L.x + m_inv.x * L.z);							// 1 MUL + 1 MAD
		}

		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		float p(const Vector3f L) const									// 1 SQRT + 2 DIV + 2 MAD + 7 MUL + 2 ADD + 1 MAX
		{
		  #if defined(LTC_USE_INVDET)
			Vector3f Loriginal = inv_transform(L);						// 2 MAD + 2 MUL
			const float l      = length(Loriginal);						// 1 SQRT + 2 ADD

			const float InvJacobian = detMinv / (l*l*l);				// 1 DIV + 3 MUL

			const float D = ONE_OVER_PIf * max(0.0f, Loriginal.z / l);	// 1 DIV + 1 MUL + 1 MAX
			return D * InvJacobian;										// 1 MUL
		  #else
			const Vector3f Loriginal = normalize(inv_transform(L));		// 1 SQRT + 2 ADD
			const Vector3f L_ = transform(Loriginal);

			const float l = length(L_);									// 1 SQRT + 2 ADD
			const float InvJacobian = (l*l*l) / detM;					// 1 DIV + 3 MUL
			  //const float Jacobian = detM / (l*l*l);						// 1 DIV + 3 MUL

			const float D = ONE_OVER_PIf * max(0.0f, Loriginal.z);

			return D * InvJacobian;										// 1 MUL
			  //return D / Jacobian;										// 1 DIV
		  #endif
		}


		/// return the integral of an edge in the upper hemisphere
		///
		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		float clipped_edge_integral(const cugar::Vector3f p0, const cugar::Vector3f p1) const
		{
			return acosf(dot(p0,p1)) * dot( normalize(cross(p0,p1)), Vector3f(0,0,1) );
		}

		/// return the integral of a quadrilateral polygon in the upper hemisphere
		///
		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		float clipped_polygon_integral(const uint32 n, const cugar::Vector3f p[5]) const
		{
			// use equation 11 from:
			//   Real-Time Polygonal-Light Shading with Linearly Transformed Cosines,
			//   Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
			float r = 0.0f;
			r += clipped_edge_integral(p[0],p[1]);
			r += clipped_edge_integral(p[1],p[2]);
			r += clipped_edge_integral(p[2],p[3]);
			if (n >= 4) r += clipped_edge_integral(p[3],p[4]);
			if (n >= 5) r += clipped_edge_integral(p[4],p[0]);
			return fabsf(r) * ONE_OVER_TWO_PIf;
		}

		/// compute the integral of a "small" sector, with the phi range smaller than PI
		///
		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		float small_hemispherical_sector_integral(const Matrix3x3f& T, const float2 theta, const float2 phi) const
		{
			// compute the spherical polygon corresponding to the sector, and transform it by the inverse LTC transform
			Vector3f p[5];
			p[0] = inv_transform( T * from_spherical_coords( cugar::Vector2f(phi.x, theta.x) ) );
			p[1] = inv_transform( T * from_spherical_coords( cugar::Vector2f(phi.y, theta.x) ) );
			p[2] = inv_transform( T * from_spherical_coords( cugar::Vector2f(phi.y, theta.y) ) );
			p[3] = inv_transform( T * from_spherical_coords( cugar::Vector2f(phi.x, theta.y) ) );

			const uint32 n = clip_quad_to_horizon( p );

			return clipped_polygon_integral( n, p );

		}

		/// compute the integral of a general sector, with arbitrary theta and phi ranges
		///
		CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
		float hemispherical_sector_integral(const Matrix3x3f& T, const float2 theta, const float2 phi) const
		{
			if (phi.y - phi.x == 2.0f*M_PIf)
			{
				// divide the domain in four parts
				const float phi_1 = phi.x*0.75f + phi.y*0.25f;
				const float phi_2 = (phi.x + phi.y)*0.5f;
				const float phi_3 = phi.x*0.5f + phi.y*0.75f;

				return
					small_hemispherical_sector_integral(T, theta, make_float2(phi.x, phi_1) ) +
					small_hemispherical_sector_integral(T, theta, make_float2(phi_1, phi_2) ) +
					small_hemispherical_sector_integral(T, theta, make_float2(phi_2, phi_3) ) +
					small_hemispherical_sector_integral(T, theta, make_float2(phi_3, phi.y) );
			}
			else if (phi.y - phi.x >= M_PIf)
			{
				// divide the domain in two parts
				const float phi_2 = (phi.x + phi.y)*0.5f;

				return
					small_hemispherical_sector_integral(T, theta, make_float2(phi.x, phi_2) ) +
					small_hemispherical_sector_integral(T, theta, make_float2(phi_2, phi.y) );
			}
			else
				return small_hemispherical_sector_integral(T, theta, phi);
		}

		Vector4f    m;
		Vector4f    m_inv;

		//Matrix3x3f	M;
		//Matrix3x3f	invM;
	  #if defined(LTC_USE_INVDET)
		float		detMinv;
	  #else
		float		detM;
	  #endif
		float		amplitude;
	};

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	LTCBsdf(const float _roughness, const float4* _tabM, const float4* _tabMinv, const float* _tabA, const uint32 _size) :
		roughness(_roughness),
		tabM(_tabM),
		tabMinv(_tabMinv),
		tabA(_tabA),
		size(_size)
	{
		const int a = max(0, min((int)size - 1, (int)floorf(sqrtf(roughness) * size)));

		tabM	+= a;
		tabMinv += a;
		tabA	+= a;
	}

	/// evaluate the BRDF f(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);
		const float NoL = dot(N, L);

		if (NoV * NoL <= 0.0f)
			return Vector3f(0.0f);

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		//const Vector3f T = normalize(V - N*dot(V, N));
		//const Vector3f B = cross(N, T);
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);
		const cugar::Vector3f local_L(
			dot(T, L),
			dot(B, L),
			fabsf(NoL));

		LTC ltc(fabsf(NoV), tabM, tabMinv, tabA, size);

		return ltc.amplitude * ltc.p(local_L) / fabsf(NoL); // The LTC already contains NoL - hence we must de-multiply it out
	}

	/// evaluate the BRDF/pdf ratio f(V,L)/p(V,L) wrt projected solid angle
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	Vector3f f_over_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);
		const float NoL = dot(N, L);

		if (NoV * NoL <= 0.0f)
			return Vector3f(0.0f);

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);
		const cugar::Vector3f local_L(
			dot(T, L),
			dot(B, L),
			fabsf(NoL));

		LTC ltc(fabsf(NoV), tabM, tabMinv, tabA, size);

		const float p = ltc.p(local_L);

		return p > 0.0f ? ltc.amplitude : 0.0f; // f/p wrt projected solid angle
	}

	/// evaluate the BRDF and the pdf in a single call
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	void f_and_p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, Vector3f& f, float& p, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);
		const float NoL = dot(N, L);

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);
		const cugar::Vector3f local_L(
			dot(T, L),
			dot(B, L),
			fabsf(NoL));

		LTC ltc(fabsf(NoV), tabM, tabMinv, tabA, size);

		p = ltc.p(local_L);
		f = (NoV * NoL <= 0.0f) ?
			Vector3f(0.0f) :
			ltc.amplitude * p / fabsf(NoL); // The LTC already contains NoL - hence we must de-multiply it out

		if (measure == kProjectedSolidAngle)
			p /= fabsf(NoL);
	}

	/// evaluate the pdf of sampling L given V, p(L|V) = p(V,L)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float p(const DifferentialGeometry& geometry, const Vector3f V, const Vector3f L, const SphericalMeasure measure = kProjectedSolidAngle) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);
		const float NoL = dot(N, L);

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);
		const cugar::Vector3f local_L(
			dot(T, L),
			dot(B, L),
			fabsf(NoL));

		LTC ltc(fabsf(NoV), tabM, tabMinv, tabA, size);

		float p = ltc.p(local_L);

		return (measure == kProjectedSolidAngle) ? p / fabsf(NoL) : p;
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
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);

		LTC ltc(fabsf(NoV), tabM, tabMinv, tabA, size);

		//const float theta = acosf(sqrtf(u.x));
		//const float phi = 2.0f*3.14159f * u.y;
		//printf("S: %f, %f, %f\n", sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta));
		//Vector3f local_L = normalize(ltc.transform( Vector3f(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta)) ));
		Vector3f local_L = normalize(ltc.transform( square_to_cosine_hemisphere(u.xy()) ));
		//printf("S -> L: %f, %f, %f\n", local_L.x, local_L.y, local_L.z);

		// reverse the sign bit
		if (NoV < 0.0f)
			local_L.z = -local_L.z;

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);
		L = local_L.x * T +
			local_L.y * B +
			local_L.z * N;

		const float NoL = dot(N, L);

		// reset the sign bit
		if (NoV < 0.0f)
			local_L.z = -local_L.z;

		p = ltc.p(local_L);
		p_proj = p / fabsf(local_L.z);

		g = (NoV * NoL <= 0.0f || p == 0.0f) ?
			Vector3f(0.0f) :
			ltc.amplitude; // g = f/p wrt projected solid angle
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
		return 0;
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
	}

	/// get the LTC for a given view vector
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	LTC get_ltc(
		const DifferentialGeometry& geometry,
		const Vector3f				V) const
	{
		const Vector3f N = geometry.normal_s;

		const float NoV = dot(N, V);

		return LTC(fabsf(NoV), tabM, tabMinv, tabA, size);
	}

	/// return the integral of the LTC within a sector of the local hemisphere defined by the geometry frame (T,B,N)
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	float hemispherical_sector_integral(
		const DifferentialGeometry& geometry,
		const Vector3f				V,
		const float2				theta,
		const float2				phi) const
	{
		//
		// compute the transformation from the local frame defined by
		// geometry.(tangent,binormal,N) to the LTC's frame (T,B,N);
		// this is equivalent to right-multiplying the matrix to go in world space
		// from local geometry coorinates, by the matrix to go from world space
		// into the LTC's frame.
		//

		// use a coordinate system in which V has coordinates (sin(theta), 0, cos(theta))
		const Vector3f N = geometry.normal_s;
		const Vector3f B = normalize(cross(N, V));
		const Vector3f T = cross(B, N);

		Matrix3x3f M;
		M[0][0] = dot(T, geometry.tangent);
		M[0][1] = dot(T, geometry.binormal);
		M[0][2] = 0.0f;
		M[1][0] = dot(B, geometry.tangent);
		M[1][1] = dot(B, geometry.binormal);
		M[1][2] = 0.0f;
		M[2][0] = 0.0f; // dot(N, geometry.tangent)
		M[2][1] = 0.0f; // dot(N, geometry.binormal)
		M[2][2] = 1.0f; // dot(N, N)

		return get_ltc(geometry, V).hemispherical_sector_integral( M, theta, phi );
	}

public:
	float			roughness;
	const float4*	tabM;
	const float4*	tabMinv;
	const float*	tabA;
	uint32			size;
};


// clip a quad to the plane z = 0
//
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
uint32 clip_quad_to_horizon(cugar::Vector3f L[5])
{
	// detect clipping config
	int config = 0;
	if (L[0].z > 0.0) config += 1;
	if (L[1].z > 0.0) config += 2;
	if (L[2].z > 0.0) config += 4;
	if (L[3].z > 0.0) config += 8;

	// clip
	uint32 n = 0;

	if (config == 0)
	{
		// clip all
	}
	else if (config == 1) // V1 clip V2 V3 V4
	{
		n = 3;
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
		L[2] = -L[3].z * L[0] + L[0].z * L[3];
	}
	else if (config == 2) // V2 clip V1 V3 V4
	{
		n = 3;
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
	}
	else if (config == 3) // V1 V2 clip V3 V4
	{
		n = 4;
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
		L[3] = -L[3].z * L[0] + L[0].z * L[3];
	}
	else if (config == 4) // V3 clip V1 V2 V4
	{
		n = 3;
		L[0] = -L[3].z * L[2] + L[2].z * L[3];
		L[1] = -L[1].z * L[2] + L[2].z * L[1];
	}
	else if (config == 5) // V1 V3 clip V2 V4) impossible
	{
		n = 0;
	}
	else if (config == 6) // V2 V3 clip V1 V4
	{
		n = 4;
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
		L[3] = -L[3].z * L[2] + L[2].z * L[3];
	}
	else if (config == 7) // V1 V2 V3 clip V4
	{
		n = 5;
		L[4] = -L[3].z * L[0] + L[0].z * L[3];
		L[3] = -L[3].z * L[2] + L[2].z * L[3];
	}
	else if (config == 8) // V4 clip V1 V2 V3
	{
		n = 3;
		L[0] = -L[0].z * L[3] + L[3].z * L[0];
		L[1] = -L[2].z * L[3] + L[3].z * L[2];
		L[2] =  L[3];
	}
	else if (config == 9) // V1 V4 clip V2 V3
	{
		n = 4;
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
		L[2] = -L[2].z * L[3] + L[3].z * L[2];
	}
	else if (config == 10) // V2 V4 clip V1 V3) impossible
	{
		n = 0;
	}
	else if (config == 11) // V1 V2 V4 clip V3
	{
		n = 5;
		L[4] = L[3];
		L[3] = -L[2].z * L[3] + L[3].z * L[2];
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
	}
	else if (config == 12) // V3 V4 clip V1 V2
	{
		n = 4;
		L[1] = -L[1].z * L[2] + L[2].z * L[1];
		L[0] = -L[0].z * L[3] + L[3].z * L[0];
	}
	else if (config == 13) // V1 V3 V4 clip V2
	{
		n = 5;
		L[4] = L[3];
		L[3] = L[2];
		L[2] = -L[1].z * L[2] + L[2].z * L[1];
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
	}
	else if (config == 14) // V2 V3 V4 clip V1
	{
		n = 5;
		L[4] = -L[0].z * L[3] + L[3].z * L[0];
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
	}
	else if (config == 15) // V1 V2 V3 V4
	{
		n = 4;
	}
    
	if (n == 3) L[3] = L[0];
	if (n == 4) L[4] = L[0];

	return n;
}

/*! \}
 */

} // namespace cugar
