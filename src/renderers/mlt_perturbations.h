/*
 * Fermat
 *
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <mlt.h>
#include <cugar/sampling/distributions.h>
#include <bsdf.h>
#include <edf.h>
#include <bpt_utils.h>
#include <path_inversion.h>

namespace {

	// apply a spherical perturbation to a given vector
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f exponential_spherical_perturbation(const cugar::Vector3f dir, const cugar::Vector2f z, const float exp_radius = 0.1f)
	{
		// build a frame around dir
		const cugar::Vector3f n = cugar::normalize(dir);
		const cugar::Vector3f t = cugar::orthogonal(n);
		const cugar::Vector3f b = cugar::cross(n,t);

		// map z to a point on the disk
		const cugar::Bounded_exponential dist(0.0001f, exp_radius);
		//const cugar::Cauchy_distribution dist(exp_radius);

		const float phi   = z.x*2.0f*M_PIf;
		const float theta = dist.map(z.y);

		const cugar::Vector3f local_dir = cugar::from_spherical_coords(cugar::Vector2f(phi, theta));

		// and map it back to the sphere
		return
			local_dir.x * t +
			local_dir.y * b +
			local_dir.z * n;
	}

	// apply a spherical perturbation to a given vector
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f spherical_perturbation(const VertexGeometry& geom, const cugar::Vector3f dir, const cugar::Vector2f z, const float radius)
	{
	#if DISABLE_SPHERICAL_PERTURBATIONS
		return dir;
	#elif 1
		return exponential_spherical_perturbation(dir, z, radius);
	#else
		// map the old direction to the unit square (inverting a uniform sphere parameterization)
		const float s = dot(geom.normal_s, dir) >= 0.0f ? 1.0f : -1.0f;
		cugar::Vector3f local_dir(
			dot(geom.tangent, dir),
			dot(geom.binormal, dir),
			s * dot(geom.normal_s, dir ));

		cugar::Vector2f uv = cugar::cosine_hemisphere_to_square(local_dir);

		// map z to a point on the disk
		const cugar::Bounded_exponential dist(0.0001f, 0.1f);

		const float phi = z.x*2.0f*M_PIf;
		const float r   = dist.map(z.y);

		cugar::Vector2f d( cosf(phi) * r, sinf(phi) * r );

		// apply a perturbation in the uv domain
		uv.x = cugar::mod(uv.x + d.x, 1.0f);
		uv.y = cugar::mod(uv.y + d.y, 1.0f);

		// and remap it to a direction
		local_dir = cugar::square_to_cosine_hemisphere(uv);
		return
			geom.tangent * local_dir.x +
			geom.binormal * local_dir.y +
			s * geom.normal_s * local_dir.z;
	#endif
	}

	// Compute the geometric portion of acceptance ratio due to a spherical perturbation;
	// note that the full acceptance ratio between two paths is obtained as:
	//
	//   ar = [f(new) / f(old)] * [T(old) / T(new)] = [f(new)/T(new)] * [T(old)/f(old)]
	//
	// where f() is the full path throughput; this function returns the remainder
	// of the division of the geometric throughput G terms in f(new) and f(old),
	// and the vertex area probabilities in the transition terms.
	// More precisely, a single perturbation of the outgoing direction at a vertex
	// v[i] induces a change in throughput equal to: f_i() * G(v[i],v[i+1]),
	// and a transition probability of p_sigma(w_o[i]) * G'(v[i],v[i+1]), where p_sigma(w_o[i])
	// is constant and G' is equal to cos_theta(v[i+1],w_o[i]) / |v[i] - v[i+1]|^2.
	// Disregarding the change in the BSDFs, the division between the geometric portion
	// of the two terms is equal to:
	//
	//   G(v[i],v[i+1]) / G'(v[i],v[i+1]) = cos_theta(v[i],w_o[i])
	//
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float spherical_perturbation_geometric_ratio(
		const cugar::Vector3f&	old_normal,
		const cugar::Vector3f&	old_out,
		const cugar::Vector3f&	new_normal,
		const cugar::Vector3f&	new_out)
	{
	#if 1
		const float old_cos_theta_o = dot(old_normal, old_out);
		const float new_cos_theta_o = dot(new_normal, new_out);

		return old_cos_theta_o ? fabsf( new_cos_theta_o / old_cos_theta_o ) : 1.0e8f;
	#else
		return 1.0f;
	#endif
	}

	struct H_geom
	{
		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		H_geom(const cugar::Vector3f& _in, const VertexGeometry& _geom, const cugar::Vector3f& _out, float _eta) :
			in(_in), out(_out), N(_geom.normal_s), T1(_geom.tangent), T2(_geom.binormal), eta(_eta), inv_eta(1.0f / _eta) {}

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		cugar::Vector3f to_local(const cugar::Vector3f& d) const
		{
			return cugar::Vector3f( dot(d,T1), dot(d,T2), dot(d,N) );
		}

		FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
		cugar::Vector3f from_local(const cugar::Vector3f& d) const
		{
			return d.x*T1 + d.y*T2 + d.z*N;
		}

		cugar::Vector3f in;
		cugar::Vector3f out;
		cugar::Vector3f N;
		cugar::Vector3f T1;
		cugar::Vector3f T2;
		float			eta;
		float			inv_eta;
	};

	// apply a spherical perturbation to the microfacet H used for an interaction
	//
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f H_perturbation(
		H_geom					old_g,
		H_geom					new_g,
		const cugar::Vector2f	z,
		const float				exp_radius = 0.05f)
	{
		// make sure 'N and 'in are in the same hemisphere
		old_g.N = dot(old_g.N, old_g.in) < 0.0f ? -old_g.N : old_g.N;
		new_g.N = dot(new_g.N, new_g.in) < 0.0f ? -new_g.N : new_g.N;

		// fetch the original microfacet
		cugar::Vector3f H = cugar::vndf_microfacet(old_g.in, old_g.out, old_g.N, old_g.inv_eta);

		// bring H into the local coordinate system of the old point
		H = old_g.to_local( H );

		// perturb it
		H = exponential_spherical_perturbation( H, z, exp_radius );

		// bring H into the world coordinate system from the new point
		H = new_g.from_local( H );

		// make sure H points in the same direction as N
		//H = (dot(new_g.N, H) < 0.0f) ? -H : H;

		// make sure H points in the same direction as V
		//H = (dot(new_g.in, H) < 0.0f) ? -H : H;

		// and reset the output direction using the original reflection|transmission mode
		if (cugar::dot(old_g.N, old_g.out) >= 0.0f)
		{
			// reflect
			return 2 * dot(new_g.in, H) * H - new_g.in;
		}
		else
		{
			// compute whether this is a ray for which we have total internal reflection,
			// i.e. if cos_theta_t2 <= 0, and in that case return zero.
			const float VoH = dot(new_g.in, H);
			const float cos_theta_i  = VoH;
			const float cos_theta_t2 = 1.f - new_g.eta * new_g.eta * (1.f - cos_theta_i * cos_theta_i);
			if (cos_theta_t2 < 0.0f)
			{
				// reflect
				return 2 * dot(new_g.in, H) * H - new_g.in;
			}
			const float cos_theta_t = -(cos_theta_i >= 0.0f ? 1.0f : -1.0f) * sqrtf(cos_theta_t2);

			// refract
			return (new_g.eta * cos_theta_i + cos_theta_t) * H - new_g.eta * new_g.in;
		}
	}
	
	// return the solid angle probability of an H-perturbation
	//
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float H_perturbation_geometric_density(
		const bool				mode,			// 0: transmission, 1: reflection
		const cugar::Vector3f	in,
		const cugar::Vector3f	out,
			  cugar::Vector3f	N,
		const float				eta)
	{
		const float inv_eta = 1.0f / eta;

		// make sure 'N and 'in are in the same hemisphere
		N = dot(N, in) < 0.0f ? -N : N;

		// fetch the original microfacet
		cugar::Vector3f H = cugar::vndf_microfacet(in,out,N,inv_eta);

		// compute the Jacobian of the transform dw_o / d_h
		if (mode)
		{
			const float dwo_dh = 4.0f * fabsf( dot(out, H) );

			return cugar::is_finite( dwo_dh ) ? dwo_dh : 1.0e8f; 
		}
		else
		{
			const float VoH = cugar::dot( in, H );
			const float LoH = cugar::dot( out, H );
			const float sqrtDenom = VoH + inv_eta * LoH;

			const float dwo_dh = (sqrtDenom * sqrtDenom) / (inv_eta * inv_eta * fabsf( LoH ));

			return cugar::is_finite( dwo_dh ) ? dwo_dh : 1.0e8f; 
		}
	}

	// return the geometric density of an H-perturbation wrt solid angle measure
	//
	template <typename PathVertexType>
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	float H_perturbation_geometric_density(
		const PathVertexType&	v,
		const cugar::Vector3f	out)
	{
		// compute the transmission mode
		const bool mode = (dot(v.geom.normal_s, v.in) * dot(v.geom.normal_s, out) >= 0.0f);

		return H_perturbation_geometric_density(
			mode,
			v.in,
			out,
			v.geom.normal_s,
			v.bsdf.get_eta( dot( v.in, v.geom.normal_s ) ) );
	}

	// apply a spherical perturbation to the microfacet H used for an interaction
	//
	template <typename PathVertexType>
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	cugar::Vector3f H_perturbation(
		const PathVertexType&	old_v,
		const cugar::Vector3f	old_out,
		const PathVertexType&	new_v,
		const cugar::Vector2f	z,
		const float				exp_radius = 0.05f)
	{
		return H_perturbation(
			H_geom( old_v.in, old_v.geom, old_out,	old_v.bsdf.get_eta( dot(old_v.in, old_v.geom.normal_s) ) ),
			H_geom( new_v.in, new_v.geom, old_out, new_v.bsdf.get_eta( dot(new_v.in, new_v.geom.normal_s) ) ),
			z,
			exp_radius );
	}

} // anonymous namespace

