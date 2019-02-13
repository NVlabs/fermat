/*
 * Fermat
 *
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <bsdf_utils.h>
#include <bsdf.h>
#include <cugar/sampling/multijitter.h>

namespace {

__global__
void precompute_glossy_reflectance_kernel(const uint32 S, float* tables)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id >= S * S * S)
		return;

	cugar::DifferentialGeometry geom;
	geom.tangent  = cugar::Vector3f(1,0,0);
	geom.binormal = cugar::Vector3f(0,1,0);
	geom.normal_s = geom.normal_g = cugar::Vector3f(0,0,1);

	const uint32 cell_index = thread_id;

	const uint32 eta_i			= (cell_index / (S*S*S));
	const uint32 base_spec_i	= (cell_index / (S*S) % S);
	const uint32 roughness_i	= (cell_index / (S)) % S;
	const uint32 theta_i		= (cell_index) % S;

	const float eta		  = 2.0f * float(eta_i + 0.5f) / float(S);
	const float base_spec = float(base_spec_i) / float(S-1);
	const float roughness = cugar::sqr( float(roughness_i) / float(S-1) ); // square the roughness
	const float cos_theta = float(theta_i) / float(S-1);

	cugar::GGXSmithBsdf bsdf(roughness);

	const cugar::Vector3f V( sqrtf(1.0f - cos_theta*cos_theta), 0.0f, cos_theta );

	float sum = 0.0f;

	const uint32 M = 4*32;
	const uint32 N = M*M;		// = 16*1024

	for (uint32 s = 0; s < N; ++s)
	{
		const cugar::Vector2f u2 = cugar::correlated_multijitter( s, M, M, cell_index, false );
		const cugar::Vector3f u(
			u2.x,
			u2.y,
			cugar::randfloat(s*3u + 2u, cell_index)
		);

		cugar::Vector3f L;
		cugar::Vector3f g;
		float			p;
		float			p_pr;

		bsdf.sample(
			u,
			geom,
			V,
			L,
			g,
			p,
			p_pr );

		const cugar::Vector3f H(cugar::normalize(V+L));

		const float VoH = dot(V,H);

		const float F = cugar::max_comp( cugar::fresnel_schlick( VoH, eta, base_spec ) );

		sum += F * g.x;
	}

	tables[cell_index] = sum / N;
}

// for a given input direction, precompute how much energy is lost to TIR
__global__
void precompute_TIR_loss_kernel(const uint32 S, float* tables)
{
	const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	if (thread_id >= S * S * S)
		return;

	cugar::DifferentialGeometry geom;
	geom.tangent  = cugar::Vector3f(1,0,0);
	geom.binormal = cugar::Vector3f(0,1,0);
	geom.normal_s = geom.normal_g = cugar::Vector3f(0,0,1);

	const uint32 cell_index = thread_id;

	const uint32 eta_i			= (cell_index / (S*S));
	const uint32 roughness_i	= (cell_index / (S)) % S;
	const uint32 theta_i		= (cell_index) % S;

	const float eta = 2.0f * float(eta_i + 0.5f) / float(S);
	const float roughness = cugar::sqr( float(roughness_i+0.5f) / float(S-1 + 0.5f) ); // square the roughness
	const float cos_theta = float(theta_i + 0.5f) / float(S-1 + 0.5f);

	cugar::GGXSmithBsdf bsdf(roughness);

	const cugar::Vector3f V( sqrtf(1.0f - cos_theta*cos_theta), 0.0f, cos_theta );

	float sum = 0.0f;

	const cugar::Vector3f Hc = geom.normal_s;

	const uint32 M = 4*32;
	const uint32 N = M*M;		// = 16*1024

	for (uint32 s = 0; s < N; ++s)
	{
		const cugar::Vector2f u2 = cugar::correlated_multijitter( s, M, M, cell_index, false );
		const cugar::Vector3f u(
			u2.x,
			u2.y,
			cugar::randfloat(s*3u + 2u, cell_index)
		);

		cugar::Vector3f L;
		cugar::Vector3f L_o;
		cugar::Vector3f g;
		float			p;
		float			p_pr;

		// sample the BSDF
		bsdf.sample(
			u,
			geom,
			V,
			L,
			g,
			p,
			p_pr );

		//
		// compute TIR due to the clearcoat layer
		//

		// flip the refracted direction to make it an input for the coating layer
		L = -L;

		// refract through the other clearcoat interface
		float Fc_2_s;
		if (!cugar::refract(L, Hc, dot(L, Hc), 1.0f / eta, &L_o, &Fc_2_s))
		{
			// accumulate the amount of energy loss due to TIR
			sum += 1.0f * g.x;
		}
	}

	tables[cell_index] = sum / float(N);
}

} // anonymous namespace

// for a given input direction, precompute how much energy is lost to TIR
void precompute_glossy_reflectance(const uint32 S, float* tables)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(S*S*S*S, blockSize));
	precompute_glossy_reflectance_kernel<<< gridSize, blockSize >>>( S, tables );
}

// for a given input direction, precompute how much energy is lost to TIR
void precompute_TIR_loss(const uint32 S, float* tables)
{
	const uint32 blockSize(128);
	const dim3 gridSize(cugar::divide_ri(S*S*S, blockSize));
	precompute_TIR_loss_kernel<<< gridSize, blockSize >>>( S, tables );
}
