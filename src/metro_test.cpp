/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <types.h>
#include <cugar/basic/algorithms.h>
#include <cugar/sampling/lfsr.h>
#include <cugar/sampling/random.h>
#include <cugar/sampling/distributions.h>
#include <cugar/sampling/latin_hypercube.h>
#include <cugar/image/tga.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>
#include <vector>
#include <set>

enum TestType
{
	KMLT1		= 1,
	KMLT2		= 2,
	KMLT_AVG	= 4,
	KMLT_MIX	= 8,
	CMLT_EPSM	= 16,
	CMLT		= 32,
	CMLT_ST		= 64,
	MC          = 128,
	RPT			= 256
};

//#define TEST (KMLT1 | KMLT2 | KMLT_AVG | KMLT_MIX | CMLT_EPSM | CMLT | CMLT_ST)
#define TEST CMLT_ST
#define HIGH_QUALITY 0
#define MC_MIX		 0

struct Path
{
	cugar::Vector3f x0;
	cugar::Vector3f x1;
};

float RMSE(const uint32 res_x, const uint32 res_y, const cugar::Vector3f* image, const cugar::Vector3f* ref_image)
{
	float e_sum = 0.0f;

	for (uint32 i = 0; i < res_x * res_y; ++i)
	{
		if (cugar::is_finite(image[i].x) && cugar::is_finite(image[i].y) && cugar::is_finite(image[i].z) &&
			cugar::is_finite(ref_image[i].x) && cugar::is_finite(ref_image[i].y) && cugar::is_finite(ref_image[i].z))
		{
			const float e2 = fabsf( dot( image[i] - ref_image[i], image[i] - ref_image[i] ) );

			e_sum += e2 / float(res_x * res_y);
		}
	}

	return sqrtf( e_sum );
}

void save_tga(const char* filename, const uint32 res_x, const uint32 res_y, const cugar::Vector3f* image)
{
	std::vector<uint8> rgba(res_x*res_y * 4, 0);
	for (uint32 i = 0; i < res_x*res_y; ++i)
	{
		rgba[i * 4 + 0] = cugar::quantize(image[i].x, 256);
		rgba[i * 4 + 1] = cugar::quantize(image[i].y, 256);
		rgba[i * 4 + 2] = cugar::quantize(image[i].z, 256);
	}
	// dump the image to a tga
	cugar::write_tga(filename, res_x, res_y, &rgba[0], cugar::TGAPixels::RGBA);
}

void splat(uint32 res_x, uint32 res_y, cugar::Vector3f* image, float2 uv, const cugar::Vector3f w)
{
	if (cugar::is_finite(w.x) && cugar::is_finite(w.y) && cugar::is_finite(w.z) &&
		!cugar::is_nan(w.x) && !cugar::is_nan(w.y) && !cugar::is_nan(w.z))
	{
		const uint32 x = cugar::quantize(uv.x, res_x);
		const uint32 y = cugar::quantize(uv.y, res_y);

		image[x + y*res_x] += w;
	}
}

void splat(uint32 res_x, uint32 res_y, cugar::Vector3f* image, Path p, const cugar::Vector3f w)
{
	splat(res_x, res_y, image, p.x0.xy(), w);
}

template <typename TRandom>
cugar::Vector2f perturb(const cugar::Vector2f u, TRandom& random)
{
	cugar::Cauchy_distribution perturbation(0.01f);

	return cugar::mod(
		cugar::Vector2f(
			u.x + perturbation.map(random.next()),
			u.y + perturbation.map(random.next())), 1.0f);
}

template <typename TRandom>
cugar::Vector4f perturb(const cugar::Vector4f u, TRandom& random)
{
	//cugar::Cauchy_distribution perturbation(0.05f);
	cugar::Cauchy_distribution perturbation(0.01f);
	//cugar::Bounded_exponential perturbation(1.0f / 16.0f);

	return cugar::mod(
		cugar::Vector4f(
			u.x + perturbation.map(random.next()),
			u.y + perturbation.map(random.next()),
			u.z + perturbation.map(random.next()),
			u.w + perturbation.map(random.next())), 1.0f);
}

float G(const cugar::Vector3f x0, const cugar::Vector3f x1)
{
	const cugar::Vector3f v = x1 - x0;
	const float d2 = dot(v, v);
	const float d = sqrtf(d2);
	return fabsf(v.z * v.y) / cugar::max( d2*d2, 1.0e-12f );
}

bool on_light1(const cugar::Vector3f x)
{
	return
		(fabsf(x.y) < 1.0e-8f) &&
		(x.x >= 0.0f && x.x <= 1.0f) &&
		(x.z >= 0.0f && x.z <= 1.0f);
}
bool on_hole1(const cugar::Vector3f x)
{
	return (x.x < 0.8 || x.x > 0.9f);
}
bool on_light2(const cugar::Vector3f x)
{
	return
		(fabsf(x.y - 1) < 1.0e-8f) &&
		(x.x >= 0.0f && x.x <= 1.0f) &&
		(x.z >= 0.0f && x.z <= 1.0f);
}
bool on_hole2(const cugar::Vector3f x)
{
	return  (x.x >= 0.05f && x.x <= 0.075f) &&
			(x.z >= 0.0f  && x.z <= 0.075f);
}
cugar::Vector3f emission(const cugar::Vector3f x)
{
//	return
//		(on_light1(x) && on_hole1(x)) ? cugar::Vector3f(x.x, x.x, 0.1f) :
//		(on_light2(x) && on_hole2(x)) ? cugar::Vector3f(2.0f, 5.0f, 1.0f) :
//		0.0f;
	return (on_light1(x) && on_hole1(x)) ? cugar::Vector3f(x.x, x.x * 0.86f + 0.1f, x.x * 0.77f + 0.15f) :
		   (on_light2(x) && on_hole2(x)) ? cugar::Vector3f(2.0f,5.0f,1.0f) :
										   0.0f;
}
cugar::Vector3f f(const Path p)
{
	return emission(p.x1) * G(p.x0, p.x1);
}

float f_max(const Path p)
{
	return cugar::max_comp(f(p));
}

// a technique which actually samples the hole of the second area light, as well as sampling the first area light perfectly
//
float map_ref(
	const float4 u,
	Path& p)
{
	p.x0 = cugar::Vector3f(u.x, u.y, 0.0f);
	if (u.z < 0.5f)
	{
		p.x1 = cugar::Vector3f(sqrtf(2 * u.z), 0.0f, u.w);
		return p.x1.x;
	}
	else
	{
		p.x1 = cugar::Vector3f(2 * (u.z - 0.5f) * 0.025f + 0.05f, 1.0f, u.w * 0.075f);
		return 0.5f / (0.025f * 0.075f);
	}
}

// a technique which actually samples the hole of the second area light, as well as sampling the first area light perfectly
//
float pdf_ref(const Path p)
{
	return  on_light1(p.x1) ? p.x1.x :
			(on_light2(p.x1) && on_hole2(p.x1)) ? 0.5f / (0.025f * 0.075f) : 0.0f;
}

float map3(
	const float4 u,
	Path& p)
{
	p.x0 = cugar::Vector3f(u.x, u.y, 0.0f);
	p.x1 = cugar::Vector3f(u.z, 0.0f, u.w);
	return 1.0f;
}

float pdf3(const Path p)
{
	return on_light1(p.x1) ? 1.0f : 0.0f;
}

float inv_map3(const Path p, float4& u)
{
	u = cugar::Vector4f(p.x0.x, p.x0.y, p.x1.x, p.x1.z);
	return 1.0f;
}

float inv_pdf3(const Path p, const float4 u)
{
	return 1.0f;
}

float map1(
	const float4 u,
	Path& p)
{
	p.x0 = cugar::Vector3f(u.x, u.y, 0.0f);
	if (u.z < 0.5f)
	{
		p.x1 = cugar::Vector3f(sqrtf(2 * u.z), 0.0f, u.w);
		return p.x1.x;
	}
	else
	{
		p.x1 = cugar::Vector3f(2 * (u.z - 0.5f), 1.0f, u.w);
		return 0.5f;
	}
}

float pdf1(const Path p)
{
	return on_light1(p.x1) ? p.x1.x :
		   on_light2(p.x1) ? 0.5f   : 0.0f;
}

float inv_map1(const Path p, float4& u)
{
	if (on_light1(p.x1))
		u = cugar::Vector4f(p.x0.x, p.x0.y, 0.5f * p.x1.x * p.x1.x, p.x1.z);
	else if (on_light2(p.x1))
		u = cugar::Vector4f(p.x0.x, p.x0.y, p.x1.x * 0.5f + 0.5f, p.x1.z);
	else
		u = cugar::Vector4f(0.0f);

	return 1.0f / pdf1(p);
}

float inv_pdf1(const Path p, const float4 u)
{
	return 1.0f / pdf1(p);
}

float map2(
	const float4 u,
	Path& p)
{
	p.x0 = cugar::Vector3f(u.x, u.y, 0.0f);

	const cugar::Vector3f dir = cugar::square_to_cosine_hemisphere( cugar::Vector2f(u.z,u.w) );

	p.x1 = p.x0 + 1.0e4f * dir;

	// intersect the ray (x0,dir) with the plane Y = 0
	{
		const float denom = dir.y;
		if (fabsf(denom) > 1.0e-8f)
		{
			const float t = -p.x0.y / denom;
			if (t >= 0)
			{
				p.x1 = cugar::Vector3f(p.x0 + t * dir);
				p.x1.y = 0.0f; // force the point to be on the plane
				return G(p.x0, p.x1) / M_PIf;
			}
		}
	}
	// intersect the ray (x0,dir) with the plane Y = 1
	{
		const float denom = dir.y;
		if (fabsf(denom) > 1.0e-8f)
		{
			const float t = (1.0f - p.x0.y) / denom;
			if (t >= 0)
			{
				p.x1 = cugar::Vector3f(p.x0 + t * dir);
				p.x1.y = 1.0f; // force the point to be on the plane
				return G(p.x0, p.x1) / M_PIf;
			}
		}
	}
	return G(p.x0, p.x1) / M_PIf;
}
float pdf2(const Path p)
{
	return G(p.x0, p.x1) / M_PIf;
}

float inv_map2(const Path p, float4& u)
{
	const cugar::Vector3f dir = cugar::normalize(p.x1 - p.x0);

	const cugar::Vector2f uv = cugar::cosine_hemisphere_to_square(dir);

	u = cugar::Vector4f(p.x0.x, p.x0.y, uv.x, uv.y);

	return 1.0f / pdf2(p);
}
float inv_pdf2(const Path p, const float4 u)
{
	return 1.0f / pdf2(p);
}

void estimate_integrals(cugar::LFSRRandomStream& random, float& f1, float& f2)
{
	const uint32 n = 16 * 1024 * 1024;

	f1 = 0.0f;
	f2 = 0.0f;

	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\restimate %.1f%%      ", 100.0f * float(i) / float(n));

		const cugar::Vector4f u(random.next(), random.next(), random.next(), random.next());

		Path  p;
		float p1;
		float p2;

		if ((i & 1) == 0)
		{
			p1 = map1(u, p);
			p2 = pdf2(p);

			if (p1 + p2)
				f1 += f_max(p) * 2.0f / (p1 + p2);
		}
		else
		{
			p2 = map2(u, p);
			p1 = pdf1(p);

			if (p1 + p2)
				f2 += f_max(p) * 2.0f / (p1 + p2);
		}
	}

	f1 /= float(n);
	f2 /= float(n);
}

void mc_test(uint32 res_x, uint32 res_y, cugar::Vector3f* image)
{
	cugar::LFSRGeneratorMatrix generator(32);
	cugar::LFSRRandomStream random(&generator, 1u);

	float integral_f  = 0.0f;
	float integral_f1 = 0.0f;
	float integral_f2 = 0.0f;

	estimate_integrals(random, integral_f1, integral_f2);

	integral_f = integral_f1 + integral_f2;

	const float norm = float(res_x * res_y) / integral_f;

	const uint32 n = 128 * 1024 * 1024;

	integral_f1 /= integral_f;
	integral_f2 /= integral_f;

	fprintf(stderr, "\nintegral : %f (%f, %f)\n", integral_f, integral_f1, integral_f2);

	std::vector<cugar::Vector4f> samples(n);
	{
		cugar::LHSampler sampler;
		sampler.sample(n, &samples[0]);
	}
	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmc %.1f%%      ", 100.0f * float(i) / float(n));

		Path  p;
		float p1;
		float p2;

		if ((i & 1) == 0)
		{
			const cugar::Vector4f u(samples[i]);

			p1 = map_ref(u, p);
			p2 = pdf2(p);

			const cugar::Vector3f c = 2.0f * f(p) / (p1 + p2);

			splat(res_x, res_y, image, p, 0.25f * c * norm / float(n));
		}
		else
		{
			const cugar::Vector4f u(samples[i]);

			p2 = map2(u, p);
			p1 = pdf_ref(p);

			const cugar::Vector3f c = 2.0f * f(p) / (p1 + p2);

			if (cugar::is_finite(p1) && cugar::is_finite(p2))
				splat(res_x, res_y, image, p, 0.25f * c * norm / float(n));
		}
	}
	fprintf(stderr, "\n");
}

struct RepresentativeSample
{
	RepresentativeSample(const uint32 _m, cugar::LFSRRandomStream& random) : m(_m), p_vec(m), f_vec(m)
	{
		float curry = 0.0f;

		for (uint32 i = 0; i < m; ++i)
		{
			const cugar::Vector4f u(random.next(), random.next(), random.next(), random.next());

			Path  p;
			float pdf;

			pdf = map1(u, p);

			p_vec[i] = p;
			f_vec[i] = (f_max(p) / pdf) + curry;

			curry = f_vec[i];
		}
		for (uint32 i = 0; i < m; ++i)
			f_vec[i] /= f_vec[m - 1];
	}

	Path sample(cugar::LFSRRandomStream& random) const
	{
		const uint32 i = cugar::upper_bound_index(random.next(), &f_vec[0], m);
		return p_vec[i];
	}

	uint32				m;
	std::vector<Path>	p_vec;
	std::vector<float>	f_vec;
};

void birth_death_process(
	const uint32				res_x,
	const uint32				res_y,
	const uint32				n,
	RepresentativeSample&		sample_set,
	cugar::LFSRRandomStream&	random,
	cugar::Vector3f*			image,
	const cugar::Vector3f*		ref_image)
{
	typedef std::vector< std::pair<cugar::Vector4f,float> >	pop_type;
	pop_type  pop;
	pop_type  pop_new;


	memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

	// resample one point to remove start-up bias
	Path p = sample_set.sample(random);

	cugar::Vector4f u;
	inv_map2(p, u);

	pop.push_back( std::make_pair(u,1.0f) );

	const float MIN_WEIGHT = 0.25f;

	uint32 n_samples = 0;
	float  w_sum     = 0.0f;

	for (uint32 i = 0; i < n; ++i)
	{
		const uint32 n_particles = uint32(pop.size());
		fprintf(stderr, "\rwmlt %.1f%%  (i:%.2f, w:%.2f, population size: %u, samples: %u)        ", 100.0f * float(n_samples)/float(n), float(i)/float(n), w_sum/float(n), n_particles, n_samples);

		pop_new.erase(pop_new.begin(), pop_new.end());

		n_samples += n_particles;

		if (n_samples >= n)
		{
			// normalize for the early termination...
			for (uint32 j = 0; j < res_x*res_y; ++j)
				image[j] *= float(n) / float(w_sum);

			break;
		}

		if (pop.size())
		{
			for (uint32 j = 0; j < pop.size(); ++j)
			{
				Path path_old, path_new;

				      float           w_old = pop[j].second;
				const cugar::Vector4f u_old = pop[j].first;
				const float			  p_old = map2(u_old, path_old);
				const cugar::Vector3f f_old = f(path_old) / p_old;

				// accumulate the weighted sample
				{
					cugar::Vector3f c = w_old * f(path_old) / f_max(path_old);

					splat(res_x, res_y, image, path_old, 0.25f * c * float(res_x * res_y) / float(n));

					w_sum += w_old;
				}

				const cugar::Vector4f u_new = perturb(u_old, random);
				const float			  p_new = map2(u_new, path_new);
				const cugar::Vector3f f_new = f(path_new) / p_new;

				const float ratio = max_comp(f_new) / max_comp(f_old);

				if (ratio > 1.0f)
					pop_new.push_back(std::make_pair(u_new,w_old));
				else
				{
					float w_new = w_old * ratio;
					      w_old = w_old * (1.0f - ratio);

					if (w_old >= MIN_WEIGHT)
						pop_new.push_back(std::make_pair(u_old, w_old));
					else
					{
						if (random.next() < w_old / MIN_WEIGHT)
							pop_new.push_back(std::make_pair(u_old, MIN_WEIGHT));
					}

					if (w_new >= MIN_WEIGHT)
						pop_new.push_back(std::make_pair(u_new, w_new));
					else
					{
						if (random.next() < w_new / MIN_WEIGHT)
							pop_new.push_back(std::make_pair(u_new, MIN_WEIGHT));
					}
				}
			}
		}
		else
		{
			//fprintf(stderr, "\nrestart (i: %u, samples: %u)\n", i, n_samples);
			Path p = sample_set.sample(random);

			cugar::Vector4f u;
			inv_map2(p, u);

			pop_new.push_back(std::make_pair(u,1.0f));
		}

		std::swap(pop, pop_new);
	}

	const float rmse = RMSE(res_x, res_y, image, ref_image);
	fprintf(stderr, "  RMSE: %f\n", rmse);

	if (HIGH_QUALITY)
		save_tga("test_wmlt.tga", res_x, res_y, image);
	else
		save_tga("test_wmlt_low.tga", res_x, res_y, image);
}

void metropolis_test(uint32 res_x, uint32 res_y, cugar::Vector3f* image, const cugar::Vector3f* ref_image)
{
	// run two chains for n steps
	const uint32 n = (HIGH_QUALITY ? 128 : 16) * 1024 * 1024;
	const uint32 m = 100000; // discard this many samples

	cugar::LFSRGeneratorMatrix generator(32);
	cugar::LFSRRandomStream random(&generator, 1u);

	float integral_f = 0.0f;
	float integral_f1 = 0.0f;
	float integral_f2 = 0.0f;

	estimate_integrals(random, integral_f1, integral_f2);

	integral_f = integral_f1 + integral_f2;

	const float norm = float(res_x * res_y) / integral_f;

	integral_f1 /= integral_f;
	integral_f2 /= integral_f;

	fprintf(stderr, "\nintegral : %f (%f, %f)\n", integral_f, integral_f1, integral_f2);

	RepresentativeSample sample_set(m, random);

	birth_death_process(
		res_x,
		res_y,
		n,
		sample_set,
		random,
		image,
		ref_image );

	if (TEST & RPT)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		const uint32 spp = 16;

		struct PixelSamples
		{
			Path			path;
			cugar::Vector3f f;
			float			p;
		};

		std::vector<cugar::Vector4f> shifts(res_x * res_y);
		for (uint32 i = 0; i < res_x * res_y; ++i)
			shifts[i] = cugar::Vector4f(random.next(), random.next(), random.next(), random.next());

		const uint32 TILE_SIZE = 32;

		std::vector<float> cdf(TILE_SIZE*TILE_SIZE);

		std::vector<PixelSamples> samples(res_x * res_y);
		for (uint32 s = 0; s < spp; ++s)
		{
			printf("\rrpt sampling %.1f%%      ", 100.0f * float(s) / float(spp));

			const uint32 shift_x = 0; cugar::quantize(random.next(), 8);
			const uint32 shift_y = 0; cugar::quantize(random.next(), 8);

			for (uint32 i = 0; i < res_x * res_y; ++i)
			{
				const cugar::Vector4f shift = shifts[i];

				cugar::Vector4f u = cugar::mod(shift + cugar::Vector4f(random.next(), random.next(), random.next(), random.next()), 1.0f);

				const uint32 x = i % res_x;
				const uint32 y = i / res_x;

				u.x = (u.x + float(x)) / float(res_x);
				u.y = (u.y + float(y)) / float(res_y);

				Path p;
				samples[i].p = map1(u, p);
				samples[i].f = f(p);
				samples[i].path = p;

				//splat(res_x, res_y, image, p, 0.25f * (samples[i].f / samples[i].p) / float(spp * integral_f));

				const float p2 = map2(u, p);
				const float g = G(p.x0, p.x1);
				const float mis_w = 1.0f - fminf(1.0f, 10.0f / g);

				const cugar::Vector3f c = f(p) / p2;

				splat(res_x, res_y, image, p, 0.25f * mis_w * c / float(spp * integral_f));
			}

		#if 1
			for (uint32 tile_y = 0; tile_y < (res_y + TILE_SIZE - 1) / TILE_SIZE; ++tile_y)
			{
				for (uint32 tile_x = 0; tile_x < (res_x + TILE_SIZE - 1) / TILE_SIZE; ++tile_x)
				{
					const uint32 lx = tile_x * TILE_SIZE;
					const uint32 ly = tile_y * TILE_SIZE;
					const uint32 rx = lx + TILE_SIZE < res_x ? lx + TILE_SIZE : res_x;
					const uint32 ry = ly + TILE_SIZE < res_y ? ly + TILE_SIZE : res_y;

					// build a CDF
					{
						float sum = 0.0f;

						for (uint32 y = ly; y < ry; ++y)
						{
							for (uint32 x = lx; x < rx; ++x)
							{
								sum += cugar::max_comp(samples[x + y * res_x].f) / samples[x + y * res_x].p;

								cdf[x - lx + (y - ly)*(rx - lx)] = sum;
							}
						}

						for (uint32 i = 0; i < (rx - lx)*(ry - ly); ++i)
							cdf[i] /= sum;

						for (uint32 i = 0; i < (rx - lx)*(ry - ly); ++i)
							cdf[i] = (0.99f * cdf[i] + 0.01f * float(i+1) / float((rx - lx)*(ry - ly)));
					}

					// pick one sample out of the cdf
					for (uint32 y = ly; y < ry; ++y)
					{
						for (uint32 x = lx; x < rx; ++x)
						{
							const uint32 i = x + y*res_x;

							const uint32 tile_j = cugar::upper_bound_index(random.next(), cdf.begin(), (rx - lx)*(ry - ly));

							const float pdf_j = (cdf[tile_j] - (tile_j ? cdf[tile_j - 1] : 0)) * float((rx - lx)*(ry - ly));

							const uint32 x_j = (tile_j % (rx - lx)) + lx;
							const uint32 y_j = (tile_j / (rx - lx)) + ly;
							const uint32 j = x_j + y_j * res_x;

							Path p;
							p.x0 = samples[i].path.x0;
							p.x1 = samples[j].path.x1;

							const cugar::Vector3f c = f(p) / (pdf_j * samples[j].p);

							const float g = G(p.x0, p.x1);
							const float mis_w = fminf(1.0f, 10.0f / g);

							splat(res_x, res_y, image, p, 0.25f * mis_w * c / float(spp * integral_f));
						}
					}
				}
			}
		#else
			const uint32 N_COLORS  = 2;

			for (uint32 tile_y = 0; tile_y < (res_y + TILE_SIZE - 1) / TILE_SIZE; ++tile_y)
			{
				for (uint32 tile_x = 0; tile_x < (res_x + TILE_SIZE - 1) / TILE_SIZE; ++tile_x)
				{
					// build a random permutation of TILE_SIZE colors, just for this tile
					uint32 colors[TILE_SIZE * TILE_SIZE];
					for (uint32 i = 0; i < TILE_SIZE * TILE_SIZE; ++i)
						colors[i] = i % N_COLORS;

					for (uint32 i = 0; i < TILE_SIZE * TILE_SIZE; ++i)
					{
						const uint32 j = i + cugar::quantize(random.next(), TILE_SIZE * TILE_SIZE - i);
						std::swap(colors[i], colors[j]);
					}

					const uint32 lx = tile_x * TILE_SIZE;
					const uint32 ly = tile_y * TILE_SIZE;
					const uint32 rx = lx + TILE_SIZE < res_x ? lx + TILE_SIZE : res_x;
					const uint32 ry = ly + TILE_SIZE < res_y ? ly + TILE_SIZE : res_y;

					// compute balanced pdfs
					for (uint32 y = ly; y < ry; ++y)
					{
						for (uint32 x = lx; x < rx; ++x)
						{
							const uint32 color_i = colors[(x - lx) + (y - ly)*TILE_SIZE];

							const uint32 i = x + y*res_x;

							for (uint32 yy = ly; yy < ry; ++yy)
							{
								for (uint32 xx = lx; xx < rx; ++xx)
								{
									const uint32 color_j = colors[(xx - lx) + (yy - ly)*TILE_SIZE];

									if (color_j != color_i)
										continue;

									const uint32 j = xx + yy*res_x;

									Path p;
									p.x0 = samples[i].path.x0;
									p.x1 = samples[j].path.x1;

									if (i != j)
										samples[j].p += pdf2(p);
								}
							}
						} // loop over x
					} // loop over y

					// reuse and accumulate
					for (uint32 y = ly; y < ry; ++y)
					{
						for (uint32 x = lx; x < rx; ++x)
						{
							const uint32 color_i = colors[(x - lx) + (y - ly)*TILE_SIZE];

							const uint32 i = x + y*res_x;

							for (uint32 yy = ly; yy < ry; ++yy)
							{
								for (uint32 xx = lx; xx < rx; ++xx)
								{
									const uint32 color_j = colors[(xx - lx) + (yy - ly)*TILE_SIZE];

									if (color_j != color_i)
										continue;

									const uint32 j = xx + yy*res_x;

									Path p;
									p.x0 = samples[i].path.x0;
									p.x1 = samples[j].path.x1;

									const float w = 1.0f / samples[j].p;
									//const float w = 1.0f / (samples[j].p * (rx - lx)*(ry - ly));
									//const float w = 2.0f / ((samples[j].p + pdf2(p)) * (rx - lx)*(ry - ly));

									const cugar::Vector3f c = f(p) * w;

									splat(res_x, res_y, image, p, 0.25f * c / float(spp * integral_f));
								}
							}
						} // loop over x
					} // loop over y
				} // loop over tile_x
			} // loop over tile_y
		#endif

	#if 0
		#if 1
			// compute balanced pdfs
			for (uint32 i = 0; i < res_x * res_y; ++i)
			{
				const uint32 x = i % res_x;
				const uint32 y = i / res_x;

				const uint32 tile_x = (x + shift_x) / 16;
				const uint32 tile_y = (y + shift_y) / 16;

				const uint32 color = ((x + shift_x) & 1u) * ((y + shift_y) & 1u);

				const uint32 lx = tile_x * 16;
				const uint32 ly = tile_y * 16;
				const uint32 rx = lx + 16 < res_x ? lx + 16 : res_x;
				const uint32 ry = ly + 16 < res_y ? ly + 16 : res_y;

				for (uint32 yy = ly; yy < ry; ++yy)
				{
					for (uint32 xx = lx; xx < rx; ++xx)
					{
						if (color != ((xx + shift_x) & 1u) * ((yy + shift_y) & 1u))
							continue;

						const uint32 j = xx + yy*res_x;

						const uint32 local_j = (xx - lx) + (yy - ly)*(rx - lx);

						Path p;
						p.x0 = samples[i].path.x0;
						p.x1 = samples[j].path.x1;

						if (i != j)
							samples[j].p += pdf2(p);
					}
				}
			}
		#endif

			for (uint32 i = 0; i < res_x * res_y; ++i)
			{
				const uint32 x = i % res_x;
				const uint32 y = i / res_x;

			#if 1
				const uint32 tile_x = (x + shift_x) / 16;
				const uint32 tile_y = (y + shift_y) / 16;

				const uint32 color = ((x + shift_x) & 1u) * ((y + shift_y) & 1u);

				const uint32 lx = tile_x * 16;
				const uint32 ly = tile_y * 16;
				const uint32 rx = lx + 16 < res_x ? lx + 16 : res_x;
				const uint32 ry = ly + 16 < res_y ? ly + 16 : res_y;
			#else
				const uint32 lx = x > 3 ? x - 3 : 0;
				const uint32 ly = y > 3 ? y - 3 : 0;
				const uint32 rx = x < res_x - 4 ? x + 4 : res_x;
				const uint32 ry = y < res_y - 4 ? y + 4 : res_y;
			#endif

				// reuse all the samples from the neighborhood
				for (uint32 yy = ly; yy < ry; ++yy)
				{
					for (uint32 xx = lx; xx < rx; ++xx)
					{
						if (color != ((xx + shift_x) & 1u) * ((yy + shift_y) & 1u))
							continue;

						const uint32 j = xx + yy*res_x;

						const uint32 local_j = (xx - lx) + (yy - ly)*(rx - lx);

						Path p;
						p.x0 = samples[i].path.x0;
						p.x1 = samples[j].path.x1;

						//const float w = 1.0f / (samples[j].p * (rx - lx)*(ry - ly));
						const float w = 1.0f / samples[j].p;
						//const float w = 2.0f / ((samples[j].p + pdf2(p)) * (rx - lx)*(ry - ly));

						const cugar::Vector3f c = f(p) * w;

						splat(res_x, res_y, image, p, 0.25f * c / float(spp * integral_f));
					}
				}
			}
	#endif
		} // loop over s

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_rpt.tga", res_x, res_y, image);
		else
			save_tga("test_rpt_low.tga", res_x, res_y, image);
	}
	if (TEST & MC)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		for (uint32 i = 0; i < n; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rmc %.1f%%      ", 100.0f * float(i) / float(n));

			Path  p;
			float p1;
			float p2;

			if ((i & 1) == 0)
			{
				const cugar::Vector4f u(random.next(), random.next(), random.next(), random.next());

				p1 = map1(u, p);
				p2 = pdf2(p);

				const cugar::Vector3f c = 2.0f * f(p) / (p1 + p2);

				splat(res_x, res_y, image, p, 0.25f * c * norm / float(n));
			}
			else
			{
				const cugar::Vector4f u(random.next(), random.next(), random.next(), random.next());

				p2 = map2(u, p);
				p1 = pdf1(p);

				const cugar::Vector3f c = 2.0f * f(p) / (p1 + p2);

				if (cugar::is_finite(p1) && cugar::is_finite(p2))
					splat(res_x, res_y, image, p, 0.25f * c * norm / float(n));
			}
		}
		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_mc.tga", res_x, res_y, image);
		else
			save_tga("test_mc_low.tga", res_x, res_y, image);
	}
	if (TEST & KMLT1)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		// resample one point to remove start-up bias
		Path p = sample_set.sample(random);

		cugar::Vector4f u;
		inv_map1(p, u);

		for (uint32 i = 0; i < n; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rkmlt-1 %.1f%%      ", 100.0f * float(i) / float(n));

			Path			p_old, p_new;
			cugar::Vector4f u_old, u_new;

			p_old = p;
			u_old = u;

			u_new = perturb(u_old, random);
			float pdf_new = map1(u_new, p_new);
			float pdf_old = pdf1(p_old);

			const float f_old = f_max(p_old) / pdf_old;
			const float f_new = f_max(p_new) / pdf_new;

			const float a = fminf(1.0f, f_new / f_old);
			if (random.next() < a)
			{
				p = p_new;
				u = u_new;
			}

			cugar::Vector3f c = f(p) / f_max(p);

			splat(res_x, res_y, image, p, 0.25f * c * float(res_x * res_y) / float(n));
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_kmlt1.tga", res_x, res_y, image);
		else
			save_tga("test_kmlt1_low.tga", res_x, res_y, image);
	}
	if (TEST & KMLT2)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		// resample one point to remove start-up bias
		Path p = sample_set.sample(random);

		cugar::Vector4f u;
		inv_map2(p, u);

		for (uint32 i = 0; i < n; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rkmlt-2 %.1f%%      ", 100.0f * float(i) / float(n));

			Path			p_old, p_new;
			cugar::Vector4f u_old, u_new;

			p_old = p;
			u_old = u;

			u_new = perturb(u_old, random);
			float pdf_new = map2(u_new, p_new);
			float pdf_old = pdf2(p_old);

			const float f_old = f_max(p_old) / pdf_old;
			const float f_new = f_max(p_new) / pdf_new;

			const float a = fminf(1.0f, f_new / f_old);
			if (random.next() < a)
			{
				p = p_new;
				u = u_new;
			}

			cugar::Vector3f c = f(p) / f_max(p);

			splat(res_x, res_y, image, p, 0.25f * c * float(res_x * res_y) / float(n));
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_kmlt2.tga", res_x, res_y, image);
		else
			save_tga("test_kmlt2_low.tga", res_x, res_y, image);
	}
	if (TEST & KMLT_AVG)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));
		{
			// resample one point to remove start-up bias
			Path p = sample_set.sample(random);

			cugar::Vector4f u;
			inv_map1(p, u);

			for (uint32 i = 0; i < n / 2; ++i)
			{
				if ((i % 1000) == 0)
					printf("\rkmlt-avg %.1f%%      ", 100.0f * float(i) / float(n));

				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);
				float pdf_new = map1(u_new, p_new);
				float pdf_old = pdf1(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					p = p_new;
					u = u_new;
				}

				const float w = pdf1(p) / (pdf1(p) + pdf2(p));

				cugar::Vector3f c = f(p) / f_max(p);

				splat(res_x, res_y, image, p, 0.25f * w * c * float(res_x * res_y) / float(n / 2));
			}
		}
		{
			// resample one point to remove start-up bias
			Path p = sample_set.sample(random);

			cugar::Vector4f u;
			inv_map2(p, u);

			for (uint32 i = 0; i < n / 2; ++i)
			{
				if ((i % 1000) == 0)
					printf("\rkmlt-avg %.1f%%      ", 100.0f * float(i + n / 2) / float(n));

				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);
				float pdf_new = map2(u_new, p_new);
				float pdf_old = pdf2(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					p = p_new;
					u = u_new;
				}

				const float w = pdf2(p) / (pdf1(p) + pdf2(p));

				cugar::Vector3f c = f(p) / f_max(p);

				splat(res_x, res_y, image, p, 0.25f * w * c * float(res_x * res_y) / float(n / 2));
			}
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_kmlt_avg.tga", res_x, res_y, image);
		else
			save_tga("test_kmlt_avg_low.tga", res_x, res_y, image);
	}
	if (TEST & KMLT_MIX)
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));
		{
			// resample one point to remove start-up bias
			Path p = sample_set.sample(random);

			cugar::Vector4f u;
			inv_map1(p, u);

			for (uint32 i = 0; i < n / 2; ++i)
			{
				if ((i % 1000) == 0)
					printf("\rkmlt-mix %.1f%%      ", 100.0f * float(i) / float(n));

				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);
				float pdf_new = map1(u_new, p_new);
				float pdf_old = pdf1(p_old);

				pdf_new += pdf2(p_new);
				pdf_old += pdf2(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					p = p_new;
					u = u_new;
				}

				cugar::Vector3f c = f(p) / f_max(p);

				splat(res_x, res_y, image, p, 0.25f * c * integral_f1 * float(res_x * res_y) / float(n / 2));
			}
		}
		{
			// resample one point to remove start-up bias
			Path p = sample_set.sample(random);

			cugar::Vector4f u;
			inv_map2(p, u);

			for (uint32 i = 0; i < n / 2; ++i)
			{
				if ((i % 1000) == 0)
					printf("\rkmlt-mix %.1f%%      ", 100.0f * float(i + n / 2) / float(n));

				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);
				float pdf_new = map2(u_new, p_new);
				float pdf_old = pdf2(p_old);

				pdf_new += pdf1(p_new);
				pdf_old += pdf1(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					p = p_new;
					u = u_new;
				}

				cugar::Vector3f c = f(p) / f_max(p);

				splat(res_x, res_y, image, p, 0.25f * c * integral_f2 * float(res_x * res_y) / float(n / 2));
			}
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_kmlt_mix.tga", res_x, res_y, image);
		else
			save_tga("test_kmlt_mix_low.tga", res_x, res_y, image);
	}
	if (TEST & CMLT_EPSM) // CMLT with Ephimeral Primary Space Mutations
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		// resample one point to remove start-up bias
		Path p = sample_set.sample(random);

		for (uint32 i = 0; i < n; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rcmlt-epsm %.1f%%      ", 100.0f * float(i) / float(n));

			if ((i % 2) == 0)
			{
				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				float rdf_old = inv_map1(p_old, u_old);
				u_new = perturb(u_old, random);
				float pdf_new = map1(u_new, p_new);

				const float pdf_old = pdf1(p_old);
				const float rdf_new = inv_pdf1(p_new, u_new);

				const float f_old = f_max(p_old);
				const float f_new = f_max(p_new);

				const float T_old = rdf_new /** pdf_old*/;
				const float T_new = rdf_old /** pdf_new*/;

				const float a = fminf(1.0f, (f_new * T_old) / (f_old * T_new));
				if (random.next() < a)
					p = p_new;
			}
			else if ((i % 2) == 1)
			{
				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				float rdf_old = inv_map2(p_old, u_old);
				u_new = perturb(u_old, random);
				float pdf_new = map2(u_new, p_new);

				const float pdf_old = pdf2(p_old);
				const float rdf_new = inv_pdf2(p_new, u_new);

				const float f_old = f_max(p_old);
				const float f_new = f_max(p_new);

				const float T_old = rdf_new /** pdf_old*/;
				const float T_new = rdf_old /** pdf_new*/;

				const float a = fminf(1.0f, (f_new * T_old) / (f_old * T_new));
				if (random.next() < a)
					p = p_new;
			}
			else
			{
				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				float rdf_old = inv_map3(p_old, u_old);
				u_new = perturb(u_old, random);
				float pdf_new = map3(u_new, p_new);

				const float pdf_old = pdf3(p_old);
				const float rdf_new = inv_pdf3(p_new, u_new);

				const float f_old = f_max(p_old);
				const float f_new = f_max(p_new);

				const float T_old = rdf_new /** pdf_old*/;
				const float T_new = rdf_old /** pdf_new*/;

				const float a = fminf(1.0f, (f_new * T_old) / (f_old * T_new));
				if (random.next() < a)
					p = p_new;
			}

			cugar::Vector3f c = f(p) / f_max(p);

			splat(res_x, res_y, image, p, 0.25f * c * float(res_x * res_y) / float(n));
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_cmlt_epsm.tga", res_x, res_y, image);
		else
			save_tga("test_cmlt_epsm_low.tga", res_x, res_y, image);
	}
	if (TEST & CMLT) // CMLT
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		// resample one point to remove start-up bias
		Path p1 = sample_set.sample(random);
		Path p2 = sample_set.sample(random);

		cugar::Vector4f u1;
		cugar::Vector4f u2;

		inv_map1(p1, u1);
		inv_map2(p2, u2);

		const uint32 chain_length = n / 2;

		for (uint32 i = 0; i < chain_length; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rcmlt %.1f%%      ", 100.0f * float(i) / float(n / 2));

			if ((i % 4) == 0)
			{
				cugar::Vector4f u2_1, u1_2;

				// try a swap
				const float j2_1 = inv_map2(p1, u2_1);
				const float j1_2 = inv_map1(p2, u1_2);

				const float j1_1 = inv_pdf1(p1, u1);
				const float j2_2 = inv_pdf2(p2, u2);

				const float a = fminf(1.0f, (j1_1 * j2_2) / (j2_1 * j1_2));
				if (random.next() < a)
				{
					u1 = u1_2;
					u2 = u2_1;
					std::swap(p1, p2);
				}
			}

			if (1)
			{
				// perturb the first chain
				{
					Path			p_old, p_new;
					cugar::Vector4f u_old, u_new;

					p_old = p1;
					u_old = u1;

					u_new = perturb(u_old, random);

					float pdf_new = map1(u_new, p_new);
					float pdf_old = pdf1(p_old);

					pdf_new += pdf2(p_new);
					pdf_old += pdf2(p_old);

					const float f_old = f_max(p_old) / pdf_old;
					const float f_new = f_max(p_new) / pdf_new;

					const float a = fminf(1.0f, f_new / f_old);
					if (random.next() < a)
					{
						u1 = u_new;
						p1 = p_new;
					}
				}

				// perturb the second chain
				{
					Path			p_old, p_new;
					cugar::Vector4f u_old, u_new;

					p_old = p2;
					u_old = u2;

					u_new = perturb(u_old, random);

					float pdf_new = map2(u_new, p_new);
					float pdf_old = pdf2(p_old);

					pdf_new += pdf1(p_new);
					pdf_old += pdf1(p_old);

					const float f_old = f_max(p_old) / pdf_old;
					const float f_new = f_max(p_new) / pdf_new;

					const float a = fminf(1.0f, f_new / f_old);
					if (random.next() < a)
					{
						u2 = u_new;
						p2 = p_new;
					}
				}
			}
			else
			{
				// draw an independent proposal for the first chain
				{
					Path			p_old, p_new;
					cugar::Vector4f u_old, u_new;

					p_old = p1;
					u_old = u1;

					u_new = cugar::Vector4f(random.next(), random.next(), random.next(), random.next());

					float pdf_new = map1(u_new, p_new);
					float pdf_old = pdf1(p_old);

					pdf_new += pdf2(p_new);
					pdf_old += pdf2(p_old);

					const float f_old = f_max(p_old) / pdf_old;
					const float f_new = f_max(p_new) / pdf_new;

					const float a = fminf(1.0f, f_new / f_old);
					if (random.next() < a)
					{
						u1 = u_new;
						p1 = p_new;
					}
				}

				// draw an independent proposal for the second chain
				{
					Path			p_old, p_new;
					cugar::Vector4f u_old, u_new;

					p_old = p2;
					u_old = u2;

					u_new = cugar::Vector4f(random.next(), random.next(), random.next(), random.next());

					float pdf_new = map2(u_new, p_new);
					float pdf_old = pdf2(p_old);

					pdf_new += pdf1(p_new);
					pdf_old += pdf1(p_old);

					const float f_old = f_max(p_old) / pdf_old;
					const float f_new = f_max(p_new) / pdf_new;

					const float a = fminf(1.0f, f_new / f_old);
					if (random.next() < a)
					{
						u2 = u_new;
						p2 = p_new;
					}
				}
			}

		  #if MC_MIX
			const float pdf_mlt1 = (1.0f / integral_f) * f_max(p1) / (pdf1(p1) + pdf2(p1));
			const float pdf_mlt2 = (1.0f / integral_f) * f_max(p2) / (pdf1(p2) + pdf2(p2));
			const float pdf_qmc = 1.0f;//float(n) / sqrtf(float(n));
			const float w1 = pdf_mlt1 / (pdf_qmc + pdf_mlt1);
			const float w2 = pdf_mlt2 / (pdf_qmc + pdf_mlt2);
		  #else
			const float w1 = 1.0f;
			const float w2 = 1.0f;
		  #endif

			const cugar::Vector3f c1 = w1 * f(p1) / f_max(p1);
			const cugar::Vector3f c2 = w2 * f(p2) / f_max(p2);

			splat(res_x, res_y, image, p1, 0.25f * c1 * integral_f1 * float(res_x * res_y) / float(chain_length));
			splat(res_x, res_y, image, p2, 0.25f * c2 * integral_f2 * float(res_x * res_y) / float(chain_length));
		}
	  #if MC_MIX
		for (uint32 i = 0; i < chain_length; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rcmlt %.1f%%      ", 100.0f * float(i + n / 4) / float(n / 2));

			// draw an independent proposal for the first chain
			{
				Path			p;
				cugar::Vector4f u;

				u = cugar::Vector4f(random.next(), random.next(), random.next(), random.next());

				float pdf  = map1(u, p);
					  pdf += pdf2(p);

				const float pdf_mlt = (1.0f / integral_f) * f_max(p) / pdf;
				const float pdf_mc  = 1.0f; //float(n) / sqrtf(float(n));
				const float w = pdf_mc / (pdf_mc + pdf_mlt);
				const cugar::Vector3f c = w * f(p) / pdf;

				splat(res_x, res_y, image, p, 0.25f * c * norm / float(chain_length));
			}

			// draw an independent proposal for the second chain
			{
				Path			p;
				cugar::Vector4f u;

				u = cugar::Vector4f(random.next(), random.next(), random.next(), random.next());

				float pdf  = map2(u, p);
					  pdf += pdf1(p);

				const float pdf_mlt = (1.0f / integral_f) * f_max(p) / pdf;
				const float pdf_mc  = 1.0f; //float(n) / sqrtf(float(n));
				const float w = pdf_mc / (pdf_mc + pdf_mlt);
				const cugar::Vector3f c = w * f(p) / pdf;

				splat(res_x, res_y, image, p, 0.25f * c * norm / float(chain_length));
			}
		}
	  #endif

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_cmlt.tga", res_x, res_y, image);
		else
			save_tga("test_cmlt_low.tga", res_x, res_y, image);
	}
	if (TEST & CMLT_ST) // CMLT with Simulated-Tempering
	{
		memset(&image[0], 0x00, res_x*res_y * sizeof(cugar::Vector3f));

		// resample one point to remove start-up bias
		Path p = sample_set.sample(random);

		cugar::Vector4f u;
		uint32          t = 0;

		inv_map1(p, u);

		const uint32 chain_length = n;

		float  ar_swaps = 0.0f;
		uint32 swaps = 0;

		for (uint32 i = 0; i < chain_length; ++i)
		{
			if ((i % 1000) == 0)
				printf("\rcmlt-st %.1f%%    (%.3f ar) ", 100.0f * float(i) / float(chain_length), swaps ? float(ar_swaps)/swaps : 0);

			if ((i % 4) == 0)
			{
				if (t == 0)
				{
					cugar::Vector4f v;

					// try a swap
					const float j_2 = inv_map2(p, v);
					const float j_1 = inv_pdf1(p, u);

					const float a = fminf(1.0f, (integral_f2/integral_f1) * (j_1 / j_2));
					if (random.next() < a)
					{
						u = v;
						t = 1;
					}

					ar_swaps += a;
				}
				else
				{
					cugar::Vector4f v;

					// try a swap
					const float j_1 = inv_map1(p, v);
					const float j_2 = inv_pdf2(p, u);

					const float a = fminf(1.0f, (integral_f1 / integral_f2) * (j_2 / j_1));
					if (random.next() < a)
					{
						u = v;
						t = 0;
					}

					ar_swaps += a;
				}

				swaps++;
			}

			if (t == 0) // perturb the first chain
			{
				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);

				float pdf_new = map1(u_new, p_new);
				float pdf_old = pdf1(p_old);

				pdf_new += pdf2(p_new);
				pdf_old += pdf2(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					u = u_new;
					p = p_new;
				}
			}
			else // perturb the second chain
			{
				Path			p_old, p_new;
				cugar::Vector4f u_old, u_new;

				p_old = p;
				u_old = u;

				u_new = perturb(u_old, random);

				float pdf_new = map2(u_new, p_new);
				float pdf_old = pdf2(p_old);

				pdf_new += pdf1(p_new);
				pdf_old += pdf1(p_old);

				const float f_old = f_max(p_old) / pdf_old;
				const float f_new = f_max(p_new) / pdf_new;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
				{
					u = u_new;
					p = p_new;
				}
			}

			const cugar::Vector3f c = f(p) / f_max(p);

			splat(res_x, res_y, image, p, 0.5f * c * (t == 0 ? integral_f1 : integral_f2) * float(res_x * res_y) / float(chain_length));
		}

		const float rmse = RMSE(res_x, res_y, image, ref_image);
		fprintf(stderr, "  RMSE: %f\n", rmse);

		if (HIGH_QUALITY)
			save_tga("test_cmlt_st.tga", res_x, res_y, image);
		else
			save_tga("test_cmlt_low_st.tga", res_x, res_y, image);
	}
}

#if 0

// two ways to sample the triangle

cugar::Vector2f map1(const cugar::Vector2f uv)
{
	if (uv.x < 0.1)
	{
		return cugar::Vector2f(uv.x*5.0f, uv.y);
	}
	else
		return cugar::Vector2f((uv.x - 0.1f) / 0.9f, uv.y);
}
float pdf1(const cugar::Vector2f p)
{
	return p.x < 0.5f ?
		0.2f + 0.9f :
		0.9f;
}
cugar::Vector2f inv_map1_stochastic(cugar::Vector2f p, cugar::FLCG_random& random)
{
	if (p.x < 0.5f)
	{
		if (random.next() < 0.1f)
			return cugar::Vector2f(p.x / 5.0f, p.y); // return the first fiber
		else
			return cugar::Vector2f(p.x*0.9f + 0.1f, p.y); // return the second fiber
	}
	return cugar::Vector2f(p.x*0.9f + 0.1f, p.y);
}

float inv_pdf1_stochastic(cugar::Vector2f u)
{
	return u.x < 0.1f  ? (0.5f * 0.1f) / 0.1f :
		   u.x < 0.55f ? (0.5f * 0.9f) / 0.45f :
						(1.0f - 0.5f*0.1f - 0.5f*0.9f) / 0.45f;
}

cugar::Vector2f inv_map1(cugar::Vector2f p)
{
	return cugar::Vector2f(p.x*0.9f + 0.1f, p.y); // return only one fiber
}

float inv_pdf1(cugar::Vector2f u)
{
	return u.x > 0.1f ? 1.0f / 0.9f : 0.0f;
}

cugar::Vector2f map2(const cugar::Vector2f uv)
{
	return cugar::Vector2f(uv.y, uv.x);
}
float pdf2(const cugar::Vector2f p)
{
	return 1.0f;
}
cugar::Vector2f inv_map2(cugar::Vector2f p)
{
	return cugar::Vector2f(p.y, p.x);
}
float inv_pdf2(cugar::Vector2f u)
{
	return 1.0f;
}
float f(const cugar::Vector2f p)
{
	return p.x + p.y < 1.0f ? fmodf(p.x*p.x * 10.0f, 1.0f) : 0.0f;
}

cugar::Vector2f perturb(const cugar::Vector2f u, cugar::FLCG_random& random)
{
	cugar::Cauchy_distribution cauchy(0.01f);

	return cugar::mod(
		cugar::Vector2f(
			u.x + cauchy.map(random.next()),
			u.y + cauchy.map(random.next())), 1.0f);
}

void metropolis_test(uint32 res_x, uint32 res_y, float* image)
{
	// run two chains for n steps
	const float n = 200000000;
	const float m = 100000; // discard this many samples

	cugar::FLCG_random random;

	cugar::Vector2f u1(random.next(), random.next());
	cugar::Vector2f u2(random.next(), random.next());
	for (uint32 i = 0; i < m; ++i)
	{
		u2.x = random.next();
		u2.y = random.next();
		if (f(map2(u2)) > 0.0f)
			break;
	}

	float integral_f = 0.0f;

	for (uint32 i = 0; i < n/10; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmc %.1f%%      ", 100.0f * float(i) / float(n/10));

		cugar::Vector2f p(random.next(), random.next());

		integral_f += f(p);
	}
	integral_f /= float(n / 10);

	const float norm = float(res_x*res_y) / integral_f;

	fprintf(stderr, "integral : %f\n", integral_f);

#if 1
	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmc %.1f%%      ", 100.0f * float(i) / float(n));

		cugar::Vector2f p(random.next(), random.next());

		splat(res_x, res_y, image, p, 0.1f * f(p) * norm / float(n));
	}
#elif 0
	cugar::Vector2f p = map1(u1);

	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmetropolis test %.1f%%      ", 100.0f * float(i)/float(n));

		if ((i & 1) == 0)
		{
			const cugar::Vector2f p_old = p;
			const cugar::Vector2f u_old = inv_map1_stochastic(p_old, random);
			const cugar::Vector2f u_new = perturb(u_old, random);
			const cugar::Vector2f p_new = map1(u_new);

			const float f_old = f(p_old);
			const float f_new = f(p_new);

			const float T_old = inv_pdf1_stochastic(u_new) /** pdf1(p_old)*/;
			const float T_new = inv_pdf1_stochastic(u_old) /** pdf1(p_new)*/;

			const float a = fminf(1.0f, (f_new * T_old) / (f_old * T_new));
			if (random.next() < a)
				p = p_new;
		}
		else
		{
			const cugar::Vector2f p_old = p;
			const cugar::Vector2f u_old = inv_map2(p_old);
			const cugar::Vector2f u_new = perturb(u_old, random);
			const cugar::Vector2f p_new = map2(u_new);

			const float f_old = f(p_old);
			const float f_new = f(p_new);

			const float T_old = inv_pdf2(u_new) /** pdf2(p_old)*/;
			const float T_new = inv_pdf2(u_old) /** pdf2(p_new)*/;

			const float a = fminf(1.0f, (f_new * T_old) / (f_old * T_new));
			if (random.next() < a)
				p = p_new;
		}

		if (i > m)
			splat(res_x, res_y, image, p, 0.1f * float(res_x*res_y) / float(n - m));
	}
#elif 0
	cugar::Vector2f u = u1;
	uint32			t = 0;

	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmetropolis test %.1f%%      ", 100.0f * float(i) / float(n));

		if ((i & 1) == 0)
		{
			cugar::Vector2f u_tmp = perturb(u, random);

			if (t == 0)
			{
				const cugar::Vector2f p_old = map1(u);
				const cugar::Vector2f p_new = map1(u_tmp);

				const float f_old = f(p_old) / (pdf1(p_old) + pdf2(p_old));
				const float f_new = f(p_new) > 0.0f ? f(p_new) / (pdf1(p_new) + pdf2(p_new)) : 0.0f;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
					u = u_tmp;
			}
			else
			{
				const cugar::Vector2f p_old = map2(u);
				const cugar::Vector2f p_new = map2(u_tmp);

				const float f_old = f(p_old) / (pdf1(p_old) + pdf2(p_old));
				const float f_new = f(p_new) > 0.0f ? f(p_new) / (pdf1(p_new) + pdf2(p_new)) : 0.0f;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
					u = u_tmp;
			}
		}
		else if (t == 0)
		{
			const cugar::Vector2f p = map1(u);
			const cugar::Vector2f u_old = u;
			const cugar::Vector2f u_new = inv_map2(p);

			float jacobian_new = inv_pdf2(u_new);
			//float jacobian_old = inv_pdf1(u_old);
			float jacobian_old = inv_pdf1_stochastic(u_old);

			const float a = fminf(1.0f, jacobian_old / jacobian_new);
			//const float a = fminf(1.0f,  jacobian_new * pdf2(p)/pdf1(p));
			if (random.next() < a)
			{
				u = u_new;
				t = 1;
			}
		}
		else
		{
			const cugar::Vector2f p = map2(u);
			const cugar::Vector2f u_old = u;
			//const cugar::Vector2f u_new = inv_map1(p);
			const cugar::Vector2f u_new = inv_map1_stochastic(p,random);

			//float jacobian_new = inv_pdf1(u_new);
			float jacobian_new = inv_pdf1_stochastic(u_new);
			float jacobian_old = inv_pdf2(u_old);

			const float a = fminf(1.0f, jacobian_old / jacobian_new);
			//const float a = fminf(1.0f, jacobian_new * pdf1(p) / pdf2(p));
			if (random.next() < a)
			{
				u = u_new;
				t = 0;
			}
		}

		if (i > m)
		{
			if (t == 0) splat(res_x, res_y, image, map1(u), 0.1f * float(res_x*res_y) / float(n - m));
			else        splat(res_x, res_y, image, map2(u), 0.1f * float(res_x*res_y) / float(n - m));
		}
	}
#else
	for (uint32 i = 0; i < n; ++i)
	{
		if ((i % 1000) == 0)
			printf("\rmetropolis test %.1f%%      ", 100.0f * float(i) / float(n));

		if ((i & 1) == 0)
			//if (1)
		{
			cugar::Vector2f u1_tmp = perturb(u1, random);
			{
				const cugar::Vector2f p_old = map1(u1);
				const cugar::Vector2f p_new = map1(u1_tmp);

				const float f_old = f(p_old) / (pdf1(p_old) + pdf2(p_old));
				const float f_new = f(p_new) > 0.0f ? f(p_new) / (pdf1(p_new) + pdf2(p_new)) : 0.0f;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
					u1 = u1_tmp;
			}

			cugar::Vector2f u2_tmp = perturb(u2, random);
			{
				const cugar::Vector2f p_old = map2(u2);
				const cugar::Vector2f p_new = map2(u2_tmp);

				const float f_old = f(p_old) / (pdf1(p_old) + pdf2(p_old));
				const float f_new = f(p_new) > 0.0f ? f(p_new) / (pdf1(p_new) + pdf2(p_new)) : 0.0f;

				const float a = fminf(1.0f, f_new / f_old);
				if (random.next() < a)
					u2 = u2_tmp;
			}
		}
#if 0
		else
		{
			//u1 = inv_map1(map1(u1), random);
			const cugar::Vector2f u1_old = u1;
			const cugar::Vector2f u1_new = inv_map1(map1(u1));

			float jacobian_new = inv_pdf1(u1_new);
			float jacobian_old = inv_pdf1(u1_old);

			const float a = fminf(1.0f, jacobian_old / jacobian_new);
			if (random.next() < a)
				u1 = u1_new;
			//else
			//	fprintf(stderr, "a: %f (= %f / %f)\n", a, jacobian_old, jacobian_new);
		}
#else
		else
		{
			const cugar::Vector2f p1 = map1(u1);
			const cugar::Vector2f p2 = map2(u2);

			// try a swap
			const cugar::Vector2f u2_1 = inv_map2(p1);
			//const cugar::Vector2f u1_2 = inv_map1(p2);
			const cugar::Vector2f u1_2 = inv_map1_stochastic(p2,random);
			const float j2_1 = inv_pdf2(u2_1);
			//const float j1_2 = inv_pdf1(u1_2);
			const float j1_2 = inv_pdf1_stochastic(u1_2);

			//const float j1_1 = inv_pdf1(u1);
			const float j1_1 = inv_pdf1_stochastic(u1);
			const float j2_2 = inv_pdf2(u2);

			const float a = fminf(1.0f, (j1_1 * j2_2) / (j2_1 * j1_2));
			if (random.next() < a)
			{
				u1 = u1_2;
				u2 = u2_1;
			}
		}
#endif

		if (i > m)
		{
			splat(res_x, res_y, image, map1(u1), 0.05f * float(res_x*res_y) / float(n - m));
			splat(res_x, res_y, image, map2(u2), 0.05f * float(res_x*res_y) / float(n - m));
		}
	}
#endif
	printf("\n");
}

#endif

void metropolis_test()
{
	uint32 res = 400;

	std::vector<cugar::Vector3f> ref_image(res*res, 0.0f);
	if (0)
	{
		mc_test(res, res, &ref_image[0]);
		{
			FILE* file = fopen("test_ref.dat", "wb");
			fwrite(&ref_image[0], sizeof(cugar::Vector3f), res*res, file);
			fclose(file);
		}

		save_tga("test_ref.tga", res, res, &ref_image[0]);
	}
	else
	{
		FILE* file = fopen("test_ref.dat", "rb");
		fread(&ref_image[0], sizeof(cugar::Vector3f), res*res, file);
		fclose(file);
	}

	std::vector<cugar::Vector3f> image(res*res, 0.0f);
	metropolis_test(res, res, &image[0], &ref_image[0]);

	exit(0);
}
