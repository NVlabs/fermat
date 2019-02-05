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

#include <cugar/sampling/em.h>
#include <cugar/basic/vector.h>
#include <cugar/image/tga.h>
#include <cugar/sampling/random.h>

namespace cugar {

	namespace {

		enum EmType
		{
			STANDARD_EM		= 0,
			BATCH_JE_EM		= 1,
			ONLINE_JE_EM	= 2,
			STEPWISE_EM		= 3,
		};

		void write_image(const uint2 res, const cugar::Vector3f* img, const char* filename)
		{
			cugar::host_vector<uchar3> rgb(res.x * res.y);

			for (uint32 p = 0; p < res.x * res.y; ++p)
			{
				rgb[p].x = (uint8)cugar::quantize(img[p].x, 255);
				rgb[p].y = (uint8)cugar::quantize(img[p].y, 255);
				rgb[p].z = (uint8)cugar::quantize(img[p].z, 255);
			}
			cugar::write_tga(filename, res.x, res.y, (unsigned char*)&rgb[0], cugar::TGAPixels::RGB);
		}

		// initialize a mixture with randomly placed gaussians
		template <uint32 NCOMP>
		void initialize(cugar::Random& rand, cugar::Mixture<cugar::Gaussian_distribution_2d, NCOMP>& mixture)
		{
			for (uint32 i = 0; i < NCOMP; ++i)
			{
				cugar::Matrix2x2f cov;
				cov[0] = cugar::Vector2f(0.1f, 0.0f);
				cov[1] = cugar::Vector2f(0.0f, 0.1f);
				cugar::Gaussian_distribution_2d dist(cugar::Vector2f(rand.next(), rand.next()), cov);
				mixture.set_component(i, dist, 1.0f / float(NCOMP));
			}
		}

		template <EmType EM_TYPE, uint32 EM_COMPONENTS>
		void fit(const uint32 M, const uint32 N, cugar::Vector2f* points, float* weights, cugar::Mixture<cugar::Gaussian_distribution_2d, EM_COMPONENTS>& em_mixture)
		{
			cugar::Matrix<float, EM_COMPONENTS, 8> u_stats(0.0f);

			for (uint32 i = 0; i < M; ++i)
			{
				if (EM_TYPE == BATCH_JE_EM)
				{
					const float eta = 1.1f;

					cugar::joint_entropy_EM(em_mixture, eta, N, points, weights);
				}
				else if (EM_TYPE == ONLINE_JE_EM)
				{
					const float eta = 1.1f;

					for (uint32 j = 0; j < N; ++j)
						cugar::joint_entropy_EM(em_mixture, (eta / float(N)) * powf(float(i + 1), -0.7f), points[j], weights[j]);
				}
				else if (EM_TYPE == STANDARD_EM)
				{
					const float eta = 0.05f;

					const uint32 MINIBATCH_SIZE = 20;

					for (uint32 b = 0; b < N; b += MINIBATCH_SIZE)
					{
					  #if 0
						cugar::Vector2f batch_points[MINIBATCH_SIZE];
						float           batch_weights[MINIBATCH_SIZE];

						// use stochastic mini-batches
						for (uint32 j = 0; j < MINIBATCH_SIZE; ++j)
						{
							const uint32 r = min(uint32(rand.next() * N), uint32(N - 1));
							batch_points[j] = points[r];
							batch_weights[j] = weights[r];
						}
						cugar::EM<MINIBATCH_SIZE>(em_mixture, eta, batch_points, batch_weights);
					  #else
						// use fixed mini-batches
						cugar::EM<MINIBATCH_SIZE>(em_mixture, eta, points + b, weights + b);
					  #endif
					}
				}
				else
				{
					const uint32 MINIBATCH_SIZE = 20;

					for (uint32 b = 0; b < N; b += MINIBATCH_SIZE)
					{
						for (uint32 j = 0; j < MINIBATCH_SIZE; ++j)
						{
							const float eta = 1.0f * powf(float(i*N + b + j + 1), -0.7f);

							// use stochastic mini-batches
							//const uint32 r = min(uint32(rand.next() * N), uint32(N - 1));
							const uint32 r = b + j;
							cugar::stepwise_E(em_mixture, eta, points[r], weights[r], u_stats);
						}
						cugar::stepwise_M(em_mixture, u_stats, N);
					}
				}

				fprintf(stderr, "step %u\n", i);
				for (uint32 j = 0; j < em_mixture.NUM_COMPONENTS; ++j)
				{
					cugar::Vector2f   m = em_mixture.component(j).mean();
					cugar::Matrix2x2f p = em_mixture.component(j).precision();
					cugar::Matrix2x2f c = em_mixture.component(j).covariance();

					fprintf(stderr, "  w%u = %f\n", j, em_mixture.weight(j));
					fprintf(stderr, "  m%u = (%f, %f)\n", j, m.x, m.y);
					fprintf(stderr, "  p%u = [%f, %f : %f, %f]\n", j, p(0, 0), p(0, 1), p(1, 0), p(1, 1));
					fprintf(stderr, "  c%u = [%f, %f : %f, %f]\n", j, c(0, 0), c(0, 1), c(1, 0), c(1, 1));
				}
			}
		}

		template <uint32 NCOMP>
		void density_image(const uint32 res, cugar::host_vector<cugar::Vector3f>& img, const cugar::Mixture<cugar::Gaussian_distribution_2d, NCOMP>& mixture)
		{
			for (uint32 y = 0; y < res; ++y)
			{
				for (uint32 x = 0; x < res; ++x)
				{
					const cugar::Vector2f p(
						2.0f * float(x) / res - 1.0f,
						2.0f * float(y) / res - 1.0f);

					const float f = mixture.density(p);

					img[x + y*res] = cugar::Vector3f(f);
				}
			}
		}

		void points_image(
			const uint32			res,
			cugar::host_vector<cugar::Vector3f>& img,
			const uint32			N,
			const cugar::Vector2f*	p,
			const float*			w)
		{
			for (uint32 i = 0; i < res*res; ++i)
				img[i] = cugar::Vector3f(0.0f);

			for (uint32 i = 0; i < N; ++i)
			{
				const uint32 x = cugar::quantize((p[i].x + 1.0f)*0.5f, res);
				const uint32 y = cugar::quantize((p[i].y + 1.0f)*0.5f, res);

				img[x + y*res] = cugar::Vector3f(w[i]);
			}
		}

	} // anonymous namespace

bool em_test()
{
	fprintf(stderr, "EM test... started\n");
	cugar::Matrix2x2f cov;
	cov[0] = cugar::Vector2f(0.03f, 0.0f);
	cov[1] = cugar::Vector2f(0.0f, 0.01f);
	cugar::Gaussian_distribution_2d dist1(cugar::Vector2f(-0.5f, 0.0f), cov);
	cugar::Gaussian_distribution_2d dist2(cugar::Vector2f(0.5f, 0.0f), cov * 0.1f);

	cugar::Mixture<cugar::Gaussian_distribution_2d, 2> src_mixture;
	src_mixture.set_component(0, dist1, 0.5f);
	src_mixture.set_component(1, dist2, 0.5f);

	const uint32 EM_COMPONENTS = 2;

	cugar::Mixture<cugar::Gaussian_distribution_2d, EM_COMPONENTS> em_mixture;

	cugar::Random rand;

	// initialize the set of random samples
	cugar::host_vector<cugar::Vector2f> points;
	cugar::host_vector<float>			weights;

	const uint32 N = 1000;
	const uint32 M = 10;
	const uint32 res = 300;

	// write out the target density image
	cugar::host_vector<cugar::Vector3f> src_img(res * res);
	{
		density_image(res, src_img, src_mixture);

		write_image(make_uint2(res, res), cugar::raw_pointer(src_img), "density.tga");
	}

	// test 1
	{
		for (uint32 i = 0; i < N; ++i)
		{
			if (i & 1)
				points.push_back(dist1.map(cugar::Vector2f(rand.next(), rand.next())));
			else
				points.push_back(dist2.map(cugar::Vector2f(rand.next(), rand.next())));

			weights.push_back(1.0f);
		}

		// initialize the mixture with randomly placed gaussians
		initialize(rand, em_mixture);

		// fit a gaussian mixture
		fit<STEPWISE_EM>(M, N, cugar::raw_pointer(points), cugar::raw_pointer(weights), em_mixture);

		cugar::host_vector<cugar::Vector3f> pnt_img(res * res);
		cugar::host_vector<cugar::Vector3f> em_img(res * res);

		points_image(res, pnt_img, N, cugar::raw_pointer(points), cugar::raw_pointer(weights));
		density_image(res, em_img, em_mixture);

		write_image(make_uint2(res, res), cugar::raw_pointer(pnt_img), "pnt1.tga");
		write_image(make_uint2(res, res), cugar::raw_pointer(em_img), "em1.tga");
	}

	// test 2
	{
		const uint32 N = 10000;

		points.resize(N);
		weights.resize(N);

		for (uint32 i = 0; i < N; ++i)
		{
			points[i]  = cugar::Vector2f(cugar::randfloat(i,0), cugar::randfloat(i,1))*2.0f - cugar::Vector2f(1.0f);
			weights[i] = src_mixture.density(points[i]);
		}

		// initialize the mixture with randomly placed gaussians
		initialize(rand, em_mixture);

		// fit a gaussian mixture
		fit<STEPWISE_EM>(M, N, cugar::raw_pointer(points), cugar::raw_pointer(weights), em_mixture);

		cugar::host_vector<cugar::Vector3f> pnt_img(res * res);
		cugar::host_vector<cugar::Vector3f> em_img(res * res);

		points_image(res, pnt_img, N, cugar::raw_pointer(points), cugar::raw_pointer(weights));
		density_image(res, em_img, em_mixture);

		write_image(make_uint2(res, res), cugar::raw_pointer(pnt_img), "pnt2.tga");
		write_image(make_uint2(res, res), cugar::raw_pointer(em_img), "em2.tga");
	}

	fprintf(stderr, "EM test... done\n");
	return true;
}

} // namespace cugar

