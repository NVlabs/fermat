/*
 * cugar
 * Copyright (c) 2011-2014, NVIDIA CORPORATION. All rights reserved.
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

namespace cugar {

template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void joint_entropy_EM(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f										x,
	const float											w)
{
	// [jpantaleoni]
    // This algorithm is an implementation of the joint entropy expectation maximization described in:
    //
    // Batch and On-line Parameter Estimation of Gaussian Mixtures Based on the Joint Entropy,
    // Yoram Singer, Manfred K. Warmuth
    //
    // extended to support importance sampling weights w.
    //
    // In practice, this solution is obtained reformulating the definition of the log-likelihood
    // (or "loss") term:
    //
    //   E^X[ ln(P(X|\Theta) ] ~ 1/|S| \sum_{x \in S} ln(P(S|\Theta)
    //
    // as an expectation over a variable X with distribution f:
    //
    //   E_f^X[ ln(P(X|\Theta) ] = \int ln(P(x|\Theta) f(x) dx
    //
    // and assuming that instead of having X ~ f, we have a variable X ~ g, and use the importance sampling formula:
    //
    //   E_f^X[ ln(P(X|\Theta) ] = E_g^X[ f(X)/g(X) ln(P(X|\Theta) ] = \int w(x) ln(P(x|\Theta) g(x) dx
    //
    // with the "weight" w defined as w(x) = f(x)/g(x).
    // If we plug in the new definition of the loss function in eq. (8), (9) and (10) from the above paper, we'll
    // see that the results stay almost unmodified, except for a multiplication of each summand in the right hand side
    // by a factor w(x).
    // If we further proceed to getting equations (11), (12) and (13), we obtain the same equations by redefining
    // \beta_i(x) as \beta_i(x) = w(x) P(x|\Theta_i) / P(x|\Theta).
    //
    // Notice that, in order to avoid confusion, we renamed the mixture weights as \alpha_i.
    //
    const float p_sum = mixture.density(x);

    float alpha_sum = 0.0f;
    for (uint32 i = 0; i < NCOMPONENTS; ++i)
    {
        // note: we incorporate the importance sampling weight w inside beta_i
        const float beta_i = w * mixture.density(i,x) / p_sum;

        alpha_sum += mixture.weight(i) * expf( -eta * beta_i );
    }

    for (uint32 i = 0; i < NCOMPONENTS; ++i)
    {
        // note: we incorporate the importance sampling weight w inside beta_i
        const float beta_i = w * mixture.density(i,x) / p_sum;

        const float alpha_i = mixture.weight(i) * expf( -eta * beta_i ) / alpha_sum;

        Vector2f mu_i = mixture.component(i).mean();
                 mu_i = mu_i + eta * beta_i * (x - mu_i); // TODO: what if eta * beta_i gets higher than 1? On average things should work out, but
                                                          // in practice high weights might cause very large oscillations - should we bound eta * beta_i to 1?

        const Vector2f x_mu = (x - mu_i); // NOTE: use the new mu_i

        Matrix2x2f T;
        T(0, 0)           = x_mu[0] * x_mu[0];
        T(1, 0) = T(0, 1) = x_mu[0] * x_mu[1];
        T(1, 1)           = x_mu[1] * x_mu[1];

        const Matrix2x2f P   = mixture.component(i).precision();
        const Matrix2x2f P_i = P + eta * beta_i * (P - P * T * P);

        mixture.set_weight( i, alpha_i );
        mixture.set_component( i, Gaussian_distribution_2d( mu_i, P_i, Gaussian_distribution_2d::PRECISION_MATRIX ) );
    }
}

template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void joint_entropy_EM(
          Mixture<Gaussian_distribution_2d,NCOMPONENTS>&	mixture,
    const float												eta,
    const uint32											N,
    const Vector2f*											x,
    const float*											w)
{
    // [jpantaleoni]
    // This algorithm is an implementation of the joint entropy expectation maximization described in:
    //
    // Batch and On-line Parameter Estimation of Gaussian Mixtures Based on the Joint Entropy,
    // Yoram Singer, Manfred K. Warmuth
    //
    // extended to support importance sampling weights w.
    //
    // In practice, this solution is obtained reformulating the definition of the log-likelihood
    // (or "loss") term:
    //
    //   E^X[ ln(P(X|\Theta) ] ~ 1/|S| \sum_{x \in S} ln(P(S|\Theta)
    //
    // as an expectation over a variable X with distribution f:
    //
    //   E_f^X[ ln(P(X|\Theta) ] = \int ln(P(x|\Theta) f(x) dx
    //
    // and assuming that instead of having X ~ f, we have a variable X ~ g, and use the importance sampling formula:
    //
    //   E_f^X[ ln(P(X|\Theta) ] = E_g^X[ f(X)/g(X) ln(P(X|\Theta) ] = \int w(x) ln(P(x|\Theta) g(x) dx
    //
    // with the "weight" w defined as w(x) = f(x)/g(x).
    // If we plug in the new definition of the loss function in eq. (8), (9) and (10) from the above paper, we'll
    // see that the results stay almost unmodified, except for a multiplication of each summand in the right hand side
    // by a factor w(x).
    // If we further proceed to getting equations (11), (12) and (13), we obtain the same equations by redefining
    // \beta_i(x) as \beta_i(x) = w(x) P(x|\Theta_i) / P(x|\Theta).
    //
    // Notice that, in order to avoid confusion, we renamed the mixture weights as \alpha_i.
    //

	const float eta_over_N = eta / float(N);

    float W = 0.0f;
    for (uint32 s = 0; s < N; ++s)
        W += w[s];

    const float inv_W = 1.0f / W;

	// Update all alpha's
	float alpha_new[NCOMPONENTS];
	
	float alpha_sum = 0.0f;
    for (uint32 i = 0; i < NCOMPONENTS; ++i)
    {
        float alpha_i = 0.0f;

        for (uint32 s = 0; s < N; ++s)
        {
            const float p_sum = mixture.density(x[s]); // TODO: we're recomputing this NCOMPONENTS times

            // note: we incorporate the importance sampling weight w inside beta_i
            const float beta_i = (w[s] * inv_W) * mixture.density(i,x[s]) / p_sum;

            alpha_i += beta_i;
        }

		alpha_new[i] = mixture.weight(i) * expf(-eta_over_N * alpha_i);

		alpha_sum += alpha_new[i];
    }

	for (uint32 i = 0; i < NCOMPONENTS; ++i)
	{
		// Finish computing alpha_i
		const float alpha_i = alpha_new[i] / alpha_sum;

		alpha_new[i] = alpha_i;

		//mixture.set_weight(i, alpha_i);
	}

	// Update all mu's
	cugar::Vector2f mu_new[NCOMPONENTS];

	for (uint32 i = 0; i < NCOMPONENTS; ++i)
	{
		const Vector2f old_mu_i = mixture.component(i).mean();
		Vector2f mu_i = Vector2f(0, 0);

		// Update alpha_i and mu_i
		for (uint32 s = 0; s < N; ++s)
		{
			const float p_sum = mixture.density(x[s]); // TODO: we're recomputing this NCOMPONENTS times

													   // note: we incorporate the importance sampling weight w inside beta_i
			const float beta_i = (w[s] * inv_W) * mixture.density(i, x[s]) / p_sum;

			mu_i += beta_i * (x[s] - old_mu_i); // TODO: what if eta * beta_i gets higher than 1? On average things should work out, but
												// in practice high weights might cause very large oscillations - should we bound eta * beta_i to 1?
		}

		// Finish computing beta_i
		mu_i = old_mu_i + eta_over_N * mu_i;

		mu_new[i] = mu_i;
	}
	//for (uint32 i = 0; i < NCOMPONENTS; ++i)
	//	mixture.set_component(i, Gaussian_distribution_2d(mu_new[i], mixture.component(i).precision(), Gaussian_distribution_2d::PRECISION_MATRIX));

	// update all P's
	cugar::Matrix2x2f P_new[NCOMPONENTS];

	for (uint32 i = 0; i < NCOMPONENTS; ++i)
	{
		//const Vector2f mu_i = mixture.component(i).mean();
		const Vector2f mu_i = mu_new[i];

		const Matrix2x2f P   = mixture.component(i).precision();
              Matrix2x2f P_i = Matrix2x2f(0.0f);

        // Update P_i
        for (uint32 s = 0; s < N; ++s)
        {
            const float p_sum = mixture.density(x[s]); // TODO: we're recomputing this NCOMPONENTS times

            // note: we incorporate the importance sampling weight w inside beta_i
            const float beta_i = (w[s] * inv_W) * mixture.density(i, x[s]) / p_sum;

            const Vector2f x_mu = (x[s] - mu_i); // NOTE: use the new mu_i

            Matrix2x2f T;
            T(0, 0)           = x_mu[0] * x_mu[0];
            T(1, 0) = T(0, 1) = x_mu[0] * x_mu[1];
            T(1, 1)           = x_mu[1] * x_mu[1];

            P_i += beta_i * (P - P * T * P);
        }

        // Finish computing P_i
        P_i = P + eta_over_N * P_i;

		P_new[i] = P_i;
    }
	//for (uint32 i = 0; i < NCOMPONENTS; ++i)
	//	mixture.set_component(i, Gaussian_distribution_2d(mixture.component(i).mean(), P_new[i], Gaussian_distribution_2d::PRECISION_MATRIX));
	for (uint32 i = 0; i < NCOMPONENTS; ++i)
		mixture.set_component(i, Gaussian_distribution_2d(mu_new[i], P_new[i], Gaussian_distribution_2d::PRECISION_MATRIX), alpha_new[i]);
}

// Online step-wise Expectation Maximization
//
template <uint32 N, uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void EM(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f*										x,
	const float*										w)
{
	float prob_comp[N][NCOMPONENTS];

  #if 0
	float W = 0.0f;
	for (uint32 s = 0; s < N; ++s)
		W += w[s];

	const float inv_W = 1.0f / W;
  #endif

	// compute the probability of each point being in each class
	for (uint32 s = 0; s < N; ++s)
	{
		float prob_sum = 0.0f;
		for (uint32 c = 0; c < NCOMPONENTS; ++c)
			prob_sum += prob_comp[s][c] = w[s] * mixture.density(c, x[s]);

		for (uint32 c = 0; c < NCOMPONENTS; ++c)
			prob_comp[s][c] /= prob_sum;
	}

	float alpha_sum = 0.0f;
	// compute the new weights
	for (uint32 c = 0; c < NCOMPONENTS; ++c)
	{
		for (uint32 s = 0; s < N; ++s)
			alpha_sum += prob_comp[s][c];
	}

	// compute the new weights
	for (uint32 c = 0; c < NCOMPONENTS; ++c)
	{
		float alpha_c = 0.0f;
		for (uint32 s = 0; s < N; ++s)
			alpha_c += prob_comp[s][c];

		alpha_c = alpha_c / alpha_sum;

		// interpolate the weights
		alpha_c = mixture.weight(c) * (1.0f - eta) + alpha_c * eta;

		mixture.set_weight(c, alpha_c);
	}
	
	// compute the new mean and covariance statistics for each class
	for (uint32 c = 0; c < NCOMPONENTS; ++c)
	{
		cugar::Vector2f mu_c(0.0f);
		float           q(0.0f);
		for (uint32 s = 0; s < N; ++s)
		{
			mu_c += prob_comp[s][c] * x[s];
			q    += prob_comp[s][c];
		}
		mu_c /= q;

		// interpolate the mean
		mu_c = mixture.component(c).mean() * (1.0f - eta) + mu_c * eta;

		cugar::Matrix2x2f cov_c(0.0f);

		for (uint32 s = 0; s < N; ++s)
		{
			const Vector2f x_mu = (x[s] - mu_c); // NOTE: use the new mu_c

			Matrix2x2f T;
			T(0, 0) = x_mu[0] * x_mu[0];
			T(1, 0) = T(0, 1) = x_mu[0] * x_mu[1];
			T(1, 1) = x_mu[1] * x_mu[1];

			cov_c += prob_comp[s][c] * T;
		}
		cov_c /= q;

		// interpolate the covariance matrix
		cov_c = mixture.component(c).covariance() * (1.0f - eta) + cov_c * eta;

		mixture.set_component(c, Gaussian_distribution_2d( mu_c, cov_c, Gaussian_distribution_2d::COVARIANCE_MATRIX));
	}
}

// Online step-wise Expectation Maximization
//
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void stepwise_E(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const float											eta,
	const Vector2f										x,
	const float											w,
	Matrix<float, NCOMPONENTS, 8>&						u)					// sufficient statistics
{
	// compute the responsibilities for each lobe
	float gamma[NCOMPONENTS];

	float gamma_sum = 0.0f;
	for (uint32 c = 0; c < NCOMPONENTS; ++c)
		gamma_sum += gamma[c] = mixture.density(c, x);

	for (uint32 c = 0; c < NCOMPONENTS; ++c)
		gamma[c] /= gamma_sum;

	// compute the sufficient statistics
	for (uint32 c = 0; c < NCOMPONENTS; ++c)
	{
		u[c][0] = (1.0f - eta) * u[c][0] + eta * w * gamma[c] * 1.0f;
		u[c][1] = (1.0f - eta) * u[c][1] + eta * w * gamma[c] * x.x;
		u[c][2] = (1.0f - eta) * u[c][2] + eta * w * gamma[c] * x.y;
		u[c][3] = (1.0f - eta) * u[c][3] + eta * w * gamma[c] * (x.x * x.x);
		u[c][4] = (1.0f - eta) * u[c][4] + eta * w * gamma[c] * (x.y * x.y);
		u[c][5] = (1.0f - eta) * u[c][5] + eta * w * gamma[c] * (x.x * x.y);
		u[c][6] = (1.0f - eta) * u[c][6] + eta * w;
		u[c][7] = u[c][7] + 1; // point counter
	}
}

// Online step-wise Expectation Maximization
//
template <uint32 NCOMPONENTS>
CUGAR_HOST_DEVICE
void stepwise_M(
	Mixture<Gaussian_distribution_2d, NCOMPONENTS>&		mixture,
	const Matrix<float, NCOMPONENTS, 8>&						u,			// sufficient statistics
	const uint32										N)					// total number of points
{
	const float PRIOR_A = 1.0e-2f;
	const float PRIOR_B = 5.0e-4f;
	const float PRIOR_V = 1.0e-2f;

	for (uint32 c = 0; c < NCOMPONENTS; ++c)
	{
		const float u_gamma = u[c][0];
		const float w_avg   = u[c][6];
		const float n       = min( u[c][7], float(N) );

		if (u_gamma > 1.0e-8f)
		{
			Matrix2x2f cov_c;
			cov_c[0][0] = PRIOR_B;
			cov_c[0][1] = 0.0f;
			cov_c[1][0] = 0.0f;
			cov_c[1][1] = PRIOR_B;

			const Vector2f s(u[c][1], u[c][2]);
			//const Vector2f mu = mixture.component(c).mean();

			// update the mean
			const Vector2f mu_c = s / u_gamma;

			// update the covariance matrix
			const float A_xx = s.x * mu_c.x + mu_c.x * s.x;		// NOTE: we are using the update values of mu_c
			const float A_xy = s.x * mu_c.y + mu_c.x * s.y;
			const float A_yy = s.y * mu_c.y + mu_c.y * s.y;

			const float B_xx = mu_c.x * mu_c.x;
			const float B_xy = mu_c.x * mu_c.y;
			const float B_yy = mu_c.y * mu_c.y;

			const float s_xx = u[c][3];
			const float s_yy = u[c][4];
			const float s_xy = u[c][5];

			cov_c[0][0] += n * ((s_xx - A_xx + u_gamma * B_xx) / w_avg);
			cov_c[0][1] += n * ((s_xy - A_xy + u_gamma * B_xy) / w_avg);
			cov_c[1][0] += n * ((s_xy - A_xy + u_gamma * B_xy) / w_avg);
			cov_c[1][1] += n * ((s_yy - A_yy + u_gamma * B_yy) / w_avg);

			cov_c /= PRIOR_A + n * (u_gamma / w_avg);

			// update the mixing coefficients
			const float alpha_c = (n * (u_gamma / w_avg) + PRIOR_V) / (n + NCOMPONENTS * PRIOR_V);

			mixture.set_component(
				c,
				Gaussian_distribution_2d(mu_c, cov_c, Gaussian_distribution_2d::COVARIANCE_MATRIX),
				alpha_c);
		}
	} 
}

} // namespace cugar
