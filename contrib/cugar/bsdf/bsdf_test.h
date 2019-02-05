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
#include <cugar/sampling/random.h>
#include <cugar/bsdf/differential_geometry.h>
#include <cugar/bsdf/ggx_smith.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
*  \{
*/

inline bool bsdf_test(uint32 test_index, const GGXSmithBsdf& bsdf)
{
	const float max_inversion_error = 0.00001f;

	Random random;

	// perform an inversion test
	DifferentialGeometry geom;
	geom.normal_s = Vector3f(0,0,1);
	geom.normal_g = geom.normal_s;
	geom.tangent  = Vector3f(1,0,0);
	geom.binormal = Vector3f(0,1,0);
	for (uint32 i = 0; i < 1000; ++i)
	{
		Vector3f u( random.next(),  random.next(), random.next() );
		Vector3f V, L, L_inv;
		Vector3f g;
		float    p, p_proj;

		V = normalize( Vector3f(random.next(), random.next(), random.next()*2.0f - 1.0f) );

		bsdf.sample( u, geom, V, L, g, p, p_proj );

		// skip non-invertible cases (e.g. TIR)
		if (p == 0.0f)
			continue;

		Vector3f z;

		if (bsdf.invert( geom, V, L, random, z, p, p_proj ) == false)
		{
			fprintf(stderr, "failed inversion [%u:%u]!\n", test_index, i);
			fprintf(stderr, "  u : (%f,%f,%f)\n", u.x, u.y, u.z);
			fprintf(stderr, "  V : (%f,%f,%f)\n", V.x, V.y, V.z);
			fprintf(stderr, "  L : (%f,%f,%f)\n", L.x, L.y, L.z);
			return false;
		}
		bsdf.sample( z, geom, V, L_inv, g, p, p_proj );

		const float err = 1.0f - dot(L, L_inv);
		if (err > max_inversion_error)
		{
			fprintf(stderr, "failed inversion [%u:%u]!\n", test_index, i);
			fprintf(stderr, "  u     : (%f,%f,%f)\n", u.x, u.y, u.z);
			fprintf(stderr, "  u_inv : (%f,%f,%f)\n", z.x, z.y, z.z);
			fprintf(stderr, "  V     : (%f,%f,%f)\n", V.x, V.y, V.z);
			fprintf(stderr, "  L     : (%f,%f,%f)\n", L.x, L.y, L.z);
			fprintf(stderr, "  L_inv : (%f,%f,%f)\n", L_inv.x, L_inv.y, L_inv.z);
			return false;
		}

		cugar::Vector3f test_f;
		float			test_p_proj;

		bsdf.f_and_p( geom, V, L, test_f, test_p_proj, kProjectedSolidAngle );

		// check p_proj
		if (fabsf( test_p_proj - p_proj ) / max( p_proj, 0.01f ) > 0.03f) // up to 3% error
		{
			fprintf(stderr, "mismatching pdf evaluation [%u:%u]!\n", test_index, i);
			fprintf(stderr, "  p     : %f != %f\n", p_proj, test_p_proj);
			return false;
		}

		// check g = f/p
		cugar::Vector3f test_g = test_f / test_p_proj;
		if (fabsf( test_g.x - g.x ) / max( g.x, 0.01f ) > 0.03f) // up to 3% error
		{
			fprintf(stderr, "mismatching f/p evaluation [%u:%u]!\n", test_index, i);
			fprintf(stderr, "  g     : %f != %f\n", g.x, test_g.x);
			return false;
		}

		// check f = g * p
		cugar::Vector3f f = g * p_proj;
		if (fabsf( test_f.x - f.x ) / max( f.x, 0.01f ) > 0.03f) // up to 3% error
		{
			fprintf(stderr, "mismatching f/p evaluation [%u:%u]!\n", test_index, i);
			fprintf(stderr, "  f     : %f != %f\n", f.x, test_f.x);
			return false;
		}
	}
	return true;
}

/// unit test
///
inline bool bsdf_test()
{
	fprintf(stderr, "bsdf test... started\n");
	if (!bsdf_test( 0, GGXSmithBsdf(0.001f, false, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 1, GGXSmithBsdf(0.01f, false, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 2, GGXSmithBsdf(0.10f, false, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 3, GGXSmithBsdf(0.50f, false, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 4, GGXSmithBsdf(0.001f, true, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 5, GGXSmithBsdf(0.01f, true, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 6, GGXSmithBsdf(0.10f, true, 1.5f, 1.0f) )) return false;
	if (!bsdf_test( 7, GGXSmithBsdf(0.50f, true, 1.5f, 1.0f) )) return false;
	fprintf(stderr, "bsdf test... done\n");
	return true;
}

/*! \}
 */

} // namespace cugar
