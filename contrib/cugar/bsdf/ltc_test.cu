#include <cugar/linalg/matrix.h>

#include <cugar/bsdf/ggx.h>
#include <cugar/bsdf/ltc.h>
#include <cugar/sampling/random.h>

namespace ltc_ggx
{
	typedef float mat33[9];

	#include <cugar/bsdf/ltc_ggx.inc>
};

bool ltc_test()
{
	//unsigned ltc_format;
	//unsigned ltc_w;
	//unsigned ltc_h;
	//if (LoadDDSHeader("ltc_mat_ggx.dds", ltc_format, ltc_w, ltc_h) == false)
	//	return 1;
	//
	//std::vector<float> ltc_mat_data(ltc_w*ltc_h*4);
	//std::vector<float> ltc_amp_data(ltc_w*ltc_h*2);

	cugar::Vector3f V = cugar::normalize(cugar::Vector3f(0.3f, 0.25f, 0.5f));
	//cugar::Vector3f V = cugar::normalize(cugar::Vector3f(1.0, 0, 0.1));
	cugar::Vector3f N(0, 0, 1);
	cugar::Vector3f B = cugar::normalize(cugar::cross(N, V));
	cugar::Vector3f T = cugar::cross(B, N);

	cugar::DifferentialGeometry geom;
	geom.normal_s = geom.normal_g = N;
	geom.tangent  = cugar::orthogonal(N);
	geom.binormal = cugar::cross(N, geom.tangent);

	float4 ltc_tab[ltc_ggx::size*ltc_ggx::size];
	float4 ltc_tab_inv[ltc_ggx::size*ltc_ggx::size];

	cugar::LTCBsdf::preprocess(ltc_ggx::size, (const cugar::Matrix3x3f*)ltc_ggx::tabM, ltc_tab, ltc_tab_inv);

	cugar::LTCBsdf ltc(
		0.1f,
		ltc_tab,
		ltc_tab_inv,
		ltc_ggx::tabAmplitude,
		ltc_ggx::size);

	cugar::Vector3f L = cugar::reflect(-V, N);
	fprintf(stderr, "N.V : %f\n", dot(N, V));
	fprintf(stderr, "N.L : %f\n", dot(N, L));
	cugar::Vector3f fV = ltc.f(geom, V, V);
	fprintf(stderr, "f : %f\n", fV.x);
	cugar::Vector3f fL = ltc.f(geom, V, L);
	fprintf(stderr, "f : %f\n", fL.x);

	{
		cugar::Vector3f g;
		float			p;
		float			p_proj;
		ltc.sample(cugar::Vector3f(1, 0, 0), geom, V, L, g, p, p_proj);

		fprintf(stderr, "L  : %f, %f, %f\n", L.x, L.y, L.z);
		fprintf(stderr, "g  : %f\n", g.x);
		fprintf(stderr, "p  : %f\n", p);
		fprintf(stderr, "p' : %f\n", p_proj);
	}
	{
		cugar::Vector3f g;
		float			p;
		float			p_proj;
		ltc.sample(cugar::Vector3f(0, 0, 0), geom, V, L, g, p, p_proj);

		fprintf(stderr, "L  : %f, %f, %f\n", L.x, L.y, L.z);
		fprintf(stderr, "g  : %f\n", g.x);
		fprintf(stderr, "p  : %f\n", p);
		fprintf(stderr, "p' : %f\n", p_proj);
	}
	{
		cugar::Vector3f g;
		float			p;
		float			p_proj;
		ltc.sample(cugar::Vector3f(0, 0.5f, 0), geom, V, L, g, p, p_proj);

		fprintf(stderr, "L  : %f, %f, %f\n", L.x, L.y, L.z);
		fprintf(stderr, "g  : %f\n", g.x);
		fprintf(stderr, "p  : %f\n", p);
		fprintf(stderr, "p' : %f\n", p_proj);
	}
	{
		const float I = ltc.hemispherical_sector_integral(geom, V, make_float2(0.0f,0.5f*M_PIf), make_float2(0.0f, 2.0f*M_PIf));
		const float I1 = ltc.hemispherical_sector_integral(geom, V, make_float2(0.0f,0.5f*M_PIf), make_float2(0.0f,M_PIf));
		const float I2 = ltc.hemispherical_sector_integral(geom, V, make_float2(0.0f,0.5f*M_PIf), make_float2(M_PIf, 2.0f*M_PIf));

		fprintf(stderr, "full hemispherical integral: %f = %f + %f\n", I, I1, I2);
	}
	return true;
}