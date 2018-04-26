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

#include <glut_viewer.h>

#include <cugar/linalg/matrix.h>
#include <cugar/sampling/random.h>
#include <cugar/image/tga.h>
#include <bsdf.h>

void metropolis_test();
bool ltc_test();

bool load_image(const char* filename, std::vector<cugar::Vector3f>& img, uint2& res)
{
	cugar::TGAHeader tga_header;
	unsigned char* rgb = cugar::load_tga(filename, &tga_header);

	if (rgb)
	{
		res.x = tga_header.width;
		res.y = tga_header.height;

		img.resize(tga_header.width * tga_header.height);

		for (uint32 p = 0; p < uint32(tga_header.width) * uint32(tga_header.height); ++p)
			img[p] = cugar::Vector3f(
				float(rgb[3 * p + 0]) / 255.0f,
				float(rgb[3 * p + 1]) / 255.0f,
				float(rgb[3 * p + 2]) / 255.0f );

		delete[] rgb;
		return true;
	}
	return false;
}

float diff_image(const uint2 res, const cugar::Vector3f* ref, cugar::Vector3f* dst, uchar3* rgb)
{
	for (uint32 p = 0; p < res.x * res.y; ++p)
		dst[p] -= ref[p];

	float e_sum = 0.0f;
	for (uint32 p = 0; p < res.x * res.y; ++p)
	{
		const float e2 = fabsf(dot(dst[p], dst[p]));

		e_sum += e2 / float(res.x * res.y);
	}

	e_sum = sqrtf(e_sum);

	// apply some false coloring
	for (uint32 p = 0; p < res.x * res.y; ++p)
	{
		const float max_e = cugar::max3(fabsf(dst[p].x), fabsf(dst[p].y), fabsf(dst[p].z));

		const float t_high = 0.5f;
		const float r_high = 1.0f - t_high;
		cugar::Vector3f col1 = cugar::Vector3f(0.2f, 0.3f, 0.9f);
		cugar::Vector3f col2 = cugar::Vector3f(1.0f, 0.9f, 0.2f);
		cugar::Vector3f col3 = cugar::Vector3f(1.0f, 0.0f, 0.0f);

		dst[p] = (max_e < t_high) ?
			col1 * (1.0f - (max_e - 0.0f) / t_high) + col2 * ((max_e - 0.0f) / t_high) :
			col2 * (1.0f - sqrtf(max_e - t_high) / r_high) + col3 * (sqrtf(max_e - t_high) / r_high);

		rgb[p].x = cugar::quantize(dst[p].x, 255);
		rgb[p].y = cugar::quantize(dst[p].y, 255);
		rgb[p].z = cugar::quantize(dst[p].z, 255);
	}
	return e_sum;
}

int main(int argc, char** argv)
{
#if 0
	const uint32 S = 32;
	std::vector<float> tables(S*S*S*S);
	precompute_glossy_reflectance(S, &tables[0]);
	FILE* file = fopen("fresnel.dat", "wb");
	fwrite(&tables[0], sizeof(float), S*S*S*S, file);
	fclose(file);
#endif

#if 0
	metropolis_test();
	exit(0);
#endif

#if 0
	ltc_test();
	exit(0);
#endif

	if (strcmp(argv[1], "-diff") == 0)
	{
		std::vector<cugar::Vector3f> img1;
		std::vector<cugar::Vector3f> img2;
		uint2 res1;
		uint2 res2;

		load_image(argv[2], img1, res1);
		load_image(argv[3], img2, res2);

		if (res1.x == res2.x && res1.y == res2.y)
		{
			std::vector<uchar3> rgb(res1.x * res1.y);

			const float rmse = diff_image(res1, &img1[0], &img2[0], &rgb[0]);

			fprintf(stderr, "RMSE: %f\n", rmse);

			cugar::write_tga("diff.tga", res1.x, res1.y, (unsigned char*)&rgb[0], cugar::TGAPixels::RGB);
		}
		else
			fprintf(stderr, "error: differing image resolutions!\n");

		exit(1);
	}

	bool start_viewer = false;
	const char* output_name = "output";

	std::vector<cugar::Vector3f>	ref_img;
	uint2							ref_res = make_uint2(0,0);
	uint32							n_passes = 1024;

	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-view") == 0)
			start_viewer = true;
		else if (strcmp(argv[i], "-ref") == 0)
		{
			fprintf(stderr, "loading reference image... started (%s)\n", argv[i+1]);
			load_image(argv[++i], ref_img, ref_res);
			fprintf(stderr, "loading reference image... done (%u, %u)\n", ref_res.x, ref_res.y);
		}
		else if (strcmp(argv[i], "-o") == 0)
			output_name = argv[++i];
		else if (strcmp(argv[i], "-passes") == 0)
			n_passes = atoi(argv[++i]);
	}

	if (start_viewer == false)
	{
		Renderer renderer;
		renderer.init(argc, argv);
		renderer.clear();

		for (uint32 i = 0; i <= n_passes; ++i)
		{
			renderer.render(i);

			if (i+1 >= 1 && cugar::is_pow2(i+1))
			{
				DomainBuffer<HOST_BUFFER, uint8> h_rgba = renderer.m_rgba;
				const uint8* rgba = h_rgba.ptr();

				// dump the image to a tga
				char filename[1024];
				sprintf(filename, "%s-%u.tga", output_name, i+1);
				fprintf(stderr, "\nsaving %s\n", filename);

				cugar::write_tga(filename, renderer.m_res_x, renderer.m_res_y, rgba, cugar::TGAPixels::RGBA);

				if (ref_res.x == renderer.m_res_x &&
					ref_res.y == renderer.m_res_y)
				{
					std::vector<cugar::Vector3f> img(ref_res.x * ref_res.y);
					std::vector<uchar3>			 rgb(ref_res.x * ref_res.y);

					// dequantize the rendered image
					for (uint32 p = 0; p < ref_res.x * ref_res.y; ++p)
						img[p] = cugar::Vector3f(rgba[p * 4 + 0], rgba[p * 4 + 1], rgba[p * 4 + 2]) / 255.0f;

					// calculate the diff
					const float rmse = diff_image(ref_res, &ref_img[0], &img[0], &rgb[0]);

					fprintf(stderr, "RMSE: %f\n", rmse);

					// and save the output
					sprintf(filename, "%s-%u-diff.tga", output_name, i+1);
					cugar::write_tga(filename, ref_res.x, ref_res.y, (unsigned char*)&rgb[0], cugar::TGAPixels::RGB);
				}
			}
		}
	}
	else
	{
		GlutViewer renderer;
		s_renderer = &renderer;
		s_renderer->init(argc, argv);
	}
	return 0;
}