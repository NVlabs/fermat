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

#include <renderer.h>
#include <pathtracer.h>
#include <files.h>
#include <bpt.h>
#include <cmlt.h>
#include <pssmlt.h>
#include <rpt.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_matrix.h>
#include <mesh/MeshStorage.h>
#include <eaw.h>
#include <cugar/basic/cuda/arch.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/functors.h>
#include <cugar/basic/cuda/sort.h>
#include <cugar/image/tga.h>
#include <cugar/bsdf/ltc.h>
#include <buffers.h>
#include <vector>

namespace ltc_ggx
{
	typedef float mat33[9];

#include <cugar/bsdf/ltc_ggx.inc>
};

void load_scene(const char* filename, MeshStorage& mesh, const std::vector<std::string>& dirs, std::vector<std::string>& scene_dirs);


//------------------------------------------------------------------------------
__global__ void fill_n_kernel(const int n, uint32_t* pixels)
{
	const int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < n) pixels[idx] = idx;
}

void fill_n(const int n, Buffer<uint32_t>& pixels)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(n, blockSize.x));
	fill_n_kernel << < gridSize, blockSize >> >(n, pixels.ptr());
}

//------------------------------------------------------------------------------
__global__ void to_rgba_kernel(const RendererView renderer, uint8* rgba)
{
	const uint32 idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < renderer.res_x * renderer.res_y)
	{
		if (renderer.shading_mode == kShaded)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::COMPOSITED_C, idx);
			//cugar::Vector4f c =
			//	renderer.fb(FBufferDesc::DIRECT_C, idx) +
			//	renderer.fb(FBufferDesc::DIFFUSE_C, idx) * renderer.fb(FBufferDesc::DIFFUSE_A, idx) +
			//	renderer.fb(FBufferDesc::SPECULAR_C, idx) * renderer.fb(FBufferDesc::SPECULAR_A, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kFiltered)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::FILTERED_C, idx);
			//cugar::Vector4f c =
			//	renderer.fb(FBufferDesc::DIRECT_C, idx) +
			//	renderer.fb(FBufferDesc::DIFFUSE_C, idx) * renderer.fb(FBufferDesc::DIFFUSE_A, idx) +
			//	renderer.fb(FBufferDesc::SPECULAR_C, idx) * renderer.fb(FBufferDesc::SPECULAR_A, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kAlbedo)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::DIFFUSE_A, idx) +
								renderer.fb(FBufferDesc::SPECULAR_A, idx);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kDiffuseAlbedo)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::DIFFUSE_A, idx);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kSpecularAlbedo)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::SPECULAR_A, idx);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kDiffuseColor)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::DIFFUSE_C, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kSpecularColor)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::SPECULAR_C, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kDirectLighting)
		{
			cugar::Vector4f c = renderer.fb(FBufferDesc::DIRECT_C, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kVariance)
		{
			float c = renderer.fb(FBufferDesc::COMPOSITED_C, idx).w;

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + 1);
			c = powf(c, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode >= kAux0 && (renderer.shading_mode - kAux0 < renderer.fb.n_channels - FBufferDesc::NUM_CHANNELS))
		{
			const uint32 aux_channel = renderer.shading_mode - kAux0 + FBufferDesc::NUM_CHANNELS;
			cugar::Vector4f c = renderer.fb(aux_channel, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / 2.2f);
			c.y = powf(c.y, 1.0f / 2.2f);
			c.z = powf(c.z, 1.0f / 2.2f);
			c.w = powf(c.w, 1.0f / 2.2f);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
	}
}

void to_rgba(const RendererView renderer, uint8* rgba)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(renderer.res_x * renderer.res_y, blockSize.x));
	to_rgba_kernel <<< gridSize, blockSize >>>(renderer, rgba);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("to_rgba"));
}
//------------------------------------------------------------------------------
__global__ void multiply_frame_kernel(RendererView renderer, const float scale)
{
	const uint32 idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < renderer.res_x * renderer.res_y)
	{
		// before scaling, save out luminance data
		renderer.fb(FBufferDesc::LUMINANCE, idx) = cugar::Vector4f(
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::DIRECT_C,		idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::DIFFUSE_C,		idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::SPECULAR_C,	idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::COMPOSITED_C,	idx)).xyz()) );

		renderer.fb(FBufferDesc::DIFFUSE_C,		idx)	*= scale;
		renderer.fb(FBufferDesc::DIFFUSE_A,		idx)	*= scale;
		renderer.fb(FBufferDesc::SPECULAR_C,	idx)	*= scale;
		renderer.fb(FBufferDesc::SPECULAR_A,	idx)	*= scale;
		renderer.fb(FBufferDesc::DIRECT_C,		idx)	*= scale;
		renderer.fb(FBufferDesc::COMPOSITED_C,	idx)	*= scale;
	}
}
//------------------------------------------------------------------------------
__global__ void update_variances_kernel(RendererView renderer, const uint32 n)
{
	const uint32 idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < renderer.res_x * renderer.res_y)
	{
		// fetch the previous frame's luminances
		const cugar::Vector4f old_lum = renderer.fb(FBufferDesc::LUMINANCE, idx);

		// compute the new frame's luminances
		const cugar::Vector4f new_lum = cugar::Vector4f(
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::DIRECT_C,		idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::DIFFUSE_C,		idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::SPECULAR_C,	idx)).xyz()),
			cugar::max_comp(cugar::Vector4f(renderer.fb(FBufferDesc::COMPOSITED_C,	idx)).xyz()) );

		// compute the change in variance (x(n) - avg(n-1))*(x(n) - avg(n)), which can be written as the sum of two terms:
		//  1. n*avg(n) - (n-1)*avg(n-1) - avg(n-1) = n*(avg(n) - avg(n-1))
		//	2. n*avg(n) - (n-1)*avg(n-1) - avg(n)   = (n-1)*(avg(n) - avg(n-1))
		const cugar::Vector4f delta_lum_1 = n * (new_lum - old_lum);
		const cugar::Vector4f delta_lum_2 = (n - 1) * (new_lum - old_lum);
		const cugar::Vector4f delta_var = (delta_lum_1 * delta_lum_2) / (n*n);

		// add the variance deltas to the old variances (previously rescaled by (n-1)/n) stored in the alpha components of the respective channels
		renderer.fb(FBufferDesc::DIRECT_C,		idx).w += delta_var.x;
		renderer.fb(FBufferDesc::DIFFUSE_C,		idx).w += delta_var.y;
		renderer.fb(FBufferDesc::SPECULAR_C,	idx).w += delta_var.z;
		renderer.fb(FBufferDesc::COMPOSITED_C,	idx).w += delta_var.w;
	}
}

//------------------------------------------------------------------------------

__global__ void filter_variance_kernel(const FBufferChannelView img, float* var, const uint32 FW)
{
	const uint32 x = threadIdx.x + blockIdx.x*blockDim.x;
	const uint32 y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < img.res_x &&
		y < img.res_y)
	{
		const int32 lx = x > FW				? x - FW : 0;
		const int32 rx = x + FW < img.res_x ? x + FW : img.res_x - 1;

		const int32 ly = y > FW				? y - FW : 0;
		const int32 ry = y + FW < img.res_y ? y + FW : img.res_y - 1;

		float variance = 0.0f;

		for (int yy = ly; yy <= ry; yy++)
			for (int xx = lx; xx <= rx; xx++)
				variance += img(xx, yy).w;

		variance /= (ry - ly + 1) * (rx - lx + 1);
		
		var[x + y * img.res_x] = variance;
	}
}

void filter_variance(const FBufferChannelView img, float* var, const uint32 FW = 1)
{
	dim3 blockSize(32, 4);
	dim3 gridSize(cugar::divide_ri(img.res_x, blockSize.x), cugar::divide_ri(img.res_y, blockSize.y));

	filter_variance_kernel << < gridSize, blockSize >> > (img, var, FW);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("filter_variance"));
}

//------------------------------------------------------------------------------

void Renderer::multiply_frame(const float scale)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(m_res_x * m_res_y, blockSize.x));
	multiply_frame_kernel <<< gridSize, blockSize >>>(view(0), scale);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("multiply_frame") );
}

//------------------------------------------------------------------------------

void Renderer::rescale_frame(const uint32 instance)
{
	multiply_frame( float(instance)/float(instance+1) );
}

//------------------------------------------------------------------------------

void Renderer::update_variances(const uint32 instance)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(m_res_x * m_res_y, blockSize.x));
	update_variances_kernel <<< gridSize, blockSize >>>(view(0), instance + 1);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("update_variances") );
}

//------------------------------------------------------------------------------

// Renderer initialization
//
void Renderer::init(int argc, char** argv)
{
	const char* filename = NULL;

	m_renderer_type = kBPT;
	m_exposure = 1.0f;
	m_res_x = 1600;
	m_res_y = 900;
	m_aspect = 0.0f;
	m_shading_rate = 1.0f;
	m_shading_mode = kShaded;

	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-i") == 0)
			filename = argv[++i];
		else if (strcmp(argv[i], "-pt") == 0)
			m_renderer_type = kPT;
		else if (strcmp(argv[i], "-bpt") == 0)
			m_renderer_type = kBPT;
		else if (strcmp(argv[i], "-cmlt") == 0)
			m_renderer_type = kCMLT;
		else if (strcmp(argv[i], "-pssmlt") == 0)
			m_renderer_type = kPSSMLT;
		else if (strcmp(argv[i], "-rpt") == 0)
			m_renderer_type = kRPT;
		else if (strcmp(argv[i], "-r") == 0 ||
				 strcmp(argv[i], "-res") == 0)
		{
			m_res_x = atoi(argv[++i]);
			m_res_y = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-a") == 0 ||
			strcmp(argv[i], "-aspect") == 0)
		{
			m_aspect = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-c") == 0)
		{
			FILE* camera_file = fopen(argv[++i], "r");
			if (camera_file == NULL)
			{
				fprintf(stderr, "failed opening camera file %s\n", argv[i]);
				exit(0);
			}
			fscanf(camera_file, "%f %f %f", &m_camera.eye.x, &m_camera.eye.y, &m_camera.eye.z);
			fscanf(camera_file, "%f %f %f", &m_camera.aim.x, &m_camera.aim.y, &m_camera.aim.z);
			fscanf(camera_file, "%f %f %f", &m_camera.up.x, &m_camera.up.y, &m_camera.up.z);
			fscanf(camera_file, "%f", &m_camera.fov);
			m_camera.dx = normalize(cross(m_camera.aim - m_camera.eye, m_camera.up));
			fclose(camera_file);
		}
	}

	if (m_aspect == 0.0f)
		m_aspect = float(m_res_x) / float(m_res_y);

	if (filename == NULL)
	{
		fprintf(stderr, "options:\n");
		fprintf(stderr, "  -i scene.obj         specify the input scene\n");
		fprintf(stderr, "  -r int int           specify the resolution\n");
		fprintf(stderr, "  -a float             specify the aspect ratio\n");
		fprintf(stderr, "  -c camera.txt		specify a camera file\n");
		fprintf(stderr, "  -pt                  use the PT renderer\n");
		fprintf(stderr, "  -bpt                 use the BPT renderer\n");
		fprintf(stderr, "  -mlt                 use the MLT renderer\n");
		fprintf(stderr, "  -cmlt                use the CMLT renderer\n");
		fprintf(stderr, "  -pssmlt              use the PSSMLT renderer\n");
		exit(0);
	}


	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-c") == 0)
		{
			FILE* camera_file = fopen(argv[++i], "r");
			fscanf(camera_file, "%f %f %f", &m_camera.eye.x, &m_camera.eye.y, &m_camera.eye.z);
			fscanf(camera_file, "%f %f %f", &m_camera.aim.x, &m_camera.aim.y, &m_camera.aim.z);
			fscanf(camera_file, "%f %f %f", &m_camera.up.x, &m_camera.up.y, &m_camera.up.z);
			fscanf(camera_file, "%f", &m_camera.fov);
			m_camera.dx = normalize(cross(m_camera.aim - m_camera.eye, m_camera.up));
			fclose(camera_file);
		}
	}

	m_rgba.alloc(m_res_x * m_res_y * 4);
	m_var.alloc(m_res_x * m_res_y);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	fprintf(stderr, "cuda device: %s\n", prop.name);
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	fprintf(stderr, "  memory: %.3f GB\n",
		float(total) / (1024 * 1024 * 1024));

	std::vector<unsigned int> devices(1);
	devices[0] = 0;

	cudaSetDevice( devices[0] );

	// create an Optix Prime context
	m_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);

	m_context->setCudaDeviceNumbers( devices );

	switch (m_renderer_type)
	{
	case kPT:		{ m_renderer = new PathTracer(); break; }
	case kBPT:		{ m_renderer = new BPT(); break; }
	case kCMLT:		{ m_renderer = new CMLT(); break; }
	case kPSSMLT:	{ m_renderer = new PSSMLT(); break; }
	case kRPT:		{ m_renderer = new RPT(); break; }
	default:		{ m_renderer = new PathTracer(); break; }
	};

	const uint32 aux_channels = m_renderer->auxiliary_channel_count();

	m_fb.set_channel_count(FBufferDesc::NUM_CHANNELS + aux_channels);
	m_fb.set_channel(FBufferDesc::DIFFUSE_C,	"diffuse_color");
	m_fb.set_channel(FBufferDesc::DIFFUSE_A,	"diffuse_albedo");
	m_fb.set_channel(FBufferDesc::SPECULAR_C,	"specular_color");
	m_fb.set_channel(FBufferDesc::SPECULAR_A,	"specular_albedo");
	m_fb.set_channel(FBufferDesc::DIRECT_C,		"direct_color");
	m_fb.set_channel(FBufferDesc::COMPOSITED_C, "composited_color");
	m_fb.set_channel(FBufferDesc::FILTERED_C,	"filtered_color");
	m_fb.set_channel(FBufferDesc::LUMINANCE,	"luminance");
	m_renderer->register_auxiliary_channels( m_fb, FBufferDesc::NUM_CHANNELS );
	m_fb.resize(m_res_x, m_res_y);

	m_fb_temp[0].resize(m_res_x, m_res_y);
	m_fb_temp[1].resize(m_res_x, m_res_y);

#if 0
	// pre-computer the samples buffer
	m_samples.alloc(m_res_x * m_res_y);
	{
		DomainBuffer<RTP_BUFFER_TYPE_HOST, float2> samples(m_res_x * m_res_y);

		cugar::MJSampler sampler;
		sampler.sample(m_res_x, m_res_y, (cugar::Vector2f*)samples.ptr());

		m_samples = samples;
	}
#endif

	// Load the LTC coefficients
	{
		fprintf(stderr, "initializing LTC coefficients... started\n");
		DomainBuffer<HOST_BUFFER, float4>		ltc_M;
		DomainBuffer<HOST_BUFFER, float4>		ltc_Minv;

		ltc_M.alloc(ltc_ggx::size * ltc_ggx::size);
		ltc_Minv.alloc(ltc_ggx::size * ltc_ggx::size);

		cugar::LTCBsdf::preprocess(ltc_ggx::size, (const cugar::Matrix3x3f*)ltc_ggx::tabM, ltc_M.ptr(), ltc_Minv.ptr());

		m_ltc_size = ltc_ggx::size;
		m_ltc_M    = ltc_M;
		m_ltc_Minv = ltc_Minv;
		m_ltc_A.alloc(ltc_ggx::size * ltc_ggx::size);
		m_ltc_A.copy_from(ltc_ggx::size * ltc_ggx::size, HOST_BUFFER, ltc_ggx::tabAmplitude);
		fprintf(stderr, "initializing LTC coefficients... done\n");
	}

	fprintf(stderr, "loading mesh file %s... started\n", filename);

	std::vector<std::string> scene_dirs;
	{
		scene_dirs.push_back(""); // always look in the current directory

		char local_path[2048];
		extract_path(filename, local_path);
		scene_dirs.push_back(local_path);
	}

	// Create the Model object
	//
	try
	{
		std::vector<std::string> dirs = scene_dirs;

		if (strlen(filename) > 3 && strcmp(filename+strlen(filename)-3, ".fa") == 0)
			load_scene(filename, m_mesh, dirs, scene_dirs);
		else
			loadModel(filename, m_mesh);

		// compute the bbox
		if (1)
		{
			cugar::Vector3f bmin(1.0e16f, 1.0e16f, 1.0e16f);
			cugar::Vector3f bmax(-1.0e16f, -1.0e16f, -1.0e16f);

			float3* v = reinterpret_cast<float3*>(m_mesh.getVertexData());
			for (int32_t i = 0; i < m_mesh.getNumVertices(); ++i)
			{
				bmin = cugar::min(bmin, cugar::Vector3f(v[i]));
				bmax = cugar::max(bmax, cugar::Vector3f(v[i]));
			}

			// scale the model
			if (0)
			{
				const cugar::Vector3f center = (bmin + bmax) * 0.5f;
				const float scale = 1.0f / cugar::max3(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
				for (int32_t i = 0; i < m_mesh.getNumVertices(); ++i)
					v[i] = (v[i] - center) * scale;
			}

			// print the bounding box
			fprintf(stderr, "  bbox[%f, %f, %f][%f, %f, %f]\n",
				bmin[0], bmin[1], bmin[2],
				bmax[0], bmax[1], bmax[2]);
		}
	}
	catch (MeshException e)
	{
		fprintf(stderr, "  error loading mesh file %s : %s\n", filename, e.what());
		exit(1);
	}
	fprintf(stderr, "loading mesh file %s... done (%d triangles, %d materials, %d groups)\n", filename, m_mesh.getNumTriangles(), m_mesh.getNumMaterials(), m_mesh.getNumGroups());
	{
		// print the group names
		for (int32 i = 0; i < m_mesh.getNumGroups(); ++i)
			fprintf(stderr, "    group[%d] : %s, %u triangles\n", i,
				m_mesh.getGroupName(i).c_str(),
				m_mesh.getGroupOffsets()[i + 1] - m_mesh.getGroupOffsets()[i]);
	}

	// load all textures
	{
		fprintf(stderr, "loading %u textures... started\n", (uint32)m_mesh.m_textures.size());

		m_textures_h.resize( m_mesh.m_textures.size() );
		m_textures_d.resize( m_mesh.m_textures.size() );
		for (size_t i = 0; i < m_mesh.m_textures.size(); ++i)
		{
			m_textures_h[i] = HostMipMapStoragePtr(new MipMapStorage<HOST_BUFFER>());
			m_textures_d[i] = DeviceMipMapStoragePtr(new MipMapStorage<CUDA_BUFFER>());

			// try to load the texture
			char local_path[2048];
			extract_path(filename, local_path);

			char texture_name[2048];
			strcpy(texture_name, m_mesh.m_textures[i].c_str());

			if (find_file(texture_name, scene_dirs))
			{
				if (strcmp(texture_name + strlen(texture_name) - 4, ".tga") == 0)
				{
					cugar::TGAHeader tga_header;
					unsigned char* rgb = cugar::load_tga(texture_name, &tga_header);

					if (rgb)
					{
						MipMapStorage<HOST_BUFFER>::TexturePtr texture_h(new TextureStorage<HOST_BUFFER>());

						texture_h->resize(tga_header.width, tga_header.height);

						float4* tex = texture_h->ptr();

						for (uint32 p = 0; p < uint32(tga_header.width) * uint32(tga_header.height); ++p)
							tex[p] = make_float4(
								float(rgb[3 * p + 0]) / 255.0f,
								float(rgb[3 * p + 1]) / 255.0f,
								float(rgb[3 * p + 2]) / 255.0f,
								0.0f);

						// generate the mipmap for this texture
						m_textures_h[i]->set(texture_h);

						// and copy it to the device
						*m_textures_d[i] = *m_textures_h[i];

						delete[] rgb;
					}
					else
						fprintf(stderr, "warning: unable to load texture %s\n", texture_name);
				}
				else
					fprintf(stderr, "warning: unsupported texture format %s\n", texture_name);
			}
			else
				fprintf(stderr, "warning: unable to find texture %s\n", texture_name);
		}

		m_texture_views_h.alloc(m_mesh.m_textures.size());
		for (uint32 i = 0; i < m_textures_h.size(); ++i)
			m_texture_views_h.set(i, m_textures_h[i]->view());

		m_texture_views_d.alloc(m_mesh.m_textures.size());
		for (uint32 i = 0; i < m_textures_d.size(); ++i)
			m_texture_views_d.set(i, m_textures_d[i]->view());

		fprintf(stderr, "loading %u textures... done\n", (uint32)m_mesh.m_textures.size());
	}

	// checking materials
	for (int32_t i = 0; i < m_mesh.getNumTriangles(); ++i)
	{
		const int m = m_mesh.getMaterialIndices()[i];
		if (m < 0 || m >= m_mesh.getNumMaterials())
		{
			fprintf(stderr, "material[%u] : %u out of range\n", i, m);
			exit(1);
		}
	}

    // copy to the device
	m_mesh_d = m_mesh;
	{
		size_t mem_free, mem_tot;
		cudaSetDevice(0);
		cudaMemGetInfo(&mem_free, &mem_tot);
		fprintf(stderr, "free device memory: %.3f GB\n", float(mem_free) / (1024 * 1024 * 1024));
	}

	fprintf(stderr, "creatign RT index... started\n");

	try
	{
		m_model = m_context->createModel();
		m_model->setTriangles(
			m_mesh.getNumTriangles(), RTP_BUFFER_TYPE_HOST, m_mesh.getVertexIndices(),
			m_mesh.getNumVertices(), RTP_BUFFER_TYPE_HOST, m_mesh.getVertexData());
		m_model->update(0);
	}
	catch (optix::prime::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}

	fprintf(stderr, "creatign RT index... done\n");

	fprintf(stderr, "initializing path sampler... started\n");

	m_renderer->init(argc, argv, *this);

	fprintf(stderr, "initializing path sampler... done\n");
	{
		size_t mem_free, mem_tot;
		cudaSetDevice(0);
		cudaMemGetInfo(&mem_free, &mem_tot);
		fprintf(stderr, "free device memory: %.3f GB\n", float(mem_free) / (1024 * 1024 * 1024));
	}

#if 0
	cugar::host_vector<uint32_t> h_randoms(1024 * 1024);
	for (uint32_t i = 0; i < 1024 * 1024; ++i)
		h_randoms[i] = rand();
	cugar::device_vector<uint32_t> d_randoms = h_randoms;
	cugar::device_vector<uint32_t> d_vals = h_randoms;
	cugar::device_vector<uint8_t> temp_storage;
	cugar::radix_sort<cugar::device_tag>(1024 * 1024, cugar::raw_pointer(d_randoms), cugar::raw_pointer(d_vals), temp_storage);

	for (uint32_t i = 0; i < 10; ++i)
	{
		d_randoms = h_randoms;

		const uint32_t n_keys = (1u << (i + 1)) * 1024;

		cugar::cuda::Timer timer;
		timer.start();

		cugar::radix_sort<cugar::device_tag>(n_keys, cugar::raw_pointer(d_randoms), cugar::raw_pointer(d_vals), temp_storage);

		timer.stop();
		fprintf(stderr, "%u K items : %.2fms\n", n_keys / 1024, timer.seconds() * 1000.0f);
	}
#endif
}

void Renderer::clear()
{
	for (uint32_t c = 0; c < m_fb.channel_count(); ++c)
		m_fb.channels[c].clear();
}

void Renderer::update_model()
{
	//m_model = m_context->createModel();
	m_model->setTriangles(
		m_mesh.getNumTriangles(), RTP_BUFFER_TYPE_HOST, m_mesh.getVertexIndices(),
		m_mesh.getNumVertices(), RTP_BUFFER_TYPE_HOST, m_mesh.getVertexData());
	m_model->update(0);
	m_model->finish();
	CUDA_CHECK(cugar::cuda::sync_and_check_error("model update"));

	// copy to the device
	m_mesh_d = m_mesh;

	// TODO: update m_mesh_lights if needed!
}

// Renderer display function
//
void Renderer::render(const uint32 instance)
{
	try
	{
		RendererView renderer_view = view(instance);

		// clear the primary Gbuffer
		m_fb.gbuffer.clear();

		//cudaDeviceSynchronize();

		m_renderer->render(instance, *this);

		// apply filtering, if enabled
		filter( instance );
		
		to_rgba(renderer_view, m_rgba.ptr());
	}
	catch (cugar::cuda_error& error)
	{
		fprintf(stderr, "caught cuda error: %s\n", error.what());
		exit(0);
	}
}

RendererView Renderer::view(const uint32 instance)
{
	RendererView renderer_view(
		m_camera,
		m_light,
        m_mesh_d.view(),
		m_mesh_lights.view(false),
		m_mesh_lights.view(true),
		m_texture_views_d.ptr(),
		m_ltc_size,
		m_ltc_M.ptr(),
		m_ltc_Minv.ptr(),
		m_ltc_A.ptr(),
		m_res_x,
		m_res_y,
		m_aspect,
		m_exposure,
        m_shading_rate,
        m_shading_mode,
		m_fb.view(),
		instance );

    return renderer_view;
}

void Renderer::filter(const uint32 instance)
{
	// setup some ping-pong buffers
	FBufferChannelView pingpong[2];
	pingpong[0] = m_fb_temp[0].view();
	pingpong[1] = m_fb_temp[1].view();

	// clear the output filter
	m_fb.channels[FBufferDesc::FILTERED_C] = m_fb.channels[FBufferDesc::DIRECT_C];

	FBufferChannelView output = m_fb.channels[FBufferDesc::FILTERED_C].view();

	EAWParams eaw_params;
	eaw_params.phi_normal	= /*sqrtf(float(instance + 1)) **/ 128.0f;
	eaw_params.phi_position = /*sqrtf(float(instance + 1)) **/ 8.0f;
	eaw_params.phi_color	= float(instance + 1) / 20.0f;
	//eaw_params.phi_color	= float(instance*instance + 1) / 10000.0f;

	const uint32 n_iterations = 5;

	// filter the diffuse channel
	{
		GBufferView			gbuffer = m_fb.gbuffer.view();
		FBufferChannelView	input	= m_fb.channels[FBufferDesc::DIFFUSE_C].view();
		FBufferChannelView	weight	= m_fb.channels[FBufferDesc::DIFFUSE_A].view();

		filter_variance(input, m_var.ptr(), 2);

		EAW(
			n_iterations,
			output,						// destination
			weight,						// weight
			input,						// input
			gbuffer,					// gbuffer
			m_var.ptr(),				// variance
			eaw_params, pingpong);
	} 
	// filter the specular channel
	{
		GBufferView			gbuffer	= m_fb.gbuffer.view();
		FBufferChannelView	input	= m_fb.channels[FBufferDesc::SPECULAR_C].view();
		FBufferChannelView	weight	= m_fb.channels[FBufferDesc::SPECULAR_A].view();

		filter_variance(input, m_var.ptr(), 2);

		EAW(
			n_iterations,
			output,						// destination
			weight,						// weight
			input,						// input
			gbuffer,					// gbuffer
			m_var.ptr(),				// variance
			eaw_params, pingpong);
	}
}