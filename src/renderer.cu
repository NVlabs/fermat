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

#include <renderer.h>
#include <renderer_impl.h>
#include <pathtracer.h>
#include <rt.h>
#include <files.h>
#include <bpt.h>
#include <mlt.h>
#include <cmlt.h>
#include <pssmlt.h>
#include <rpt.h>
#include <psfpt.h>
#include <fermat_loader.h>
#include <pbrt_importer.h>
#include <mesh/MeshStorage.h>
#include <eaw.h>
#include <xbl.h>
#include <cugar/basic/cuda/arch.h>
#include <cugar/basic/cuda/timer.h>
#include <cugar/basic/primitives.h>
#include <cugar/basic/functors.h>
#include <cugar/basic/cuda/sort.h>
#include <cugar/basic/timer.h>
#include <cugar/image/tga.h>
#include <cugar/bsdf/ltc.h>
#include <buffers.h>
#include <vector>

namespace ltc_ggx
{
	typedef float mat33[9];

#include <cugar/bsdf/ltc_ggx.inc>
};

void load_assimp(const char* filename, MeshStorage& out_mesh, const std::vector<std::string>& dirs, std::vector<std::string>& scene_dirs);


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
__global__ void to_rgba_kernel(const RenderingContextView renderer, uint8* rgba)
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
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

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
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

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
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

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
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

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
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

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
			c = powf(c, 1.0f / renderer.gamma);

			rgba[idx * 4 + 0] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kUV)
		{
			cugar::Vector4f c = renderer.fb.gbuffer.uv(idx);

			// visualize the ST interpolated texture coordinates
			c.x = c.z;
			c.y = c.w;
			c.z = 0.5f;
			c.w = 0.0f;

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kCharts)
		{
			const uint32 tri_id = renderer.fb.gbuffer.tri(idx);

			// find the chart containing this triangle
			const uint32 group_id = tri_id < renderer.mesh.num_triangles ?
				cugar::upper_bound_index( tri_id, renderer.mesh.group_offsets, renderer.mesh.num_groups+1 ) : uint32(-1);

			// visualize the chart index as a color
			cugar::Vector4f c;
			c.x = cugar::randfloat(0, group_id) * 0.5f + 0.5f;
			c.y = cugar::randfloat(1, group_id) * 0.5f + 0.5f;
			c.z = cugar::randfloat(2, group_id) * 0.5f + 0.5f;
			c.w = 0.0f;

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
		else if (renderer.shading_mode == kNormal)
		{
			cugar::Vector4f geo = renderer.fb.gbuffer.geo(idx);

			cugar::Vector3f normal = GBufferView::unpack_normal(geo);

			rgba[idx * 4 + 0] = uint8(fminf(normal.x * 128.0f + 128.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(normal.y * 128.0f + 128.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(normal.z * 128.0f + 128.0f, 255.0f));
			rgba[idx * 4 + 3] = 0;
		}
		else if (renderer.shading_mode >= kAux0 && (renderer.shading_mode - kAux0 < renderer.fb.n_channels - FBufferDesc::NUM_CHANNELS))
		{
			const uint32 aux_channel = renderer.shading_mode - kAux0 + FBufferDesc::NUM_CHANNELS;
			cugar::Vector4f c = renderer.fb(aux_channel, idx);

			c *= renderer.exposure; // Hardcoded Exposure Adjustment
			c = c / (c + cugar::Vector4f(1));
			c.x = powf(c.x, 1.0f / renderer.gamma);
			c.y = powf(c.y, 1.0f / renderer.gamma);
			c.z = powf(c.z, 1.0f / renderer.gamma);
			c.w = powf(c.w, 1.0f / renderer.gamma);

			rgba[idx * 4 + 0] = uint8(fminf(c.x * 256.0f, 255.0f));
			rgba[idx * 4 + 1] = uint8(fminf(c.y * 256.0f, 255.0f));
			rgba[idx * 4 + 2] = uint8(fminf(c.z * 256.0f, 255.0f));
			rgba[idx * 4 + 3] = uint8(fminf(c.w * 256.0f, 255.0f));
		}
	}
}

void to_rgba(const RenderingContextView renderer, uint8* rgba)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(renderer.res_x * renderer.res_y, blockSize.x));
	to_rgba_kernel <<< gridSize, blockSize >>>(renderer, rgba);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("to_rgba"));
}
//------------------------------------------------------------------------------
__global__ void multiply_frame_kernel(RenderingContextView renderer, const float scale)
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
__global__ void clamp_frame_kernel(RenderingContextView renderer, const float max_value)
{
	const uint32 idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < renderer.res_x * renderer.res_y)
	{
		renderer.fb(FBufferDesc::DIFFUSE_C,		idx)	= cugar::min( cugar::Vector4f(renderer.fb(FBufferDesc::DIFFUSE_C,		idx)), max_value );
		renderer.fb(FBufferDesc::SPECULAR_C,	idx)	= cugar::min( cugar::Vector4f(renderer.fb(FBufferDesc::SPECULAR_C,		idx)), max_value );
		renderer.fb(FBufferDesc::DIRECT_C,		idx)	= cugar::min( cugar::Vector4f(renderer.fb(FBufferDesc::DIRECT_C,		idx)), max_value );
		renderer.fb(FBufferDesc::COMPOSITED_C,	idx)	= cugar::min( cugar::Vector4f(renderer.fb(FBufferDesc::COMPOSITED_C,	idx)), max_value );

		FERMAT_ASSERT(
			cugar::is_finite( renderer.fb(FBufferDesc::COMPOSITED_C,	idx).x ) &&
			cugar::is_finite( renderer.fb(FBufferDesc::COMPOSITED_C,	idx).y ) &&
			cugar::is_finite( renderer.fb(FBufferDesc::COMPOSITED_C,	idx).z ) &&
			cugar::is_finite( renderer.fb(FBufferDesc::COMPOSITED_C,	idx).w ) );
	}
}
//------------------------------------------------------------------------------
__global__ void update_variances_kernel(RenderingContextView renderer, const uint32 n)
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

void RenderingContextImpl::multiply_frame(const float scale)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(m_res_x * m_res_y, blockSize.x));
	multiply_frame_kernel <<< gridSize, blockSize >>>(view(0), scale);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("multiply_frame") );
}

//------------------------------------------------------------------------------

void RenderingContextImpl::rescale_frame(const uint32 instance)
{
	multiply_frame( float(instance)/float(instance+1) );
}

// clamp the output framebuffer to a given maximum
//
// \param max_value
void RenderingContextImpl::clamp_frame(const float max_value)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(m_res_x * m_res_y, blockSize.x));
	clamp_frame_kernel <<< gridSize, blockSize >>>(view(0), max_value);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("clamp_frame") );
}

//------------------------------------------------------------------------------

void RenderingContextImpl::update_variances(const uint32 instance)
{
	dim3 blockSize(128);
	dim3 gridSize(cugar::divide_ri(m_res_x * m_res_y, blockSize.x));
	update_variances_kernel <<< gridSize, blockSize >>>(view(0), instance + 1);
	CUDA_CHECK( cugar::cuda::sync_and_check_error("update_variances") );
}

// load a plugin
//
uint32 RenderingContextImpl::load_plugin(const char* plugin_name)
{
	typedef uint32 (__stdcall *register_plugin_function)(RenderingContext& renderer);

	fprintf(stderr, "  loading plugin \"%s\"... started\n", plugin_name);
	m_plugins.push_back(DLL(plugin_name));

	register_plugin_function plugin_entry_function = (register_plugin_function)m_plugins.front().get_proc_address("register_plugin");
	if (!plugin_entry_function)
	{
		fprintf(stderr, "failed loading plugin entry function!\n");
		throw cugar::runtime_error("failed loading plugin entry function");
	}
	fprintf(stderr, "  loading plugin \"%s\"... done\n", plugin_name);

	fprintf(stderr, "  initializing plugin \"%s\"... started\n", plugin_name);
	const uint32 r = plugin_entry_function( *m_this );
	fprintf(stderr, "  initializing plugin \"%s\"... done\n", plugin_name);
	return r;
}

//------------------------------------------------------------------------------


// RenderingContext initialization
//
void RenderingContextImpl::init(int argc, char** argv)
{
	const char* filename = NULL;

	register_renderer("pt", &PathTracer::factory );
	register_renderer("bpt", &BPT::factory );
	register_renderer("cmlt", &CMLT::factory );
	register_renderer("mlt", &MLT::factory );
	register_renderer("pssmlt", &PSSMLT::factory );
	register_renderer("rpt", &RPT::factory );
	register_renderer("psfpt", &PSFPT::factory );
	//register_renderer("hellopt", &HelloPT::factory );

	m_renderer_type = kBPT;
	m_exposure = 1.0f;
	m_gamma = 2.2f;
	m_res_x = 1600;
	m_res_y = 900;
	m_aspect = 0.0f;
	m_shading_rate = 1.0f;
	m_shading_mode = kShaded;

	// set the directional light
	m_light.dir   = cugar::normalize(cugar::Vector3f(1.0f,-0.5f,1.0f));
	m_light.color = cugar::Vector3f(22.0f,21.0f,18.0f)*4;

	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-i") == 0)
			filename = argv[++i];
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
		else if (strcmp(argv[i], "-plugin") == 0)
		{
			m_renderer_type = load_plugin( argv[++i] );
			m_renderer = m_renderer_factories[m_renderer_type]();
		}
		else if (argv[i][0] == '-')
		{
			for (uint32 r = 0; r < m_renderer_names.size(); ++r)
			{
				if (m_renderer_names[r] == argv[i]+1)
				{
					m_renderer_type = r;
					m_renderer = m_renderer_factories[r]();
				}
			}
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

	bool overwrite_camera = false;
	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "-c") == 0)
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

			overwrite_camera = true;
		}
	}

	m_rgba.alloc(m_res_x * m_res_y * 4);
	m_var.alloc(m_res_x * m_res_y);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	fprintf(stderr, "cuda device: %s\n", prop.name);
	fprintf(stderr, "  SM version : %d.%d\n",
		prop.major, prop.minor);
	fprintf(stderr, "  SM count   : %d \n",
		prop.multiProcessorCount);
	fprintf(stderr, "  SM clock   : %d \n",
		prop.clockRate);
	fprintf(stderr, "  mem clock  : %d \n",
		prop.memoryClockRate);
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	fprintf(stderr, "  memory     : %.3f GB\n",
		float(total) / (1024 * 1024 * 1024));

	std::vector<unsigned int> devices(1);
	devices[0] = 0;

	cudaSetDevice( devices[0] );

	// make sure we do have a renderer
	if (m_renderer == NULL)
		m_renderer = PathTracer::factory();

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
	m_fb_temp[2].resize(m_res_x, m_res_y);
	m_fb_temp[3].resize(m_res_x, m_res_y);

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

	// Load the glossy reflectance profile
	{
		fprintf(stderr, "initializing glossy reflectance profile... started\n");
		DomainBuffer<HOST_BUFFER, float>		glossy_reflectance;

		const uint32 S = 32;

		glossy_reflectance.alloc(S*S*S*S);

		ScopedFile file("glossy_reflectance.dat", "rb");
		if (!file)
		{
			fprintf(stderr, "  error opening glossy_reflectance.dat\n");
			exit(1);
		}
		if (fread(glossy_reflectance.ptr(), sizeof(float), S*S*S*S, file) != S*S*S*S)
		{
			fprintf(stderr, "  error loading glossy_reflectance.dat\n");
			exit(1);
		}

		m_glossy_reflectance = glossy_reflectance;
		fprintf(stderr, "initializing glossy reflectance profile... done\n");
	}

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
		std::vector<Camera>				cameras;
		std::vector<DirectionalLight>	dir_lights;

		if (strlen(filename) > 3 && strcmp(filename+strlen(filename)-3, ".fa") == 0)
			load_scene(filename, m_mesh, cameras, dir_lights, dirs, scene_dirs);
		else if ((strlen(filename) > 4 && strcmp(filename+strlen(filename)-4, ".obj") == 0) ||
	 			 (strlen(filename) > 4 && strcmp(filename+strlen(filename)-4, ".ply") == 0))
			loadModel(filename, m_mesh);
		else if (strlen(filename) > 5 && strcmp(filename+strlen(filename)-5, ".pbrt") == 0)
		{
			pbrt::FermatImporter importer(filename, &m_mesh, &m_camera, &dir_lights, &scene_dirs);
			pbrt::import(filename, &importer);
			importer.finish();

			// copy the film options
			m_exposure	= importer.m_film.exposure;
			m_gamma		= importer.m_film.gamma;
		}
		else
			load_assimp(filename, m_mesh, dirs, scene_dirs);

		// check whether we need to pick the loaded camera
		if (cameras.size() && overwrite_camera == false)
			m_camera = cameras[0];

		// store directional lights on both host and device
		m_dir_lights_h.alloc( dir_lights.size() );
		m_dir_lights_h.copy_from( dir_lights.size(), HOST_BUFFER, &dir_lights.front() );
		m_dir_lights_d = m_dir_lights_h;

		// perform normal compression
		m_mesh.compress_normals();
		m_mesh.compress_tex();
			
		#if UNIFIED_VERTEX_ATTRIBUTES
		// unify vertex attributes
		unify_vertex_attributes( m_mesh );
		#endif

		// apply material flags
		apply_material_flags( m_mesh );

		// compute the bbox
		if (1)
		{
			cugar::Vector3f bmin(1.0e16f, 1.0e16f, 1.0e16f);
			cugar::Vector3f bmax(-1.0e16f, -1.0e16f, -1.0e16f);

			MeshView::vertex_type* v = reinterpret_cast<MeshView::vertex_type*>(m_mesh.getVertexData());
			for (int32_t i = 0; i < m_mesh.getNumVertices(); ++i)
			{
				bmin = cugar::min(bmin, vertex_comp(v[i]));
				bmax = cugar::max(bmax, vertex_comp(v[i]));
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
	fprintf(stderr, "loading mesh file %s... done\n", filename);
	fprintf(stderr, "  triangles : %d\n", m_mesh.getNumTriangles());
	fprintf(stderr, "  vertices  : %d\n", m_mesh.getNumVertices());
	fprintf(stderr, "  normals   : %d\n", m_mesh.getNumNormals());
	fprintf(stderr, "  materials : %d\n", m_mesh.getNumMaterials());
	fprintf(stderr, "  groups    : %d\n", m_mesh.getNumGroups());
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

#if 0
    fprintf(stderr, "creating UV index... started\n");
    {
        // initialize a uv-bvh on the host
        HostUVBvh uv_bvh;

        build( &uv_bvh, m_mesh );

        output_uv_tris( m_mesh );

        // and copy it to the device
        m_uv_bvh = uv_bvh;
    }
    fprintf(stderr, "creating UV index... done\n");
#endif

    // copy to the device
	m_mesh_d = m_mesh;
	{
		size_t mem_free, mem_tot;
		cudaSetDevice(0);
		cudaMemGetInfo(&mem_free, &mem_tot);
		fprintf(stderr, "free device memory: %.3f GB\n", float(mem_free) / (1024 * 1024 * 1024));
	}

	fprintf(stderr, "creating RT index... started\n");

  #if 1
	m_rt_context = new RTContext();
	m_rt_context->create_geometry(
		m_mesh_d.getNumTriangles(),
		m_mesh_d.getVertexIndices(),
		m_mesh_d.getNumVertices(),
		m_mesh_d.getVertexData(),
		m_mesh_d.getNormalIndices(),
		m_mesh_d.getNormalData(),
		m_mesh_d.getTextureCoordinateIndices(),
		m_mesh_d.getTextureCoordinateData(),
		m_mesh_d.getMaterialIndices());

	// setup the material buffer
	m_rt_context->bind_buffer( "g_materials", m_mesh_d.getNumMaterials(), sizeof(MeshMaterial), m_mesh_d.m_materials.ptr(), RT_FORMAT_USER );

	// setup texture buffers
	//m_rt_context->bind_buffer( "g_textures", m_texture_views_d.count(), sizeof(MipMapView), m_texture_views_d.ptr(), RT_FORMAT_USER );
	
	// perform a small test launch
	//m_rt_context->launch(0,128);
  #else
	m_rt_context = NULL;
  #endif

	fprintf(stderr, "creating RT index... done\n");

	const uint32 n_dimensions	= 6 * 12;
	const uint32 tiled_dim		= 256;
	fprintf(stderr, "  initializing sampler: %u dimensions\n", n_dimensions);

	m_sequence.setup(n_dimensions, tiled_dim);

	fprintf(stderr, "initializing path sampler... started\n");

	m_renderer->init(argc, argv, *m_this);

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

void RenderingContextImpl::clear()
{
	for (uint32_t c = 0; c < m_fb.channel_count(); ++c)
		m_fb.channels[c].clear();
}

void RenderingContextImpl::update_model()
{
	m_rt_context->create_geometry(
		m_mesh_d.getNumTriangles(),
		m_mesh_d.getVertexIndices(),
		m_mesh_d.getNumVertices(),
		m_mesh_d.getVertexData(),
		m_mesh_d.getNormalIndices(),
		m_mesh_d.getNormalData(),
		m_mesh_d.getTextureCoordinateIndices(),
		m_mesh_d.getTextureCoordinateData(),
		m_mesh_d.getMaterialIndices());

	// TODO: update m_mesh_lights if needed!
	m_renderer->update_scene(*m_this);

	// TODO: update the m_rt_context!
}

// register a new rendering interface type
//
uint32 RenderingContextImpl::register_renderer(const char* name, RendererFactoryFunction factory)
{
	m_renderer_names.push_back( name );
	m_renderer_factories.push_back( factory );
	return uint32( m_renderer_factories.size() - 1 );
}

// RenderingContext display function
//
void RenderingContextImpl::render(const uint32 instance)
{
	try
	{
		RenderingContextView renderer_view = view(instance);

		// setup optix vars
		m_rt_context->bind_var( "g_renderer", renderer_view );

		// clear the primary Gbuffer
		m_fb.gbuffer.clear();

		//cudaDeviceSynchronize();

		m_renderer->render(instance, *m_this);

		// apply filtering, if enabled
		if (m_shading_mode == kFiltered)
			filter( instance );
		
		to_rgba(renderer_view, m_rgba.ptr());
	}
	catch (cugar::cuda_error& error)
	{
		fprintf(stderr, "caught cuda error: %s\n", error.what());
		exit(0);
	}
}

RenderingContextView RenderingContextImpl::view(const uint32 instance)
{
	RenderingContextView renderer_view(
		m_camera,
		(uint32)m_dir_lights_d.count(),
		m_dir_lights_d.ptr(),
        m_mesh_d.view(),
		m_mesh_lights.view(false),
		m_mesh_lights.view(true),
		m_texture_views_d.ptr(),
		m_ltc_size,
		m_ltc_M.ptr(),
		m_ltc_Minv.ptr(),
		m_ltc_A.ptr(),
		m_glossy_reflectance.ptr(),
		m_res_x,
		m_res_y,
		m_aspect,
		m_exposure,
		m_gamma,
        m_shading_rate,
        m_shading_mode,
		m_fb.view(),
		instance );

    return renderer_view;
}

// compute the scene's bbox
//
cugar::Bbox3f RenderingContextImpl::compute_bbox()
{
	MeshView mesh_view = m_mesh.view();

	cugar::Bbox3f bbox;
	for (int32_t i = 0; i < m_mesh.getNumVertices(); ++i)
		bbox.insert( load_vertex( mesh_view, i ) );

	return bbox;
}

void RenderingContextImpl::filter(const uint32 instance)
{
	// clear the output filter
	m_fb.channels[FBufferDesc::FILTERED_C] = m_fb.channels[FBufferDesc::DIRECT_C];

	FBufferChannelView output = m_fb.channels[FBufferDesc::FILTERED_C].view();

	cugar::Vector3f U, V, W;
	camera_frame( m_camera, m_aspect, U, V, W );

#if 1
	// setup some ping-pong buffers
	FBufferChannelView pingpong[2];
	pingpong[0] = m_fb_temp[0].view();
	pingpong[1] = m_fb_temp[1].view();

	EAWParams eaw_params;
	eaw_params.phi_normal	= /*sqrtf(float(instance + 1)) **/ 2.0f;
	eaw_params.phi_position = /*sqrtf(float(instance + 1)) **/ 1.0f;
	//eaw_params.phi_color	= float(instance + 1) / 20.0f;
	eaw_params.phi_color	= float(instance*instance + 1) / 10000.0f;
	eaw_params.E            = m_camera.eye;
	eaw_params.U            = U;
	eaw_params.V            = V;
	eaw_params.W            = W;

	const uint32 n_iterations = 7;

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
#elif 0
	XBLParams xbl_params;
	xbl_params.taps			= 32;
	xbl_params.phi_normal	= 32.0f;
	xbl_params.phi_position = 1.0f;
	xbl_params.phi_color	= 0.0f;
	//xbl_params.phi_color	= float(instance*instance + 1) / 10000.0f;
	//eaw_params.phi_color	= float(instance*instance + 1) / 10000.0f;
	xbl_params.E            = m_camera.eye;
	xbl_params.U            = U;
	xbl_params.V            = V;
	xbl_params.W            = W;

	// filter the diffuse channel
	{
		GBufferView			gbuffer = m_fb.gbuffer.view();
		FBufferChannelView	input	= m_fb.channels[FBufferDesc::DIFFUSE_C].view();
		FBufferChannelView	weight	= m_fb.channels[FBufferDesc::DIFFUSE_A].view();

		filter_variance(input, m_var.ptr(), 2);

		XBL(
			output,						// destination
			FilterOp(kFilterOpDemodulateInput | kFilterOpModulateOutput | kFilterOpAddMode),
			weight,						// weight
			1.0e-4f,					// min weight
			input,						// input
			gbuffer,					// gbuffer
			m_var.ptr(),				// variance
			xbl_params,
			21u,
			1u,
			m_sequence.view());
	} 
	// filter the specular channel
	{
		GBufferView			gbuffer	= m_fb.gbuffer.view();
		FBufferChannelView	input	= m_fb.channels[FBufferDesc::SPECULAR_C].view();
		FBufferChannelView	weight	= m_fb.channels[FBufferDesc::SPECULAR_A].view();

		filter_variance(input, m_var.ptr(), 2);

		XBL(
			output,						// destination
			FilterOp(kFilterOpDemodulateInput | kFilterOpModulateOutput | kFilterOpAddMode),
			weight,						// weight
			1.0e-4f,					// min weight
			input,						// input
			gbuffer,					// gbuffer
			m_var.ptr(),				// variance
			xbl_params,
			21u,
			1u,
			m_sequence.view());
	}
#endif
}

// constructor
//
RenderingContext::RenderingContext()
{
	m_impl = new RenderingContextImpl( this );
}

// initialize the renderer
//
void RenderingContext::init(int argc, char** argv)
{
	m_impl->init( argc, argv );
}

// render a frame
//
// \param instance		the sequence instance / frame number in a progressive render
void RenderingContext::render(const uint32 instance)
{
	m_impl->render( instance );
}

// clear all framebuffers
//
void RenderingContext::clear()
{
	m_impl->clear();	
}

// rescale the output framebuffer by a constant
//
void RenderingContext::multiply_frame(const float scale)
{
	m_impl->multiply_frame( scale );
}

// rescale the output framebuffer by n/(n-1)
//
// \param instance		the sequence instance / frame number in a progressive render, used for rescaling
void RenderingContext::rescale_frame(const uint32 instance)
{
	m_impl->rescale_frame( instance );
}

// clamp the output framebuffer to a given maximum
//
// \param max_value
void RenderingContext::clamp_frame(const float max_value)
{
	m_impl->clamp_frame( max_value );
}

// update the variance estimates
//
// \param instance		the sequence instance / frame number in a progressive render, used for rescaling
void RenderingContext::update_variances(const uint32 instance)
{
	m_impl->update_variances( instance );
}

// update the internal data-structures (e.g. BVHs) associated to the geometry
//
void RenderingContext::update_model()
{
	m_impl->update_model();
}

// perform filtering
//
// \param instance		the sequence instance / frame number in a progressive render
void RenderingContext::filter(const uint32 instance)
{
	m_impl->filter( instance );
}

// return the current output resolution
//
uint2 RenderingContext::res() const { return m_impl->res(); }

// return a view of the renderer
//
RenderingContextView RenderingContext::view(const uint32 instance) { return m_impl->view( instance ); }

// return the camera
//
Camera& RenderingContext::get_camera() { return m_impl->get_camera(); }

// return the directional light count
//
uint32 RenderingContext::get_directional_light_count() const
{
	return (uint32)m_impl->m_dir_lights_d.count();
}

// return the host-side directional lights
//
const DirectionalLight* RenderingContext::get_host_directional_lights() const
{
	return m_impl->m_dir_lights_h.ptr();
}

// return the device-side directional lights
//
const DirectionalLight* RenderingContext::get_device_directional_lights() const
{
	return m_impl->m_dir_lights_d.ptr();
}

// set the number of directional lights
//
void RenderingContext::set_directional_light_count(const uint32 count)
{
	m_impl->m_dir_lights_h.alloc( count );
	m_impl->m_dir_lights_d.alloc( count );
}

// set a directional light
//
void RenderingContext::set_directional_light(const uint32 i, const DirectionalLight& light)
{
	m_impl->m_dir_lights_h.set( i, light );
	m_impl->m_dir_lights_d.set( i, light );
}

// return the target resolution
//
uint2 RenderingContext::get_res() const { return m_impl->get_res(); }

// return the target aspect ratio
//
float RenderingContext::get_aspect_ratio() const { return m_impl->get_aspect_ratio(); }

// return the target exposure
//
void RenderingContext::set_aspect_ratio(const float v) { m_impl->m_aspect = v; }

// return the target exposure
//
float RenderingContext::get_exposure() const { return m_impl->get_exposure(); }

// set the target exposure
//
void RenderingContext::set_exposure(const float v) { m_impl->m_exposure = v; }

// return the target gamma
//
float RenderingContext::get_gamma() const { return m_impl->m_gamma; }

// set the target gamma
//
void RenderingContext::set_gamma(const float v) { m_impl->m_gamma = v; }

// return the shading mode
//
ShadingMode& RenderingContext::get_shading_mode() { return m_impl->m_shading_mode; }

// return the frame buffer
//
FBufferStorage& RenderingContext::get_frame_buffer() { return m_impl->m_fb; }

// return the frame buffer
//
uint8* RenderingContext::get_device_rgba_buffer() { return m_impl->m_rgba.ptr(); }

// return the number of textures
//
uint32 RenderingContext::get_texture_count() const { return uint32( m_impl->m_textures_h.size() ); }

// return the scene's host-side textures
//
RenderingContext::HostMipMapStoragePtr* RenderingContext::get_host_textures() { return &m_impl->m_textures_h.front(); }

// return the scene's device-side textures
//
RenderingContext::DeviceMipMapStoragePtr* RenderingContext::get_device_textures() { return &m_impl->m_textures_d.front(); }

// return the scene's host-side textures
//
MipMapView* RenderingContext::get_host_texture_views() { return m_impl->get_host_texture_views(); }

// return the scene's device-side textures
//
MipMapView* RenderingContext::get_device_texture_views() { return m_impl->get_device_texture_views(); }

// return the scene's host-side mesh
//
MeshStorage& RenderingContext::get_host_mesh() { return m_impl->get_host_mesh(); }

// return the scene's device-side mesh
//
DeviceMeshStorage& RenderingContext::get_device_mesh() { return m_impl->get_device_mesh(); }

// return the scene's device-side mesh emitters
//
MeshLightsStorage& RenderingContext::get_mesh_lights() { return m_impl->get_mesh_lights(); }

// return the ray tracing context
//
RTContext* RenderingContext::get_rt_context() const { return m_impl->get_rt_context(); }

// return the sampling sequence
//
TiledSequence& RenderingContext::get_sequence() { return m_impl->m_sequence; }

// return the renderer
//
RendererInterface* RenderingContext::get_renderer() const { return m_impl->get_renderer(); }

// register a new rendering interface type
//
uint32 RenderingContext::register_renderer(const char* name, RendererFactoryFunction factory)
{
	return m_impl->register_renderer( name, factory );
}

// compute the scene's bbox
//
cugar::Bbox3f RenderingContext::compute_bbox() { return m_impl->compute_bbox(); }
