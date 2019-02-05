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

#include "xbl.h"

#include <cugar/basic/cuda/arch.h>
#include <cugar/linalg/vector.h>
#include <cugar/spherical/mappings.h>

namespace {

FERMAT_HOST_DEVICE
float norm_diff(const cugar::Vector3f a, const cugar::Vector3f b)
{
	const float d = cugar::max(1e-8f, cugar::dot(a, b));
	return 1.0f - d;
}

__device__ float myexp(float in)
{
	float tmp = 0.0f;
	if (fabsf(in) < 1.0f)
	{
		tmp = (1.0f + 0.45f*in);
	}
	return tmp * tmp;
//	return __expf(in);
}

} // anonymous

/// perform a step of cross-bilateral filtering
///
__global__
void XBL_mad_kernel(
		  FBufferChannelView	dst,
	const uint32				op,
	const FBufferChannelView	w_img,
	const float					w_min,
	const FBufferChannelView	img,
	const GBufferView			gb,
	const float*				var,
	const XBLParams				params,
	const uint32				filter_radius,
	const uint32				step_size,
	const TiledSequenceView		sequence)
{
	const int32 x = threadIdx.x + blockIdx.x * blockDim.x;
	const int32 y = threadIdx.y + blockIdx.y * blockDim.y;

	// check whether this pixel is in range
	if (x >= dst.res_x ||
		y >= dst.res_y)
		return;

	const cugar::Vector4f  weightCenter		= cugar::max( cugar::Vector4f(w_img(x, y)), w_min );
	const cugar::Vector4f  imgCenter		= img(x, y);

	const cugar::Vector4f  colorCenter		=
		(op & kFilterOpModulateInput)   ? imgCenter * weightCenter :
		(op & kFilterOpDemodulateInput) ? imgCenter / weightCenter :
		imgCenter;

	const cugar::Vector4f  packed_geo		= gb.geo(x, y);
	const cugar::Vector3f  normalCenter		= GBufferView::unpack_normal(packed_geo);
	const cugar::Vector3f  positionCenter	= GBufferView::unpack_pos(packed_geo);

	// check whether this pixel represents a miss (TODO: for stochastic effects, we might want to do some filtering in this case too...)
	if (GBufferView::is_miss(packed_geo))
	{
		cugar::Vector4f r  = (op & kFilterOpAddMode) ? dst(x, y) : cugar::Vector4f(0.0f);

		r +=
			(op & kFilterOpModulateOutput)	 ?	colorCenter * weightCenter :
			(op & kFilterOpDemodulateOutput) ?	colorCenter / weightCenter :
												colorCenter;

		dst(x, y) = r;
		return;
	}

	const float posRadius = 20 * cugar::min(
		cugar::length(params.U) / img.res_x,
		cugar::length(params.V) / img.res_y) * dot( positionCenter - params.E, params.W ) / cugar::square_length(params.W);

	const float variance	= var ? var[x + y * img.res_x] : 1.0f;
	const float phiNormal	= params.phi_normal * step_size * step_size;
	const float phiPosition	= params.phi_position / (posRadius*posRadius);
	const float phiColor	= params.phi_color / cugar::max( 1.0e-3f, cugar::sqr(variance) );
	const float nThreshold  = 0.9f;

	float			sumWeight = 0.0;
	cugar::Vector3f	sumColor = cugar::Vector3f(0.0f);

    const float sigma  = 10.0f;
    const float sigma2 = sigma*sigma;

	//for (int yy = -int(filter_radius); yy <= int(filter_radius); yy++)
	for (uint32 s = 0; s < params.taps; ++s)
	{
		const float u = cugar::randfloat(s,0) + sequence.shift(x, y, 0u);
		const float v = cugar::randfloat(s,1) + sequence.shift(x, y, 1u);

		const cugar::Vector2f xy =
			s == 0	? cugar::Vector2f(0.0f)								// make sure to sample the central pixel
					: cugar::square_to_unit_disk(cugar::Vector2f(u,v));
		const float xx = xy.x * filter_radius;
		const float yy = xy.y * filter_radius;

		//for (int xx = -int(filter_radius); xx <= int(filter_radius); xx++)
		{
			const int2 p = make_int2(x + xx * step_size, y + yy * step_size);
			const bool inside =
				/*__all*/(p.x >= 0 && p.y >= 0) &&
				/*__all*/(p.x < img.res_x && p.y < img.res_y);

			if (inside)
			{
				const float d2 = (xx*xx + yy*yy) / sigma2;

				const cugar::Vector4f  weightP = cugar::max( cugar::Vector4f(w_img(p)), w_min );
				const cugar::Vector4f  imgP	   = img(p);

				const cugar::Vector4f  colorP =
					(op & kFilterOpModulateInput)	? imgP * weightP :
					(op & kFilterOpDemodulateInput)	? imgP / weightP :
					imgP;

				const cugar::Vector4f  geoP = gb.geo(p);
				const cugar::Vector3f  normalP = GBufferView::unpack_normal(geoP);
				const cugar::Vector3f  positionP = GBufferView::unpack_pos(geoP);

				if (GBufferView::is_miss(geoP) == false)
				{
					// check whether we have to skip this pixel
					if (dot(normalP, normalCenter) < nThreshold)
						continue;

					// compute the normal weight
					const float		wNormal			= norm_diff(normalP, normalCenter) * phiNormal;

					// compute the color weight
					cugar::Vector3f diffCol			= colorP.xyz() - colorCenter.xyz();
					const float		wColor			= cugar::dot(diffCol, diffCol) * phiColor;

					// compute the positional weight
					cugar::Vector3f	diffPosition	= (positionP - positionCenter);
					const float		wPosition		= dot(diffPosition, diffPosition) * phiPosition;

					const float w = myexp(0.0
						- d2
						- cugar::max(wPosition, 0.0f)
						- cugar::max(wNormal, 0.0f)
						- cugar::max(wColor, 0.0f)
					);

					sumWeight += w;
					sumColor  += w * colorP.xyz();
				}
			}
		}
	}

	cugar::Vector4f r  = (op & kFilterOpAddMode) ? dst(x, y) : cugar::Vector4f(0.0f);

	cugar::Vector4f c = (sumWeight ? cugar::Vector4f(sumColor / sumWeight, colorCenter.w) : colorCenter);

	r +=
		(op & kFilterOpModulateOutput)		? c * weightCenter :
		(op & kFilterOpDemodulateOutput)	? c / weightCenter :
		c;

	dst(x, y) = r;
}

// perform a step of cross-bilateral filtering, multiplying the result by a weight and adding it to the output, i.e. solving:
//
//   dst += w_img * xbl(img)
//
void XBL(
	FBufferChannelView			dst,
	const FilterOp				op,
	const FBufferChannelView	w_img,
	const float					w_min,
	const FBufferChannelView	img,
	const GBufferView			gb,
	const float*				var,
	const XBLParams				params,
	const uint32				filter_radius,
	const uint32				step_size,
	const TiledSequenceView		sequence)
{
	dim3 blockSize(32, 4);
	dim3 gridSize(cugar::divide_ri(dst.res_x, blockSize.x), cugar::divide_ri(dst.res_y, blockSize.y));

	XBL_mad_kernel<< < gridSize, blockSize >> > (dst, op, w_img, w_min, img, gb, var, params, filter_radius, step_size, sequence);
	CUDA_CHECK(cugar::cuda::sync_and_check_error("XBL_mad_kernel"));
}
