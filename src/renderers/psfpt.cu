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

#define DEVICE_TIMING 0

#include <psfpt.h>
#include <psfpt_impl.h>

void PSFPT::keyboard(unsigned char character, int x, int y, bool& invalidate)
{
	invalidate = false;

	switch (character)
	{
	case 'i':
		m_options.direct_lighting = !m_options.direct_lighting;
		invalidate = true;
		break;
	case 'w':
		m_options.psf_width *= 0.5f;
		invalidate = true;
		fprintf(stderr, "\npsf width = %f\n", m_options.psf_width);
		break;
	case 'W':
		m_options.psf_width *= 2.0f;
		invalidate = true;
		fprintf(stderr, "\npsf width = %f\n", m_options.psf_width);
		break;
	case 'k':
		if (m_options.psf_depth)
			m_options.psf_depth--;
		invalidate = true;
		fprintf(stderr, "\npsf depth = %u\n", m_options.psf_depth);
		break;
	case 'K':
		m_options.psf_depth++;
		invalidate = true;
		fprintf(stderr, "\npsf depth = %u\n", m_options.psf_depth);
		break;
	}
}

/*
// dump some speed stats
//
void PSFPT::dump_speed_stats(FILE* stats)
{
	fprintf(stats, "%f, %f, %f, %f, %f\n",
		m_stats.primary_rt_time.mean() * 1000.0f,
		m_stats.path_rt_time.mean() * 1000.0f,
		m_stats.shadow_rt_time.mean() * 1000.0f,
		m_stats.path_shade_time.mean() * 1000.0f,
		m_stats.shadow_shade_time.mean() * 1000.0f);
}
*/