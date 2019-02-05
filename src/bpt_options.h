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

#pragma once

#include <types.h>

///@addtogroup Fermat
///@{

///@addtogroup BPTLib
///@{

///
/// Basic options for bidirectional path tracing
///
struct BPTOptionsBase
{
	uint32	max_path_length;
	bool	direct_lighting_nee;
	bool	direct_lighting_bsdf;
	bool	indirect_lighting_nee;
	bool	indirect_lighting_bsdf;
	bool	visible_lights;
	bool	use_vpls;
	float	light_tracing;

	FERMAT_HOST_DEVICE
	BPTOptionsBase() :
		max_path_length(6),
		direct_lighting_nee(true),
		direct_lighting_bsdf(true),
		indirect_lighting_nee(true),
		indirect_lighting_bsdf(true),
		visible_lights(true),
		use_vpls(false),
		light_tracing(1.0f) {}

	void parse(const int argc, char** argv)
	{
		for (int i = 0; i < argc; ++i)
		{
			if (strcmp(argv[i], "-pl") == 0 ||
				strcmp(argv[i], "-path-length") == 0 ||
				strcmp(argv[i], "-max-path-length") == 0)
				max_path_length = atoi(argv[++i]);
			else if (strcmp(argv[i], "-bounces") == 0)
				max_path_length = atoi(argv[++i]) + 1;
			else if (strcmp(argv[i], "-nee") == 0)
				direct_lighting_nee = indirect_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-nee") == 0)
				direct_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-direct-bsdf") == 0)
				direct_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-nee") == 0)
				indirect_lighting_nee = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-indirect-bsdf") == 0)
				indirect_lighting_bsdf = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-visible-lights") == 0)
				visible_lights = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-use-vpls") == 0)
				use_vpls = atoi(argv[++i]) > 0;
			else if (strcmp(argv[i], "-light-tracing") == 0)
				light_tracing = (float)atof(argv[++i]);
		}
	}
};

///@} BPTLib
///@} Fermat
