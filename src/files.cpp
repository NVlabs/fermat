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

#include "files.h"
#include "types.h"
#include <stdio.h>
#include <string>

void extract_path(const char* filename, char* local_path)
{
	uint32 path_terminator = 0;

	// find the last occurrence of '/' or '\'
	for (int i = (int)strlen(filename); i >= 0; --i)
	{
		if (filename[i] == '/' || filename[i] == '\\')
		{
			path_terminator = uint32(i);
			break;
		}
	}

	strncpy(local_path, filename, path_terminator);
	local_path[path_terminator] = '\0';
}

bool find_file(char* filename, const uint32 n_dirs, char** dirs)
{
	for (uint32 i = 0; i < n_dirs; ++i)
	{
		std::string name = dirs[i];
		name.append("/");
		name.append(filename);

		ScopedFile file(name.c_str(), "r");
		if (file)
		{
			strcpy(filename, name.c_str());
			return true;
		}
	}
	return false;
}

bool find_file(char* filename, const std::vector<std::string>& dirs)
{
	for (size_t i = 0; i < dirs.size(); ++i)
	{
		std::string name = dirs[i];
		name.append("/");
		name.append(filename);

		ScopedFile file(name.c_str(), "r");
		if (file)
		{
			strcpy(filename, name.c_str());
			return true;
		}
	}
	return false;
}
