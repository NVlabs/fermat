/*
 * Copyright (c) 2019, NVIDIA Corporation
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

/*
    Very simple PFM reading/writing.
*/

#include <cugar/image/pfm.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

namespace cugar {

static inline int is_terminator(char c) {
    return c == ' ' || c == '\n' || c == '\t';
}
// reads a block of characters until a whitespace is found
static int read_block(FILE *fp, char *buffer, const int size)
{
	int n = 0;
	int c = fgetc(fp);
	while (c != EOF && !is_terminator(c) && n < size)
	{
		buffer[n++] = c;	// copy the last character read
		c = fgetc(fp);		// read a new character
	}

	if (n < size)
	{
		buffer[n] = '\0';
		return n;
	}
	return -1;
}

// Load an uncompressed PFM file; the pixel memory is allocated
// by the routine and must be freed by the caller using delete[].
float* load_pfm(const char *filename, uint32* xres, uint32* yres)
{
	FILE* file = fopen(filename, "rb");
	if (!file)
		return nullptr;

	float		scale = 1.0f;
	bool		is_little_endian;
	unsigned	num_channels;
	unsigned	num_floats;

	float*		raw_data = nullptr;
	float*		rgb_data = nullptr;

	char block[16];
	if (!read_block(file, block, sizeof(block)))
		goto error;


	if (strcmp(block, "Pf") == 0)		num_channels = 1;
	else if (strcmp(block, "PF") == 0)	num_channels = 3;
	else
		goto error;

	if (read_block(file, block, sizeof(block)) == -1)
		goto error;

	*xres = atoi(block);

	if (read_block(file, block, sizeof(block)) == -1)
		goto error;

	*yres = atoi(block);

	if (read_block(file, block, sizeof(block)) == -1)
		goto error;

	scale = (float)atof(block);

	num_floats = num_channels * *xres * *yres;

    // read the raw data
    raw_data = new float[num_floats];

	// read bottom up
    for (int y = *yres - 1; y >= 0; --y)
	{
        if (fread(&raw_data[y * *xres * num_channels], sizeof(float), *xres * num_channels, file) != *xres * num_channels)
            goto error;
    }

    // apply endian conversian and scale if appropriate
    is_little_endian = (scale < 0.f);
    if (!is_little_endian) // assume the host is little-endian - TODO: fix on big-endian machines!
	{
        for (unsigned int i = 0; i < num_floats; ++i)
		{
			uint8* bytes = reinterpret_cast<uint8*>(&raw_data[i]);
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
        }
    }

	// scale the data
    for (unsigned int i = 0; i < num_floats; ++i)
		raw_data[i] *= std::abs(scale);

	// check if no conversion is needed
	if (num_channels == 3)
	{
		fclose(file);
		return raw_data;
	}

	// convert to rgb if needed
	rgb_data = new float [ 3 * *xres * *yres ];
    for (unsigned int i = 0; i < num_floats; ++i)
	{
		rgb_data[ i*3 + 0 ] = raw_data[i];
		rgb_data[ i*3 + 1 ] = raw_data[i];
		rgb_data[ i*3 + 2 ] = raw_data[i];
	}
	return rgb_data;

error:
	fclose(file);
	delete [] raw_data;
	return nullptr;
}

} // namespace cugar
