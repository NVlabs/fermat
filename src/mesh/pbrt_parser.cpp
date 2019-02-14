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

#include <pbrt_parser.h>
#include <buffers.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace pbrt {

bool parse_string(FILE* file, char* out, char terminator = '\n')
{
	int status = 0;

	char* outp = out;

	while (1)
	{
		// fetch the next character from the input
		char c = fgetc(file);

		if (status == 0)
		{
			if (c == '"')
			{
				// string begin delimiter
				status = 1;
			}
			else if (c == terminator)
			{
				// no string found!
				return false;
			}
			else if (c != ' ') // eat whitespaces
			{
				char error_string[1024];
				fprintf(stderr,"%s\n", error_string);
				throw PBRTParserError(error_string);
			}
		}
		else if (status == 1)
		{
			if (c == '"')
			{
				// string end delimiter
				*outp = '\0';
				break;
			}
			else
			{
				// fetch a string character
				*outp++ = c;
			}
		}
	}
	return true;
}


bool parse_token(FILE* file, char& last, char* out, char terminator = '\n')
{
	int status = 0;

	while (1)
	{
		char c = last;

		if (status == 0)
		{
			if (c == terminator)
			{
				// no string found!
				return false;
			}
			else if (c != ' ' && c != '\n')
			{
				status = 1;
				*out++ = c;
			}
		}
		else if (status == 1)
		{
			if (c == terminator || c == ' '  || c == '\n')
			{
				// string end delimiter
				*out = '\0';
				break;
			}
			else
			{
				// fetch a string character
				*out++ = c;
			}
		}

		// fetch the next character from the input
		last = fgetc(file);
	}
	return true;
}

void parse_value(FILE* file, ValueType type, Value& value)
{
	value.set_type( type );

	// find the beginning of the parameter sequence, marked by '['
	while (1)
	{
		char c = fgetc(file);
		if (c == '[')
			break;
		else if (c != ' ' && c != '\n')
		{
			char error_string[1024];
			sprintf(error_string, "expected '[', found '%c'\n", c);
			throw PBRTParserError(error_string);
		}
	}

	if (type == STRING_TYPE)
	{
		char string[1024];
		while (parse_string(file, string, ']'))
			value.svec->push_back(std::string(string));
	}
	else if (type == BOOL_TYPE)
	{
		char string[1024];
		while (parse_string(file, string, ']'))
		{
			if (strcmp(string, "false") == 0)
				value.nvec->push_back(Number(0));
			else if (strcmp(string, "true") == 0)
				value.nvec->push_back(Number(1));
			else
			{
				char error_string[1024];
				sprintf(error_string, "expected true or false, found '%s'\n", string);
				throw PBRTParserError(error_string);
			}
		}
	}
	else if (
		type == FLOAT_TYPE ||
		type == POINT_TYPE ||
		type == NORMAL_TYPE ||
		type == VECTOR_TYPE ||
		type == RGB_TYPE ||
		type == XYZ_TYPE ||
		type == SPECTRUM_TYPE)
	{
		char string[1024];
		char c = fgetc(file);
		while (parse_token(file, c, string, ']'))
		{
			value.nvec->push_back(Number((float)atof(string)));
		}
	}
	else if (type == INT_TYPE)
	{
		char string[1024];
		char c = fgetc(file);
		while (parse_token(file, c, string, ']'))
		{
			value.nvec->push_back(Number(atoi(string)));
		}
	}
}

void parse_parameter_list(FILE* file, ParameterList& params)
{
	char type_name[1024];
	char type_string[1024];
	char name_string[1024];
	ValueType type;

	params.clear();

	while (parse_string(file, type_name, '\n'))
	{
		sscanf(type_name, "%s %s", type_string, name_string);
		//fprintf(stderr, "%s : %s\n", type_string, name_string);

		if (strcmp(type_string, "string") == 0)
			type = STRING_TYPE;
		else if (strcmp(type_string, "texture") == 0)
			type = STRING_TYPE;
		else if (strcmp(type_string, "float") == 0)
			type = FLOAT_TYPE;
		else if (strcmp(type_string, "integer") == 0)
			type = INT_TYPE;
		else if (strcmp(type_string, "bool") == 0)
			type = BOOL_TYPE;
		else if (strcmp(type_string, "point") == 0)
			type = POINT_TYPE;
		else if (strcmp(type_string, "normal") == 0)
			type = NORMAL_TYPE;
		else if (strcmp(type_string, "vector") == 0)
			type = VECTOR_TYPE;
		else if (strcmp(type_string, "rgb") == 0)
			type = RGB_TYPE;
		else if (strcmp(type_string, "xyz") == 0)
			type = XYZ_TYPE;
		else if (strcmp(type_string, "spectrum") == 0)
			type = SPECTRUM_TYPE;

		Value value;
		parse_value(file, type, value);

		params.names.push_back(std::string(name_string));
		params.values.push_back(value);
	}
}

void import(FILE* file, Importer* importer)
{
	char buf[4096];

	ParameterList params;

	while (fscanf(file, "%s", buf) != EOF)
	{
		if (strcmp(buf, "Integrator") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->integrator(name, params);
		}
		else if (strcmp(buf, "Identity") == 0)
		{
			importer->identity();
		}
		else if (strcmp(buf, "Transform") == 0)
		{
			Value value;
			parse_value(file, FLOAT_TYPE, value);
			importer->transform(value);
		}
		else if (strcmp(buf, "Rotate") == 0)
		{
			float angle, x, y, z;
			fscanf(file, "%f %f %f %f", &angle, &x, &y, &z);
			importer->rotate(angle, x, y, z);
		}
		else if (strcmp(buf, "Translate") == 0)
		{
			float x, y, z;
			fscanf(file, "%f %f %f", &x, &y, &z);
			importer->translate(x, y, z);
		}
		else if (strcmp(buf, "Scale") == 0)
		{
			float x, y, z;
			fscanf(file, "%f %f %f", &x, &y, &z);
			importer->scale(x, y, z);
		}
		else if (strcmp(buf, "LookAt") == 0)
		{
			float ex, ey, ez;
			float lx, ly, lz;
			float ux, uy, uz;
			fscanf(file, "%f %f %f", &ex, &ey, &ez);
			fscanf(file, "%f %f %f", &lx, &ly, &lz);
			fscanf(file, "%f %f %f", &ux, &uy, &uz);
			importer->look_at(ex, ey, ez, lx, ly, lz, ux, uy, uz);
		}
		else if (strcmp(buf, "Sampler") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->sampler(name, params);
		}
		else if (strcmp(buf, "PixelFilter") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->pixel_filter(name, params);
		}
		else if (strcmp(buf, "Film") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->film(name, params);
		}
		else if (strcmp(buf, "Camera") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->camera(name, params);
		}
		else if (strcmp(buf, "WorldBegin") == 0)
		{
			importer->world_begin();
		}
		else if (strcmp(buf, "WorldEnd") == 0)
		{
			importer->world_end();
		}
		else if (strcmp(buf, "AttributeBegin") == 0)
		{
			importer->attribute_begin();
		}
		else if (strcmp(buf, "AttributeEnd") == 0)
		{
			importer->attribute_end();
		}
		else if (strcmp(buf, "TransformBegin") == 0)
		{
			importer->transform_begin();
		}
		else if (strcmp(buf, "TransformEnd") == 0)
		{
			importer->transform_end();
		}
		else if (strcmp(buf, "Texture") == 0)
		{
			char name[1024];
			char type[1024];
			char source[1024];
			parse_string(file, name);
			parse_string(file, type);
			parse_string(file, source);
			parse_parameter_list(file, params);
			importer->texture(name, type, source, params);
		}
		else if (strcmp(buf, "MakeNamedMedium") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->make_named_medium(name, params);
		}
		else if (strcmp(buf, "MakeNamedMaterial") == 0)
		{
			char name[1024];
			parse_string(file, name);
			parse_parameter_list(file, params);
			importer->make_named_material(name, params);
		}
		else if (strcmp(buf, "NamedMaterial") == 0)
		{
			char name[1024];
			parse_string(file, name);
			importer->named_material(name);
		}
		else if (strcmp(buf, "MediumInterface") == 0)
		{
			char name1[1024];
			char name2[1024];
			parse_string(file, name1);
			parse_string(file, name2);
			importer->medium_interface(name1, name2);
		}
		else if (strcmp(buf, "Material") == 0)
		{
			char type[1024];
			parse_string(file, type);
			parse_parameter_list(file, params);
			importer->material(type, params);
		}
		else if (strcmp(buf, "AreaLightSource") == 0)
		{
			char type[1024];
			parse_string(file, type);
			parse_parameter_list(file, params);
			importer->area_light_source(type, params);
		}
		else if (strcmp(buf, "LightSource") == 0)
		{
			char type[1024];
			parse_string(file, type);
			parse_parameter_list(file, params);
			importer->light_source(type, params);
		}
		else if (strcmp(buf, "Shape") == 0)
		{
			char type[1024];
			parse_string(file, type);
			parse_parameter_list(file, params);
			importer->shape(type, params);
		}
		else
		{
			fgets(buf, 4096, file);
		}
	}
}

void import(const char* filename, Importer* importer)
{
	FILE* file = fopen(filename, "r");
	import(file, importer);
	fclose(file);
}

} // namespace pbrt
