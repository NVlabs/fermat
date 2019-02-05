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

#include <buffers.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace pbrt {

class PBRTParserError : public std::runtime_error
{
public:
	explicit PBRTParserError( const std::string& what_arg )
		: std::runtime_error( what_arg )
	{ }
};

struct Number
{
	union {
		float	f;
		int		i;
	};

	Number(const float v) : f(v) {}
	Number(const int v) : i(v) {}
};

typedef std::vector<Number>			NumVec;
typedef std::vector<std::string>	StringVec;

enum ValueType {
	NULL_TYPE		= 0,
	STRING_TYPE		= 1,
	INT_TYPE		= 2,
	BOOL_TYPE		= 3,
	FLOAT_TYPE		= 4,
	POINT_TYPE		= 5,
	NORMAL_TYPE		= 6,
	VECTOR_TYPE		= 7,
	RGB_TYPE		= 8,
	XYZ_TYPE		= 9,
	SPECTRUM_TYPE	= 10,
};

struct Value
{
	uint32 type;
	union {
		NumVec		*nvec;
		StringVec	*svec;
	};

	Value() : type(NULL_TYPE) {}
	~Value()
	{
		if (type == STRING_TYPE)
			delete svec;
		else if (type != NULL_TYPE)
			delete nvec;
	}

	Value(const Value& other)
	{
		type = other.type;
		switch (other.type)
		{
		case NULL_TYPE:
			break;
		case STRING_TYPE:
			svec = new StringVec(*other.svec);
			break;
		default:
			nvec = new NumVec(*other.nvec);
			break;
		}
	}

	size_t size() const
	{
		return (type == STRING_TYPE) ? svec->size() : nvec->size();
	}
	const float* get_floats() const
	{
		return reinterpret_cast<const float*>(&(*nvec)[0].f);
	}
	const int* get_ints() const
	{
		return reinterpret_cast<const int*>(&(*nvec)[0].i);
	}
	const std::string& get_string(const size_t i) const
	{
		return (*svec)[i];
	}
	float get_float(const size_t i) const
	{
		return (*nvec)[i].f;
	}
	int get_int(const size_t i) const
	{
		return (*nvec)[i].i;
	}
	bool get_bool(const size_t i) const
	{
		return (*nvec)[i].i ? true : false;
	}

	void clear()
	{
		if (type == STRING_TYPE)
			delete svec;
		else if (type != NULL_TYPE)
			delete nvec;

		type = NULL_TYPE;
	}

	void set_type(const ValueType _type)
	{
		clear();

		type = _type;
		if (type == STRING_TYPE)
			svec = new StringVec();
		else if (type != NULL_TYPE)
			nvec = new NumVec();
	}

	const char* type_string() const
	{
		switch (type)
		{
		case STRING_TYPE:
			return "string";
		case INT_TYPE:
			return "integer";
		case BOOL_TYPE:
			return "bool";
		case FLOAT_TYPE:
			return "float";
		case POINT_TYPE:
			return "point";
		case NORMAL_TYPE:
			return "normal";
		case VECTOR_TYPE:
			return "vector";
		case RGB_TYPE:
			return "rgb";
		case XYZ_TYPE:
			return "xyz";
		case SPECTRUM_TYPE:
			return "spectrum";
		}
		return "null";
	}
};

struct ParameterList
{
	std::vector<std::string>	names;
	std::vector<Value>			values;

	void clear()
	{
		names.erase(names.begin(), names.end());
		values.erase(values.begin(), values.end());
	}
};

struct Importer
{
	virtual void identity() {}
	virtual void transform(const Value& floats) {}
	virtual void rotate(const float angle, const float x, const float y, const float z) {}
	virtual void scale(const float x, const float y, const float z) {}
	virtual void translate(const float x, const float y, const float z) {}
	virtual void look_at(
		const float ex, const float ey, const float ez,
		const float lx, const float ly, const float lz,
		const float ux, const float uy, const float uz) {}

	virtual void integrator(const char* name, const ParameterList& params) {}
	virtual void sampler(const char* name, const ParameterList& params) {}
	virtual void pixel_filter(const char* name, const ParameterList& params) {}
	virtual void film(const char* name, const ParameterList& params) {}
	virtual void camera(const char* name, const ParameterList& params) {}

	virtual void world_begin() {}
	virtual void world_end() {}

	virtual void attribute_begin() {}
	virtual void attribute_end() {}

	virtual void transform_begin() {}
	virtual void transform_end() {}

	virtual void texture(const char* name, const char* texel_type, const char* texture_type, const ParameterList& params) {}
	virtual void make_named_medium(const char* name, const ParameterList& params) {}
	virtual void make_named_material(const char* name, const ParameterList& params) {}
	virtual void named_material(const char* name) {}
	virtual void medium_interface(const char* name1, const char* name2) {}
	virtual void material(const char* name, const ParameterList& params) {}
	virtual void area_light_source(const char* type, const ParameterList& params) {}

	virtual void shape(const char* type, const ParameterList& params) {}
};

struct EchoImporter : public Importer
{
	EchoImporter(FILE* _file) : stack_depth(0), file(_file) {};

	void indent()
	{
		for (unsigned i = 0; i < stack_depth; ++i)
			fprintf(file, "\t");
	}

	void print_value(const Value& value)
	{
		fprintf(file, "[");
		if (value.type == STRING_TYPE)
		{
			for (size_t i = 0; i < value.size(); ++i)
				fprintf(file, " \"%s\"", value.get_string(i).c_str());
		}
		else if (
			value.type == FLOAT_TYPE ||
			value.type == POINT_TYPE ||
			value.type == NORMAL_TYPE ||
			value.type == VECTOR_TYPE ||
			value.type == RGB_TYPE ||
			value.type == XYZ_TYPE ||
			value.type == SPECTRUM_TYPE)
		{
			for (size_t i = 0; i < value.size(); ++i)
				fprintf(file, " %f", value.get_float(i));
		}
		else if (value.type == INT_TYPE)
		{
			for (size_t i = 0; i < value.size(); ++i)
				fprintf(file, " %i", value.get_int(i));
		}
		else if (value.type == BOOL_TYPE)
		{
			for (size_t i = 0; i < value.size(); ++i)
				fprintf(file, " %s", value.get_bool(i) ? "\"true\"" : "\"false\"");
		}
		fprintf(file, " ]");
	}
	void print_params(const ParameterList& params)
	{
		for (size_t i = 0; i < params.names.size(); ++i)
		{
			fprintf(file, " \"%s %s\" ", params.values[i].type_string(), params.names[i].c_str());
			print_value(params.values[i]);
		}
	}

	virtual void identity() { indent(); fprintf(file,"Identity\n"); }
	virtual void transform(const Value& floats) { indent(); fprintf(file,"Transform "); print_value(floats); fprintf(file,"\n"); }
	virtual void rotate(const float angle, const float x, const float y, const float z) { indent(); fprintf(file, "Rotate %f %f %f %f\n", angle, x, y, z); }
	virtual void scale(const float x, const float y, const float z) { indent(); fprintf(file, "Scale %f %f %f\n", x, y, z); }
	virtual void translate(const float x, const float y, const float z) { indent(); fprintf(file, "Translate %f %f %f\n", x, y, z); }
	virtual void look_at(
		const float ex, const float ey, const float ez,
		const float lx, const float ly, const float lz,
		const float ux, const float uy, const float uz)
	{
		fprintf(file, "LookAt %f %f %f %f %f %f %f %f %f\n", ex, ey, ez, lx, ly, lz, ux, uy, uz);
	}

	virtual void integrator(const char* name, const ParameterList& params) { indent(); fprintf(file,"Integrator \"%s\" ", name); print_params(params); fprintf(file,"\n"); }
	virtual void sampler(const char* name, const ParameterList& params) { indent(); fprintf(file,"Sampler \"%s\" ", name); print_params(params); fprintf(file,"\n"); }
	virtual void pixel_filter(const char* name, const ParameterList& params) { indent(); fprintf(file,"PixelFilter \"%s\" ", name); print_params(params); fprintf(file,"\n"); }
	virtual void film(const char* name, const ParameterList& params) { indent(); fprintf(file,"Film \"%s\" ", name); print_params(params); fprintf(file,"\n"); }
	virtual void camera(const char* name, const ParameterList& params) { indent(); fprintf(file,"Camera \"%s\" ", name); print_params(params); fprintf(file,"\n"); }

	virtual void world_begin() { indent(); fprintf(file,"WorldBegin\n"); ++stack_depth; }
	virtual void world_end() { --stack_depth; indent(); fprintf(file,"WorldEnd\n"); }

	virtual void attribute_begin() { indent(); fprintf(file,"AttributeBegin\n"); ++stack_depth; }
	virtual void attribute_end() { --stack_depth; indent(); fprintf(file,"AttributeEnd\n"); }

	virtual void transform_begin() { indent(); fprintf(file,"TransformBegin\n"); ++stack_depth; }
	virtual void transform_end() { --stack_depth; indent(); fprintf(file,"TransformEnd\n"); }

	virtual void texture(const char* name, const char* texel_type, const char* texture_type, const ParameterList& params)
	{
		indent();
		fprintf(file,"Texture \"%s\" \"%s\" \"%s\" ", name, texel_type, texture_type);
		print_params(params);
		fprintf(file,"\n");
	}
	virtual void make_named_medium(const char* name, const ParameterList& params) { indent(); fprintf(file, "MakeNamedMedium \"%s\" ", name); print_params(params); fprintf(file,"\n"); }
	virtual void make_named_material(const char* name, const ParameterList& params) { indent(); fprintf(file, "MakeNamedMaterial \"%s\"", name); print_params(params); fprintf(file,"\n"); }
	virtual void named_material(const char* name) { indent(); fprintf(file, "NamedMaterial \"%s\"\n", name); }
	virtual void medium_interface(const char* name1, const char* name2) { indent(); fprintf(file, "MediumInterface \"%s\" \"%s\"\n", name1, name2); }
	virtual void material(const char* type, const ParameterList& params) { indent(); fprintf(file, "Material \"%s\" ", type); print_params(params); fprintf(file,"\n"); }
	virtual void area_light_source(const char* type, const ParameterList& params) { indent(); fprintf(file, "AreaLightSource \"%s\" ", type); print_params(params); fprintf(file,"\n"); }

	virtual void shape(const char* type, const ParameterList& params) { indent(); fprintf(file, "Shape \"%s\" ", type); print_params(params); fprintf(file,"\n"); }

	unsigned stack_depth;
	FILE* file;
};

void import(const char* filename, Importer* importer);

} // namespace pbrt
