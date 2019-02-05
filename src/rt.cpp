/*
 * Fermat
 *
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <rt.h>
#include <types.h>
#include <MeshView.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixpp_namespace.h>
#include <string>
#include <vector>
#include <map>

const char* g_trbvh = "Trbvh";

enum EProgram
{
    PROGRAM_CLOSEST_HIT				= 0,
    PROGRAM_ANY_HIT					= 1,
    PROGRAM_MISS					= 2,
    PROGRAM_SHADOW_CLOSEST_HIT		= 3,
    PROGRAM_SHADOW_ANY_HIT			= 4,
    PROGRAM_SHADOW_MISS				= 5,
	PROGRAM_MAX_COUNT				= 6
};

struct RTMesh
{
	optix::Buffer index_buffer;
	optix::Buffer normal_index_buffer;
	optix::Buffer tex_index_buffer;

	optix::Buffer vertex_buffer;
	optix::Buffer normal_buffer;
	optix::Buffer texture_buffer;

	optix::Buffer material_index_buffer;
};

struct RTContextImpl
{
	void create_context();
	void create_geometry(
		unsigned int	tri_count,
		const int*		index_ptr,
		unsigned int	vertex_count,
		const float*	vertex_ptr,
		const int*		normal_index_ptr,
		const float*	normal_vertex_ptr,
		const int*		tex_index_ptr,
		const float*	tex_vertex_ptr,
		const int*		material_index_ptr);

	optix::Program create_optix_program(
		const char*			filename,
		const std::string&	program_name);

	uint32 RTContextImpl::create_program(
		const char*		filename,
		const char*		program_name);

	uint32 add_ray_generation_program(const uint32 program);

	void update();

	optix::Context				context;
	optix::GeometryTriangles	geometry_triangles;
	optix::GeometryInstance		geometry_instance;
	optix::Material				material;
	optix::Program				programs[EProgram::PROGRAM_MAX_COUNT];

	std::map<uint32,optix::Program> user_programs;
	uint32							user_programs_count;

	std::vector<uint32>			ray_gen_programs;
	bool						ray_gen_programs_dirty;

	uint32						null_ray_gen_idx;
	uint32						tmin_intersection_ray_gen_idx;
	uint32						masked_intersection_ray_gen_idx;
	uint32						masked_shadow_intersection_ray_gen_idx;
	uint32						masked_shadow_binary_intersection_ray_gen_idx;

	std::vector<optix::Buffer>	user_buffers;

	optix::Buffer				ray_buffer;
	optix::Buffer				hit_buffer;

	RTMesh						mesh;
};

optix::Buffer create_optix_buffer(
	optix::Context	context,
	const RTformat	format)
{
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
	buffer->setFormat(format);
	buffer->setSize(0);
	return buffer;
}

template <typename T>
optix::Buffer create_optix_buffer(
	optix::Context	context,
	const uint32	size,
	const T*		ptr,
	const RTformat	format)
{
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
	buffer->setFormat(format);
	//buffer->setElementSize(get_elem_byte_size(format));
	buffer->setSize(size);
	if (ptr)
		buffer->setDevicePointer(0u,(void*)ptr);
	return buffer;
}

optix::Buffer create_optix_buffer(
	optix::Context	context,
	const uint32	size,
	const uint32	element_size,
	void*			ptr,
	const RTformat	format)
{
	try
	{
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
		buffer->setFormat(format);
		buffer->setElementSize(element_size);
		buffer->setSize(size);
		if (ptr)
			buffer->setDevicePointer(0u, ptr);
		return buffer;
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
	catch (...)
	{
		fprintf(stderr, "  unknown exception in OptiX\n");
		exit(1);
	}
}

template <typename T>
void update_optix_buffer(
	optix::Buffer	buffer,
	const uint32	size,
	const T*		ptr)
{
	buffer->setSize(size);
	buffer->setDevicePointer(0u,(void*)ptr);
}

void RTContextImpl::create_context()
{
	try
	{
		context = optix::Context::create();
		context->setRayTypeCount( 3 );
		context->setEntryPointCount( 1 );

		//context->setPrintEnabled(true);
		//context->setPrintBufferSize( 8192 );

		user_programs_count = 0;

		ray_gen_programs_dirty = false;

        optix::Program closest_hit	= programs[EProgram::PROGRAM_CLOSEST_HIT   ] = create_optix_program("optix_material.cu", "program_closest_hit" );
        optix::Program any_hit		= programs[EProgram::PROGRAM_ANY_HIT       ] = create_optix_program("optix_material.cu", "program_any_hit" );
        optix::Program miss			= programs[EProgram::PROGRAM_MISS          ] = create_optix_program("optix_material.cu", "program_miss" );

        programs[EProgram::PROGRAM_SHADOW_CLOSEST_HIT   ] = create_optix_program("optix_shadow_material.cu", "program_closest_hit" );
        programs[EProgram::PROGRAM_SHADOW_ANY_HIT       ] = create_optix_program("optix_shadow_material.cu", "program_any_hit" );
        programs[EProgram::PROGRAM_SHADOW_MISS          ] = create_optix_program("optix_shadow_material.cu", "program_miss" );

		material = context->createMaterial();
		material->setClosestHitProgram(0, programs[EProgram::PROGRAM_CLOSEST_HIT]);
		material->setAnyHitProgram(0, programs[EProgram::PROGRAM_ANY_HIT]);

		material->setClosestHitProgram(1, programs[EProgram::PROGRAM_SHADOW_CLOSEST_HIT]);
		material->setAnyHitProgram(1, programs[EProgram::PROGRAM_SHADOW_ANY_HIT]);

		material->setClosestHitProgram(2, programs[EProgram::PROGRAM_CLOSEST_HIT]);	// no masking

		mesh.index_buffer        = create_optix_buffer(context, RT_FORMAT_INT4);
		//mesh.normal_index_buffer = create_optix_buffer(context, RT_FORMAT_INT3);
		//mesh.tex_index_buffer    = create_optix_buffer(context, RT_FORMAT_INT3);

		mesh.vertex_buffer  = create_optix_buffer(context, sizeof(MeshView::vertex_type) == 16 ? RT_FORMAT_FLOAT4 : RT_FORMAT_FLOAT3);
		//mesh.normal_buffer  = create_optix_buffer(context, RT_FORMAT_FLOAT3);
		//mesh.texture_buffer = create_optix_buffer(context, RT_FORMAT_FLOAT2);

		//mesh.material_index_buffer = create_optix_buffer(context, RT_FORMAT_INT);

#if 1
		// for now make everything context-visible as we do not have instances... might want to revisit in the future
		context["g_index_buffer" ]->setBuffer( mesh.index_buffer );
		//context["g_normal_index_buffer" ]->setBuffer( mesh.normal_index_buffer );
		//context["g_texture_index_buffer" ]->setBuffer( mesh.tex_index_buffer );

		context["g_vertex_buffer"]->setBuffer( mesh.vertex_buffer );
		//context["g_normal_buffer"]->setBuffer( mesh.normal_buffer );
		//context["g_texture_buffer"]->setBuffer( mesh.texture_buffer );

		//context["g_material_index_buffer"]->setBuffer( mesh.material_index_buffer );
#else
		closest_hit["g_index_buffer" ]->setBuffer( mesh.index_buffer );
		closest_hit["g_normal_index_buffer" ]->setBuffer( mesh.normal_index_buffer );
		closest_hit["g_texture_index_buffer" ]->setBuffer( mesh.tex_index_buffer );

		closest_hit["g_vertex_buffer"]->setBuffer( mesh.vertex_buffer );
		closest_hit["g_normal_buffer"]->setBuffer( mesh.normal_buffer );
		closest_hit["g_texture_buffer"]->setBuffer( mesh.texture_buffer );

		any_hit["g_index_buffer" ]->setBuffer( mesh.index_buffer );
		any_hit["g_normal_index_buffer" ]->setBuffer( mesh.normal_index_buffer );
		any_hit["g_texture_index_buffer" ]->setBuffer( mesh.tex_index_buffer );

		any_hit["g_vertex_buffer"]->setBuffer( mesh.vertex_buffer );
		any_hit["g_normal_buffer"]->setBuffer( mesh.normal_buffer );
		any_hit["g_texture_buffer"]->setBuffer( mesh.texture_buffer );
#endif

		// NOTE: these must be the very first 3 ray gen programs added, in this order, in order to match the ERayGenPrograms enum
		{
			const uint32 null_ray_gen_id = create_program("null_ray_gen.cu", "program_ray_generation" );
			const uint32 tmin_ray_gen_id = create_program("optix_rt.cu", "program_tmin_ray_generation" );
			const uint32 masked_ray_gen_id = create_program("optix_rt.cu", "program_masked_ray_generation" );
			const uint32 masked_shadow_ray_gen_id = create_program("optix_rt.cu", "program_masked_shadow_ray_generation" );
			const uint32 masked_shadow_binary_ray_gen_id = create_program("optix_rt.cu", "program_masked_shadow_binary_ray_generation" );

			null_ray_gen_idx                = add_ray_generation_program( null_ray_gen_id );
			tmin_intersection_ray_gen_idx   = add_ray_generation_program( tmin_ray_gen_id );
			masked_intersection_ray_gen_idx = add_ray_generation_program( masked_ray_gen_id );
			masked_shadow_intersection_ray_gen_idx = add_ray_generation_program( masked_shadow_ray_gen_id );
			masked_shadow_binary_intersection_ray_gen_idx = add_ray_generation_program( masked_shadow_binary_ray_gen_id );
		}

		// setup some reusable buffers
		ray_buffer = create_optix_buffer( context, RT_FORMAT_FLOAT4 );
		hit_buffer = create_optix_buffer( context, RT_FORMAT_FLOAT4 );

		context["g_ray_buffer" ]->setBuffer( ray_buffer );
		context["g_hit_buffer" ]->setBuffer( hit_buffer );

		uint32* null_ptr = NULL;
		context["g_binary_hits"]->setUserData( sizeof(uint32*), &null_ptr );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

void RTContextImpl::create_geometry(
	const uint32	tri_count,
	const int*		index_ptr,
	const uint32	vertex_count,
	const float*	vertex_ptr,
	const int*		normal_index_ptr,
	const float*	normal_vertex_ptr,
	const int*		tex_index_ptr,
	const float*	tex_vertex_ptr,
	const int*		material_index_ptr)
{
	try
	{
		update_optix_buffer(mesh.index_buffer, tri_count, index_ptr);
		//update_optix_buffer(mesh.normal_index_buffer, tri_count, normal_index_ptr);
		//update_optix_buffer(mesh.tex_index_buffer, tri_count, tex_index_ptr);

		update_optix_buffer(mesh.vertex_buffer, vertex_count, vertex_ptr);
		//update_optix_buffer(mesh.normal_buffer, vertex_count, normal_vertex_ptr);
		//update_optix_buffer(mesh.texture_buffer, vertex_count, tex_vertex_ptr);

		//update_optix_buffer(mesh.material_index_buffer, tri_count, material_index_ptr);

		geometry_triangles = context->createGeometryTriangles();
		geometry_triangles->setPrimitiveCount(tri_count);
		geometry_triangles->setTriangleIndices( mesh.index_buffer, 0, mesh.index_buffer->getElementSize(), RT_FORMAT_UNSIGNED_INT3 );
		geometry_triangles->setVertices( vertex_count, mesh.vertex_buffer, 0, mesh.vertex_buffer->getElementSize(), RT_FORMAT_FLOAT3 );
		geometry_triangles->setBuildFlags( RT_GEOMETRY_BUILD_FLAG_NONE );

		geometry_instance = context->createGeometryInstance();
		geometry_instance->setGeometryTriangles( geometry_triangles );
		geometry_instance->setMaterialCount(1);
		geometry_instance->setMaterial(0, material);
		geometry_instance["model_id"]->setUint(0);

		optix::GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setChildCount(1);
		geometry_group->setChild(0, geometry_instance);
		geometry_group->setAcceleration(context->createAcceleration(g_trbvh));

		context["g_top_object"]->set( geometry_group );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

void RTContextImpl::update()
{
	try
	{
		if (ray_gen_programs_dirty)
		{
			context->setEntryPointCount( uint32(ray_gen_programs.size()) );

			for (uint32 i = 0; i < (uint32)ray_gen_programs.size(); ++i)
			{
				const uint32 program_id = ray_gen_programs[i];

				optix::Program optix_program = user_programs[program_id];

				context->setRayGenerationProgram( i, optix_program );
			}

			ray_gen_programs_dirty = false;
		}

		context->validate();
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

std::string ptxPath(const std::string& cuda_file)
{
    return 
        std::string("./ptx/")
        + cuda_file
        + ".ptx";
}

optix::Program RTContextImpl::create_optix_program(
	const char*			filename,
	const std::string&	program_name)
{
    std::string ptx_path( ptxPath( filename ) );
    return context->createProgramFromPTXFile( ptx_path, program_name );
}

uint32 RTContextImpl::create_program(
	const char*		filename,
	const char*		program_name)
{
	try
	{
		optix::Program optix_program = create_optix_program( filename, program_name );

		user_programs.insert( std::make_pair( user_programs_count, optix_program ) );

		return user_programs_count++;
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

uint32 RTContextImpl::add_ray_generation_program(const uint32 program)
{
	ray_gen_programs.push_back( program );
	ray_gen_programs_dirty = true;
	return uint32( ray_gen_programs.size() - 1u );
}

RTContext::RTContext() : impl(NULL)
{
	impl = new RTContextImpl();
	impl->create_context();
}

RTContext::~RTContext()
{
	delete impl;
}

void RTContext::create_geometry(
	unsigned int	tri_count,
	const int*		index_ptr,
	unsigned int	vertex_count,
	const float*	vertex_ptr,
	const int*		normal_index_ptr,
	const float*	normal_vertex_ptr,
	const int*		tex_index_ptr,
	const float*	tex_vertex_ptr,
	const int*		material_index_ptr)
{
	impl->create_geometry(
		tri_count,
		index_ptr,
		vertex_count,
		vertex_ptr,
		normal_index_ptr,
		normal_vertex_ptr,
		tex_index_ptr,
		tex_vertex_ptr,
		material_index_ptr);
}

void RTContext::bind_buffer(
	const char*		name,
	const uint32	size,
	const uint32	element_size,
	void*			ptr,
	const RTformat	format)
{
	try
	{
		optix::Buffer buffer = impl->context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
		buffer->setFormat(format);
		buffer->setElementSize(element_size);
		buffer->setSize(size);
		buffer->setDevicePointer(0u, ptr);

		impl->user_buffers.push_back( buffer );

		impl->context[name]->setBuffer( buffer );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

void RTContext::bind_var(
	const char*		name,
	const uint32	size,
	void*			ptr)
{
	try
	{
		impl->context[name]->setUserData( size, ptr );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::bind_var(const char* name, const int32 value)
{
	try
	{
		impl->context[name]->setInt( value );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::bind_var(const char* name, const uint32 value)
{
	try
	{
		impl->context[name]->setUint( value );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::bind_var(const char* name, const float value)
{
	try
	{
		impl->context[name]->setFloat( value );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

uint32 RTContext::create_program(
	const char*		filename,
	const char*		program_name)
{
	return impl->create_program( filename, program_name );
}

uint32 RTContext::add_ray_generation_program(const uint32 program)
{
	return impl->add_ray_generation_program( program );
}

void RTContext::launch(const uint32 index, const uint32 width)
{
	try
	{
		impl->update();

		impl->context->launch( index, width );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

void RTContext::launch(const uint32 index, const uint32 width, const uint32 height)
{
	try
	{
		impl->update();

		impl->context->launch( index, width, height );
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}

void RTContext::trace(const uint32 count, const Ray* rays, Hit* hits)
{
	try
	{
		impl->update();

		// setup the ray buffer
		impl->ray_buffer->setSize(count*2);
		impl->ray_buffer->setDevicePointer(0u, (void*)rays);

		// setup the hit buffer
		impl->hit_buffer->setSize(count);
		impl->hit_buffer->setDevicePointer(0u, (void*)hits);

		impl->context->launch( impl->tmin_intersection_ray_gen_idx, count );

		// release the buffers
		impl->ray_buffer->setSize(0u);
		impl->hit_buffer->setSize(0u);
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::trace(const uint32 count, const MaskedRay* rays, Hit* hits)
{
	try
	{
		impl->update();

		// setup the ray buffer
		impl->ray_buffer->setSize(count*2);
		impl->ray_buffer->setDevicePointer(0u, (void*)rays);

		// setup the hit buffer
		impl->hit_buffer->setSize(count);
		impl->hit_buffer->setDevicePointer(0u, (void*)hits);

		impl->context->launch( impl->tmin_intersection_ray_gen_idx, count );

		// release the buffers
		impl->ray_buffer->setSize(0u);
		impl->hit_buffer->setSize(0u);
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::trace_shadow(const uint32 count, const MaskedRay* rays, Hit* hits)
{
	try
	{
		impl->update();

		// setup the ray buffer
		impl->ray_buffer->setSize(count*2);
		impl->ray_buffer->setDevicePointer(0u, (void*)rays);

		// setup the hit buffer
		impl->hit_buffer->setSize(count);
		impl->hit_buffer->setDevicePointer(0u, (void*)hits);

		impl->context->launch( impl->masked_shadow_intersection_ray_gen_idx, count );

		// release the buffers
		impl->ray_buffer->setSize(0u);
		impl->hit_buffer->setSize(0u);
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
void RTContext::trace_shadow(const uint32 count, const MaskedRay* rays, uint32* binary_hits)
{
	try
	{
		impl->update();

		// setup the ray buffer
		impl->ray_buffer->setSize(count*2);
		impl->ray_buffer->setDevicePointer(0u, (void*)rays);

		// setup the hit buffer
		impl->context["g_binary_hits"]->setUserData( sizeof(uint32*), &binary_hits );

		impl->context->launch( impl->masked_shadow_binary_intersection_ray_gen_idx, count );

		// release the buffers
		impl->ray_buffer->setSize(0u);
	}
	catch (optix::Exception& e)
	{
		fprintf(stderr, "  error[%d] : %s\n", e.getErrorCode(), e.getErrorString().c_str());
		exit(1);
	}
}
