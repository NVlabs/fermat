/*
 * Fermat
 *
 * Copyright (c) 2008-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "MeshLoader.h"

MeshLoader::MeshLoader(MeshStorage* mesh) :
  Base(),
  m_mesh(mesh),
  m_vertex_indices( 0 ),
  m_normal_indices( 0 ),
  m_color_indices( 0 ),
  m_texture_coordinate_indices( 0 )
{
	setVertexStride(mesh->m_vertex_stride);
	setNormalStride(mesh->m_normal_stride);
	setTextureCoordinateStride(mesh->m_texture_stride);
	setVertexIndexStride(MeshView::VERTEX_TRIANGLE_SIZE);
	setNormalIndexStride(MeshView::NORMAL_TRIANGLE_SIZE);
	setTextureIndexStride(MeshView::TEXTURE_TRIANGLE_SIZE);
}


MeshLoader::~MeshLoader()
{
}

// No pre-processing needed
void MeshLoader::preProcess() { }


// ----------------------------------------------------------------------------
// allocateData() and its utility functor
//
struct MeshLoader::AllocateGroupsFunctor
{
  MeshLoader& m_mesh;
  int	      m_group_index;
  int         m_triangles_index;

  AllocateGroupsFunctor( MeshLoader& mesh ) : 
	  m_mesh(mesh), m_group_index(0), m_triangles_index(0) {}

  // Assumes that m_mesh.vertex_indices, etc., have already been initialized
  void operator()( MeshGroup& group ) {
    group.vertex_indices			 = m_mesh.m_vertex_indices				+ m_triangles_index * MeshStorage::VERTEX_TRIANGLE_SIZE;
	group.normal_indices			 = m_mesh.m_normal_indices				+ m_triangles_index * MeshStorage::NORMAL_TRIANGLE_SIZE;
	group.texture_coordinate_indices = m_mesh.m_texture_coordinate_indices  + m_triangles_index * MeshStorage::TEXTURE_TRIANGLE_SIZE;
	group.material_indices           = m_mesh.m_material_indices			+ m_triangles_index;

	m_mesh.m_group_offsets[ m_group_index ] = m_triangles_index;
	m_mesh.setGroupName( m_group_index, group.name );

    m_triangles_index += group.num_triangles;
	m_group_index++;
  }
};

void MeshLoader::allocateData()
{
	m_mesh->alloc(
		getNumTriangles(),
		getNumVertices(),
		getNumNormals(),
		getNumTextureCoordinates(),
		getNumGroups() );

	m_vertex_indices		     = m_mesh->getVertexIndices();
	m_normal_indices			 = m_mesh->getNormalIndices();
	m_texture_coordinate_indices = m_mesh->getTextureCoordinateIndices();
	m_material_indices			 = m_mesh->getMaterialIndices();
	m_group_offsets				 = m_mesh->getGroupOffsets();
	m_color_indices              = 0;

	// Tell Base::loadData*() which fields to load and where
	setVertexData( m_mesh->getVertexData() );
	setNormalData( m_mesh->getNormalData() );
	setTextureCoordinateData( m_mesh->getTextureCoordinateData() );
	setMaterialIndices(m_material_indices);
	setColorData(0);

	forEachGroup( AllocateGroupsFunctor(*this) );

	// write the sentinel group offset
	m_group_offsets[ getNumGroups() ] = getNumTriangles();
}


// ----------------------------------------------------------------------------
// Other virtual overrides
//

// No special preparation needed; data pointers are already set by allocateData
void MeshLoader::startWritingData() { }

// No post-processing needed
void MeshLoader::postProcess() { }

struct MeshLoader::AssignMaterialsFunctor
{
	MeshLoader& m_mesh;
	int       m_triangles_index;

	AssignMaterialsFunctor(MeshLoader& mesh) :
		m_mesh(mesh)
	{ }

	// Assumes that m_mesh.vertex_indices, etc., have already been initialized
	void operator()(MeshGroup& group)
	{
		for (int i = 0; i < group.num_triangles; ++i)
		{
			//if (group.material_indices[i] != group.material_number)
			//{
			//	fprintf(stderr, "at group %s, tri %d, mat %d != %d\n", group.name.c_str(), i, group.material_indices[i], group.material_number);
			//	exit(1);
			//}
		}
	}
};

void MeshLoader::finishWritingData()
{
	forEachGroup(AssignMaterialsFunctor(*this));
}
