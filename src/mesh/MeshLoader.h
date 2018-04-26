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

#pragma once

#include "MeshBase.h"
#include "MeshStorage.h"


class MeshLoader : public MeshBase
{
public:
  
  typedef MeshBase Base;

  MeshLoader(MeshStorage* mesh);

  virtual ~MeshLoader();

		int* getVertexIndices()			{ return m_vertex_indices; }
  const int* getVertexIndices() const	{ return m_vertex_indices; }
		int* getNormalIndices()			{ return m_normal_indices; }
  const int* getNormalIndices() const	{ return m_normal_indices; }
		int* getColorIndices()			{ return m_color_indices; }
  const int* getColorIndices() const	{ return m_color_indices; }
		int* getTextureCoordinateIndices()		 { return m_texture_coordinate_indices; }
  const int* getTextureCoordinateIndices() const { return m_texture_coordinate_indices; }

  void setGroupName(const uint32 i, const std::string& group_name) { m_mesh->m_group_names[i] = group_name;  }

protected:

  // Implementation of MeshBase's subclass interface
  virtual void preProcess();
  virtual void allocateData();
  virtual void startWritingData();
  virtual void postProcess();
  virtual void finishWritingData();

private:

  // A helper functor for allocateData()
  struct AllocateGroupsFunctor;
  struct AssignMaterialsFunctor;

  MeshStorage* m_mesh;
 
  int* m_vertex_indices;
  int* m_normal_indices;
  int* m_color_indices;
  int* m_texture_coordinate_indices;
  int* m_material_indices;
  int* m_group_offsets;
};
