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

#include "MeshException.h"

#include <string>
#include <map>
#include <vector>

static const int MESH_ATTRIBUTE_NOT_PROVIDED = -1;


/* --------------------------------------------------------------------------------  
 * Materials
 */

// TODO: Revise these in an sutil-centric way, rather than a .obj-centric way
/* Mimics the shading type enum found in GLM for obj shading types */
enum {
  MESH_SHADING_NONE           = (0),
  MESH_SHADING_FLAT           = (1 << 0),
  MESH_SHADING_SMOOTH         = (1 << 1),
  MESH_SHADING_TEXTURE        = (1 << 2),
  MESH_SHADING_COLOR          = (1 << 3),
  MESH_SHADING_MATERIAL       = (1 << 4),
  MESH_SHADING_FLAT_SHADE     = (1 << 5),
  MESH_SHADING_SPECULAR_SHADE = (1 << 6)
};

class SUTILCLASSAPI MeshTextureMap {
public:

  SUTILAPI MeshTextureMap() { }
  SUTILAPI ~MeshTextureMap() { } // So the CRT deletes it on the correct heap

  std::string name;
  float       scaling[2];
};

class SUTILCLASSAPI MeshMaterialParams {
public:

  SUTILAPI MeshMaterialParams() { setToDefaultParams(); }
  SUTILAPI ~MeshMaterialParams() { } // To ensure it's del'ed on the correct heap

  SUTILAPI void setToDefaultParams();

  std::string name;

  float diffuse[4];
  float diffuse_trans[4];
  float ambient[4];
  float specular[4];
  float emissive[4];
  float phong_exponent;
  float index_of_refraction;
  float opacity;
  float reflectivity[4];
  
  int   flags;
  int   shading_type;

  /* Texture maps */
  MeshTextureMap ambient_map;
  MeshTextureMap diffuse_map;
  MeshTextureMap diffuse_trans_map;
  MeshTextureMap specular_map;
  MeshTextureMap emissive_map;
  MeshTextureMap opacity_map;
  MeshTextureMap bump_map;
};

typedef std::map<std::string, int> MeshMaterialNumbersMap;


/* --------------------------------------------------------------------------------
 * Groups
 */
class SUTILCLASSAPI MeshGroup {
public:

  SUTILAPI MeshGroup() :
    num_triangles( 0 ),
    vertex_indices( 0 ),
    normal_indices( 0 ),
    color_indices( 0 ),
    texture_coordinate_indices( 0 ),
	material_indices( 0 ),
    material_number( 0 )
  { }

  SUTILAPI ~MeshGroup() { } // So the CRT deletes it on the correct heap

  std::string name;

  int num_triangles;
  
  int* vertex_indices;
  int* normal_indices;
  int* color_indices;
  int* texture_coordinate_indices;
  int* material_indices;

  int material_number;
};

typedef std::map<std::string, MeshGroup> MeshGroupMap;

enum MeshGrouping {
    kMergeGroups   = 0,
    kKeepGroups    = 1,
    kKeepSubgroups = 2,
};

/* --------------------------------------------------------------------------------
 * MeshBase abstract class.
 *
 * Implements common functionality for concrete Mesh classes, and provides
 * interface for methods that all Meshes should implement.
 */
class MeshBase {
public:

  MeshBase();

  // This destructor handles no de-allocation of itself, since resource
  // allocation is delegated to subclasses
  virtual ~MeshBase();

  /**
   * The load() method provides the overall model loading algorithm, which
   * calls upon pure virtual functions which must override.  The algorithm is
   * as follows, where 'function()' indicates a call to one of the pure virtual
   * functions:
   *
   * - preProcess()
   *
   * - Does a first pass on the model file to collect information about the number of
   *   vertices, normals, colors, and texture coordinates in the file, which it
   *   stores in the mesh's fields 'num_vertices', 'num_normals', etc.  The first
   *   pass also creates the 'groups' structure with 'num_triangles' set for each
   *   group'.
   *
   * - allocateData(), which allocates arrays, buffers, or whatever may be
   *   required for the subclass to own the data; the numbers collected during
   *   the "first pass" can be used for allocation sizes.
   *
   * - startWritingData(), which is useful for cases where the subclass must
   *   first prepare for being written to, such as mapping Buffers, in the case
   *   of OptiX buffers.
   *
   * - The second pass, in which it actually loads the data from the model file
   *   and stores it in the array pointers vertex_data, normal_data,
   *   color_data, and texture_coordinate_data, which should have been properly
   *   initalized to receive writes, either during allocateData() or
   *   startWritingData().  If any of these *_data pointers are '0', they will
   *   not be written to.  Each group in the groups structure should also have
   *   had its field pointers vertex_indices, normal_indices, etc.
   *   initialized.
   *
   * - Computes the bounding box on the model.
   *
   * - postProcess(),
   *
   * - finishWritingData(), for any cleanup that must be done, such as
   *   unmapping OptiX buffers.
   *
   * Note that for uniformity with .ply loading, and for the accuracy of
   * external algorithms that iterate over vertices (such as bounding box
   * computation), or other alforithms that require an accurate count of
   * the number of vertices actually used, 1-based indexing in .obj files
   * is translated to 0-based indexing.
   */
  virtual void loadModel( const std::string& filename, bool insertDefaultMaterial = true, const MeshMaterialParams& defaultMaterial = MeshMaterialParams() );

  /**
   * Calls functor( group ) for each group in the mesh, where 'group' is of
   * the type MeshGroup.
   */
  template <class Functor>
  void forEachGroup( Functor functor ) const;

  template <class Functor>
  void forEachGroup( Functor functor );

  int getNumVertices() const { return m_num_vertices; }
  int getNumNormals() const { return m_num_normals; }
  int getNumColors() const { return m_num_colors; }
  int getNumTextureCoordinates() const { return m_num_texture_coordinates; }

  int getNumTriangles() const { return m_num_triangles; }
  int getNumGroups() const { return (int)m_mesh_groups.size(); }

  float* getVertexData() { return m_vertex_data; }
  const float* getVertexData() const { return m_vertex_data; }

  float* getNormalData() { return m_normal_data; }
  const float* getNormalData() const { return m_normal_data; }
  
  unsigned char* getColorData() { return m_color_data; }
  const unsigned char* getColorData() const { return m_color_data; }
  
  float* getTextureCoordinateData() { return m_texture_coordinate_data; }
  const float* getTextureCoordinateData() const { return m_texture_coordinate_data; }

  int getVertexStride() const { return m_vertex_stride; }
  int getNormalStride() const { return m_normal_stride; }
  int getColorStride() const { return m_color_stride; }
  int getTextureCoordinateStride() const { return m_texture_coordinate_stride; }

  const float* getBBoxMin() const { return m_bbox_min; }
  const float* getBBoxMax() const { return m_bbox_max; }

  void updateBBox();

  const std::string& getMaterialLibraryName() const { return m_material_library_name; }

  MeshGroup& getMeshGroup(const std::string& group_name) {
    MeshGroupMap::iterator found = m_mesh_groups.find(group_name);
    if( found != m_mesh_groups.end() ) {
      return found->second;
    }
    else {
      throw MeshException( "Could not find group named '" + group_name + "'" );
    }
  }

  const MeshGroup& getMeshGroup(const std::string& group_name) const {
    MeshGroupMap::const_iterator found = m_mesh_groups.find(group_name);
    if( found != m_mesh_groups.end() ) {
      return found->second;
    }
    else {
      throw MeshException( "Could not find group named '" + group_name + "'" );
    }
  }

  size_t getMaterialCount() const { return m_material_params.size(); }

  void setMeshMaterialParams( int i, const MeshMaterialParams& params ) {
    m_material_params[i] = params;
  }
  
  MeshMaterialParams& getMeshMaterialParams( int i ) {
    return m_material_params[i];
  }
  
  const MeshMaterialParams& getMeshMaterialParams( int i ) const
  { 
    return m_material_params[i];
  }

  void          setMeshGrouping(MeshGrouping grouping)  { m_grouping = grouping; }
  MeshGrouping  getMeshGrouping() const                 { return m_grouping; }


  /*--------------------------------------------------------------------------------  
   * Load functions
   */
  void loadFromObj( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial );

  /**
   * Load a material library
   */
  void loadMaterials( const std::string& material_filename );

  /**
   * Similar to loadFromObj() bur for .ply files
   */
  void loadFromPly( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial );

protected:

  /**
   * Accessors for derived classes
   */
  void setVertexData( float* vertex_data ) { m_vertex_data = vertex_data; }
  void setNormalData( float* normal_data ) { m_normal_data = normal_data; }
  void setColorData( unsigned char* color_data ) { m_color_data = color_data; }
  void setTextureCoordinateData( float* texture_coordinate_data ) { m_texture_coordinate_data = texture_coordinate_data; }
  void setMaterialIndices(int* material_indices) { m_material_indices = material_indices; }

  void setVertexStride( int vertex_stride ) { m_vertex_stride = vertex_stride; }
  void setNormalStride( int normal_stride ) { m_normal_stride = normal_stride; }
  void setColorStride( int color_stride ) { m_color_stride = color_stride; }
  void setTextureCoordinateStride( int texture_coordinate_stride ) { m_texture_coordinate_stride = texture_coordinate_stride; }

  const std::string& getFilename() const { return m_filename; }
  const std::string& getPathName() const { return m_pathname; }

  /**
   * These pure virtual functions allow subclasses to control much about the
   * loading behavior.
   */

  //virtual int addMaterial(MeshMaterialParams& params) = 0;

  /**
   * An injection point for subclasses to do any prep before file info
   * is loaded
   */
  virtual void preProcess() = 0;

  /**
   * Call this to allocate all data for the mesh after setting num_vertices,
   * num_normals, num_colors, and num_texture_coordinates, as well as
   * num_triangles for all the groups.
   */
  virtual void allocateData() = 0;
 
  /**
   * Call this before writing any data to the mesh, to set up correct pointer
   * values to write to for vertex_data, normal_data, etc., as well as
   * vertex_indices, normal_indices, etc. for each of the groups.
   *
   * Call finishWritingData() when done.
   */
  virtual void startWritingData() = 0;

  /**
   * An injection point for subclasses to do load postprocessing
   */
  virtual void postProcess() = 0;

  /**
   * Call this when done writing to data or group data.
   */
  virtual void finishWritingData() = 0;
  
  /* -------------------------------------------------------------------------------- 
   * Utility functions
   */
  void computeAabb();

private:
  /*--------------------------------------------------------------------------------  
   * Load functions
   */

  /**
   * Only loads information about data counts, etc., from the .obj file.  Useful
   * for cases where the client code will itself allocate and manage the data
   * arrays that will be used.
   */
  void loadInfoFromObj( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial );

  /**
   * Loads the data from the .obj into arrays that have already been allocated.
   *
   * If info.num_vertices, .num_normals, or .num{any_other_attribute} is '0',
   * or if the corresponding pointers vertex_data, normal_data, etc., in 'data'
   * are 0, then the the loader will not store data for that attribute.
   */
  void loadDataFromObj( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial );

  /**
   * Contract is identical to that of loadInfoFromObj()
   */
  void loadInfoFromPly( const std::string& filename, bool insertDefaultMaterial );

  /**
   * Contract is identical to that of loadDataFromObj()
   */
  void loadDataFromPly( const std::string& filename );

  // For loaders that only use a single group, initializes m_mesh_groups with that
  // one group.
  void initSingleGroup();  

  // Return a reference to the first group found in m_mesh_groups; mostly useful for
  // loaders that only use a single group.  Throws if m_mesh_groups is empty.
  MeshGroup& getFirstGroup();

  /**
   * Returns the group called 'name', or adds it to the groups if it doesn't
   * exist yet.
   */
  MeshGroup& getOrAddGroup( const std::string& name );

  std::string m_filename;
  std::string m_pathname;

  int m_num_vertices;
  int m_num_normals;
  int m_num_colors;
  int m_num_texture_coordinates;

  /* If any of these are 0, then load() won't bother write to the
   * array for that attribute, and it won't write indices for that attribute in
   * any of the groups.
   *
   * It is up to sub classes to implement their own logic for whether data are
   * allocated or not.
   */
  float*         m_vertex_data;
  float*         m_normal_data;
  unsigned char* m_color_data;
  float*         m_texture_coordinate_data;
  int*			 m_material_indices;

  /* Measured in 'floats'.  0 indicates compacted data, equivalent to '2' for
   * texture_coordinates and '3' for the others.
   */
  int m_vertex_stride;
  int m_normal_stride;
  int m_color_stride;
  int m_texture_coordinate_stride;

  int m_num_triangles;

  MeshGrouping						m_grouping;
  MeshGroupMap						m_mesh_groups;
  MeshMaterialNumbersMap			m_material_numbers_by_name; // TODO: replace these two lines with a map from name to material parameters
  std::vector<MeshMaterialParams>	m_material_params;

  std::string m_material_library_name;

  float m_bbox_min[3];
  float m_bbox_max[3];
};


template <class Functor>
void MeshBase::forEachGroup( Functor functor ) const
{
  MeshGroupMap::const_iterator groups_end = m_mesh_groups.end();
  for( MeshGroupMap::const_iterator groups_iter = m_mesh_groups.begin();
       groups_iter != groups_end; )
  {
    functor( (groups_iter++)->second );
      // The post-increment ++ ensures the iterator is updated before functor()
      // runs, since functor may invalidate the iterator
  }
}

template <class Functor>
void MeshBase::forEachGroup( Functor functor ) 
{
  MeshGroupMap::iterator groups_end = m_mesh_groups.end();
  for( MeshGroupMap::iterator groups_iter = m_mesh_groups.begin();
       groups_iter != groups_end; )
  {
    functor( (groups_iter++)->second );
      // The post-increment ++ ensures the iterator is updated before functor()
      // runs, since functor may invalidate the iterator
  }
}
