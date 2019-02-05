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

/*                                                                               
 *  Portions of the OBJ parsing code are taken from:
 *
 *  GLM library.  Wavefront .obj file format reader/writer/manipulator.          
 *                                                                               
 *  Written by Nate Robins, 1997.                                                
 *  email: ndr@pobox.com                                                         
 *  www: http://www.pobox.com/~ndr                                               
 */      

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>

// The .ply parser
#include "rply-1.01/rply.h"

#include <files.h>
#include "MeshBase.h"
#include "MeshException.h"

// Private helper declarations
namespace {

  static const std::string default_group_name =    "null-group";
  static const std::string default_material_name = "null-material";

  static const int MAX_STRING_LENGTH = 128;

  FILE* openObjFile(const std::string& filename)
  {
    FILE* file;

    file = fopen(filename.c_str(), "r");
    if( !file ) {
      throw MeshException( "Mesh loader: can't open file " + filename );
    }

    return file;
  }

  // XXX: Might be nice to make this always return lower-case
  std::string getExtension( const std::string& filename )
  {
    // Get the filename extension
    std::string::size_type extension_index = filename.find_last_of( "." );
    return extension_index != std::string::npos ?
           filename.substr( extension_index+1 ) :
           std::string();
  }

  bool fileIsObj( const std::string& filename )
  {
    return getExtension( filename ) == "obj";
  }


  bool fileIsPly( const std::string& filename )
  {
    return getExtension( filename ) == "ply";
  }


  /**
   * Common point for generating the canonical group name when a new group
   * or material is encountered.
   */
  inline std::string groupMaterialName(
      const MeshGrouping grouping,
      const std::string& group_base_name,
      const std::string& material_name,
      const int          subgroup)
  {
    std::stringstream ss_group_material_name;
    
    if (grouping != kMergeGroups)
        ss_group_material_name << group_base_name << ":" << material_name;
    else
        ss_group_material_name << material_name;
    
    if (grouping == kKeepSubgroups)
        ss_group_material_name << ":" << subgroup;

    return ss_group_material_name.str();
  }

  /** 
   * Returns the directory of a file, given its path.  Respects both
   * forward- and back-slashes in paths.
   */
  std::string directoryOfFilePath( const std::string& filepath )
  {
    size_t slash_pos, backslash_pos;
    slash_pos     = filepath.find_last_of( '/' );
    backslash_pos = filepath.find_last_of( '\\' );

    size_t break_pos;
    if( slash_pos == std::string::npos && backslash_pos == std::string::npos ) {
      return std::string();
    } else if ( slash_pos == std::string::npos ) {
      break_pos = backslash_pos;
    } else if ( backslash_pos == std::string::npos ) {
      break_pos = slash_pos;
    } else {
      break_pos = std::max(slash_pos, backslash_pos);
    }
    
    // Include the final slash
    return filepath.substr(0, break_pos + 1);
  }

  // Helper functor, for use with forEachGraph(), which erases a group
  // from the constructor argument if the group is empty
  struct PruneEmptyGroupsFunctor {
  public:
    PruneEmptyGroupsFunctor( MeshGroupMap& group_map ) :
      m_group_map( group_map )
    { }

    void operator()(MeshGroup& group) {
      if( group.num_triangles < 1)
        m_group_map.erase( group.name );
    }

  private:
    MeshGroupMap& m_group_map;
  };

  // Helper functor for use with forEachGraph()
  struct GroupCurrentIndexInitFunctor {
  public:
    GroupCurrentIndexInitFunctor( std::map<std::string, int>& indices ) :
        m_indices( indices )
    { }
    
    void operator()( MeshGroup& group ) {
      m_indices[group.name] = 0;
    }

  private: 
    std::map<std::string, int>& m_indices;
  };

  // Another helper functor for forEachGraph(), used when loading objs
  // to copy a vertex index to the corresponding color index
  struct VertexIndexToColorIndexCopyFunctor {
  public:
    void operator()( MeshGroup& group ) {
      std::copy( group.vertex_indices, group.vertex_indices + group.num_triangles,
                 group.color_indices );
    }
  };


  // -------------------------------------------------------------------------------- 
  // Helpers for loadFromPly()
  //

  // Used to tell the ply loading code where to write vertices and normals and
  // their indices in the triangles, as well as to track the current vertices
  // being read to.
  class PlyData
  {
  public:

    // vertex_data points to an array where the loader can fill the vertices it loads,
    // and vertex_indices points to where indices for each triangle's vertice can
    // be filled.  Similarly for normal_data and normal_indices.
    PlyData(float* vertex_data, int vertex_stride,
			float* normal_data, int normal_stride,
			float* texture_data, int texture_stride,
			int* vertex_indices, int* normal_indices, int* texture_indices,
			int vindex_stride,
			int nindex_stride,
			int tindex_stride)
    : m_vertex_data( vertex_data ),
	  m_normal_data( normal_data ),
	  m_texture_data( texture_data ),
      m_vertex_indices( vertex_indices ), 
	  m_normal_indices( normal_indices ), 
	  m_texture_indices( texture_indices ),
      m_curr_vertex_index(0), m_curr_normal_index(0), m_curr_texture_index(0), m_curr_triangle_index(0),
      m_vertex_stride( vertex_stride ),
      m_normal_stride( normal_stride ),
      m_texture_stride( texture_stride ),
      m_vindex_stride( vindex_stride ),
      m_nindex_stride( nindex_stride ),
      m_tindex_stride( tindex_stride )
    { }

    friend int plyVertexLoadDataCB( p_ply_argument );
    friend int plyFaceLoadDataCB( p_ply_argument );
    
  private:

    float* m_vertex_data;
    float* m_normal_data;
    float* m_texture_data;
    int*   m_vertex_indices;
    int*   m_normal_indices;
    int*   m_texture_indices;

    int    m_curr_vertex_index;
    int    m_curr_normal_index;
    int    m_curr_texture_index;
    int    m_curr_triangle_index;

	int	   m_vertex_stride;
	int	   m_normal_stride;
	int	   m_texture_stride;

	int	   m_vindex_stride;
	int	   m_nindex_stride;
	int	   m_tindex_stride;
  };


  // Callback for rply to store vertex or normal data
  int plyVertexLoadDataCB( p_ply_argument argument ) 
  {
    int coord_index;
    PlyData* data;
    PlyData** data_pp = &data;
    ply_get_argument_user_data( argument,
                                reinterpret_cast<void**>( data_pp ), &coord_index );

    int vindex_raw = data->m_vertex_stride * data->m_curr_vertex_index;
    int nindex_raw = data->m_normal_stride * data->m_curr_normal_index;
    int tindex_raw = data->m_texture_stride * data->m_curr_texture_index;
    float value = static_cast<float>( ply_get_argument_value( argument ) );

    switch( coord_index ) {
      // Vertex property
      case 0: 
        data->m_vertex_data[vindex_raw + 0] = value;
        break;
      case 1: 
        data->m_vertex_data[vindex_raw + 1] = value;
        break;
      case 2:
        data->m_vertex_data[vindex_raw + 2] = value;
        ++data->m_curr_vertex_index;
        break;

      // Normal property
      case 3: 
        data->m_normal_data[nindex_raw + 0] = value;
        break;
      case 4:
        data->m_normal_data[nindex_raw + 1] = value;
        break;
      case 5: 
        data->m_normal_data[nindex_raw + 2] = value;
        ++data->m_curr_normal_index;
        break;

      // Texture property
      case 6: 
        data->m_texture_data[tindex_raw + 0] = value;
        break;
      case 7:
        data->m_texture_data[tindex_raw + 1] = value;
        ++data->m_curr_texture_index;
        break;
		
		// Silently ignore other coord_index values
    }
    return 1;
  }

  // Callback for rply to store triangle (face) indices.
  int plyFaceLoadDataCB( p_ply_argument argument )
  {
    PlyData* data;
    PlyData** data_pp = &data;
    ply_get_argument_user_data( argument, reinterpret_cast<void**>( data_pp ), NULL );

    int num_verts, which_vertex;
    ply_get_argument_property( argument, NULL, &num_verts, &which_vertex );
    
    int trindex_raw = data->m_curr_triangle_index;
    int value = static_cast<int>( ply_get_argument_value(argument) );

    // num_verts is disregarded; we assume only triangles are given

    switch( which_vertex ) {
      case 0:
        data->m_vertex_indices[data->m_vindex_stride * trindex_raw + 0] = value;
        if (data->m_normal_indices)
			data->m_normal_indices[data->m_vindex_stride * trindex_raw + 0] = value;
        if (data->m_texture_indices)
			data->m_texture_indices[data->m_tindex_stride * trindex_raw + 0] = value;
        break;
      case 1:
        data->m_vertex_indices[data->m_vindex_stride * trindex_raw + 1] = value;
        if (data->m_normal_indices)
			data->m_normal_indices[data->m_nindex_stride * trindex_raw + 1] = value;
        if (data->m_texture_indices)
			data->m_texture_indices[data->m_tindex_stride * trindex_raw + 1] = value;
        break;
      case 2:
        data->m_vertex_indices[data->m_vindex_stride * trindex_raw + 2] = value;
        if (data->m_normal_indices)
			data->m_normal_indices[data->m_nindex_stride * trindex_raw + 2] = value;
        if (data->m_texture_indices)
			data->m_texture_indices[data->m_tindex_stride * trindex_raw + 2] = value;
        ++data->m_curr_triangle_index;
        break;

      // Silently ignore other values of which_vertex
    }

    return 1;
  }
  

} // anonymous namespace


void MeshMaterialParams::setToDefaultParams()
{
  name = default_material_name;

  diffuse[0] = 0.7f;
  diffuse[1] = 0.7f;
  diffuse[2] = 0.7f;
  diffuse[3] = 1.0f;

  diffuse_trans[0] = 0.0f;
  diffuse_trans[1] = 0.0f;
  diffuse_trans[2] = 0.0f;
  diffuse_trans[3] = 1.0f;

  ambient[0] = 0.2f;
  ambient[1] = 0.2f;
  ambient[2] = 0.2f;
  ambient[3] = 1.0f;
  
  specular[0] = 0.0f;
  specular[1] = 0.0f;
  specular[2] = 0.0f;
  specular[3] = 1.0f;
  
  emissive[0] = 0.0f;
  emissive[1] = 0.0f;
  emissive[2] = 0.0f;
  emissive[3] = 1.0f;

  phong_exponent      = 0;
  index_of_refraction = 1;
  opacity             = 1;

  reflectivity[0] = 0;
  reflectivity[1] = 0;
  reflectivity[2] = 0;
  reflectivity[3] = 1;

  flags = 0;

  shading_type = MESH_SHADING_FLAT;

  ambient_map.scaling[0] = 1;
  ambient_map.scaling[1] = 1;

  diffuse_map.scaling[0] = 1;
  diffuse_map.scaling[1] = 1;
  
  diffuse_trans_map.scaling[0] = 1;
  diffuse_trans_map.scaling[1] = 1;

  specular_map.scaling[0] = 1;
  specular_map.scaling[1] = 1;
  
  emissive_map.scaling[0] = 1;
  emissive_map.scaling[1] = 1;

  opacity_map.scaling[0] = 1;
  opacity_map.scaling[1] = 1;

  bump_map.scaling[0] = 1;
  bump_map.scaling[1] = 1;
}


MeshBase::MeshBase() :
  m_num_vertices( 0 ),
  m_num_normals( 0 ),
  m_num_colors( 0 ),
  m_num_texture_coordinates( 0 ),
  m_vertex_data( 0 ),
  m_normal_data( 0 ),
  m_color_data( 0 ),
  m_texture_coordinate_data( 0 ),
  m_material_indices(0),
  m_vertex_stride(0),
  m_normal_stride( 0 ),
  m_color_stride( 0 ),
  m_texture_coordinate_stride( 0 ),
  m_vertex_index_stride(0),
  m_normal_index_stride( 0 ),
  m_color_index_stride( 0 ),
  m_texture_index_stride( 0 ),
  m_num_triangles( 0 ),
  m_grouping( kKeepGroups )
{ }


MeshBase::~MeshBase() { }


void MeshBase::loadModel( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial )
{
  m_filename = filename;
  m_pathname = directoryOfFilePath( filename );

  if( fileIsObj(filename) ) {
    loadFromObj( filename, insertDefaultMaterial, defaultMaterial );
  }
  else if( fileIsPly(filename) ) {
    loadFromPly( filename, insertDefaultMaterial, defaultMaterial );
  }
  else {
    throw MeshException( "Unrecognized model file extension (" + filename + ")" );
  }
}


void MeshBase::updateBBox()
{
  optix::float3* vertices_f3 = reinterpret_cast<optix::float3*>( getVertexData() );
  int num_vertices = getNumVertices();

  optix::Aabb new_bbox;
  for( int i = 0; i < num_vertices; ++i ) {
    new_bbox.include( vertices_f3[i] );
  }
  
  reinterpret_cast<optix::float3*>(m_bbox_min)[0] = new_bbox.m_min;
  reinterpret_cast<optix::float3*>(m_bbox_max)[0] = new_bbox.m_max;
}
  

void MeshBase::loadFromObj( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial )
{
  preProcess();
  loadInfoFromObj( filename, insertDefaultMaterial, defaultMaterial );
  allocateData();
  startWritingData();
  loadDataFromObj( filename, insertDefaultMaterial, defaultMaterial );
  computeAabb();
  postProcess();
  finishWritingData();
}


// Adapted from _glmReadMTL() in the GLM library
void MeshBase::loadMaterials( const std::string& material_filename )
{
  char  buf[2048];
  char  buf2[2048];

  /* open the file */
  ScopedFile file(material_filename.c_str(), "r");
  if (!file) {
    return;
  }

  /* count the number of materials in the file */
  unsigned int num_materials = 1;
  while(fscanf(file, "%s", buf) != EOF) {
    switch(buf[0]) {
      case '#':       /* comment */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;
      case 'n':       /* newmtl */
        fgets(buf, sizeof(buf), file);
        ++num_materials;
        sscanf(buf, "%s %s", buf, buf);
        break;
      default:
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;
    }
  }

  rewind(file);

  // A staging ground for loading the materials until they have all
  // been loaded, to avoid leaving the return value in an incomplete state
  std::vector<MeshMaterialParams> staging_materials( num_materials );

  /* Start with the default values for each material */
  for (size_t i = 0; i < num_materials; ++i) {
    std::stringstream ss_default_params_name;
    ss_default_params_name << staging_materials[i].name << "_" << i;
    staging_materials[i].name = ss_default_params_name.str();
  }

  /* now, read in the data */
  int curr_material_number = 0;
  while(fscanf(file, "%s", buf) != EOF) {
    switch(buf[0]) {
      case '#':       /* comment */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;

      case 'n':       /* newmtl */
        // Make sure the previous material has a name.
        assert( staging_materials[curr_material_number].name != "" );

        // Read in the new material name.
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s %s", buf2, buf2);
        ++curr_material_number;
        staging_materials[curr_material_number].name = std::string(buf2);
        break;
      
      case 'N':
		  if (buf[1] == 's')
			  fscanf(file, "%f", &staging_materials[curr_material_number].phong_exponent);
		  else if (buf[1] == 'i')
			  fscanf(file, "%f", &staging_materials[curr_material_number].index_of_refraction);
        break;

      case 'T': // Tr|Td
		  switch (buf[1])
		  {
		  case 'r':
		  {
			  float transmission;
			  fscanf(file, "%f", &transmission);
			  staging_materials[curr_material_number].opacity = 1.0f - transmission;
			  break;
		  }
		  case 'd':
			  fscanf(file, "%f %f %f",
				  &staging_materials[curr_material_number].diffuse_trans[0],
				  &staging_materials[curr_material_number].diffuse_trans[1],
				  &staging_materials[curr_material_number].diffuse_trans[2]);
			  break;
		  }
        break;

      case 'd': // d
        fscanf(file, "%f", &staging_materials[curr_material_number].opacity);
        break;

      case 'i': // illum
        fscanf(file, "%d", &staging_materials[curr_material_number].shading_type);
        break;

      case 'r': // reflectivity
        float reflectivity;
        fscanf(file, "%f", &reflectivity);
        staging_materials[curr_material_number].reflectivity[0] = reflectivity;
        staging_materials[curr_material_number].reflectivity[1] = reflectivity;
        staging_materials[curr_material_number].reflectivity[2] = reflectivity;
        break;

      case 'e': // emissive
        fscanf( file, "%f %f %f",
                &staging_materials[curr_material_number].emissive[0],
                &staging_materials[curr_material_number].emissive[1],
                &staging_materials[curr_material_number].emissive[2] );
        break;

      case 'f': // flags
        fscanf( file, "%u",
                &staging_materials[curr_material_number].flags );
        break;
	  
	  case 'm':
        {
          MeshTextureMap* map;
          // Determine which type of map.
          if (strcmp(buf,"map_Ka")==0) {
            map = &staging_materials[curr_material_number].ambient_map;
          } else if (strcmp(buf,"map_Kd")==0) {
            map = &staging_materials[curr_material_number].diffuse_map;
          } else if (strcmp(buf,"map_Ks")==0) {
            map = &staging_materials[curr_material_number].specular_map;
          } else if (strcmp(buf,"map_Ke")==0) {
            map = &staging_materials[curr_material_number].emissive_map;
          } else if (strcmp(buf, "map_Td") == 0) {
			  map = &staging_materials[curr_material_number].diffuse_trans_map;
		  }
		  else if (strcmp(buf,"map_D") == 0 ||
				   strcmp(buf,"map_d") == 0) {
            map = &staging_materials[curr_material_number].opacity_map;
		  } else if (strcmp(buf, "map_Bump") == 0 ||
					 strcmp(buf, "map_bump") == 0) {
			map = &staging_materials[curr_material_number].bump_map;
		  } else {
            // We don't know what kind of map it is, so ignore it
            fprintf(stderr, "Unknown map: \"%s\" found at %s(%d)\n", buf,
                __FILE__, __LINE__);
            break;
          }

          char string_literal[2048];
          sprintf(string_literal, "%%%ds", (int)MAX_STRING_LENGTH-1);
          //fprintf(stderr, "string_literal = %s\n", string_literal);

          // Check to see if we have scaled textures or not
          char map_name_in[2048];
          fscanf(file, string_literal, map_name_in);
          if (strcmp(map_name_in, "-s") == 0) {
            // pick up the float scaled textures
            fscanf(file, "%f %f", &map->scaling[0], &map->scaling[1]);
            // Now the name of the file
            fscanf(file, string_literal, map_name_in);
            //fprintf(stderr, "      scale = %f %f ,", scaling[0], scaling[1]);
          }
          map->name = map_name_in;
          //fprintf(stderr, "name = %s\n", map_name);
        } // end case 'm'
        break;

      case 'K':
        switch(buf[1]) {
          case 'd':
			fscanf(file, "%f %f %f",
				&staging_materials[curr_material_number].diffuse[0],
				&staging_materials[curr_material_number].diffuse[1],
				&staging_materials[curr_material_number].diffuse[2]);
            break;
          
          case 's':
            fscanf(file, "%f %f %f",
                &staging_materials[curr_material_number].specular[0],
                &staging_materials[curr_material_number].specular[1],
                &staging_materials[curr_material_number].specular[2]);
            break;
          
          case 'a':
            fscanf(file, "%f %f %f",
                &staging_materials[curr_material_number].ambient[0],
                &staging_materials[curr_material_number].ambient[1],
                &staging_materials[curr_material_number].ambient[2]);
            break;

		  case 'e':
			  fscanf(file, "%f %f %f",
				  &staging_materials[curr_material_number].emissive[0],
				  &staging_materials[curr_material_number].emissive[1],
				  &staging_materials[curr_material_number].emissive[2]);
			  break;

		  case 'r':
			  fscanf(file, "%f %f %f",
				  &staging_materials[curr_material_number].reflectivity[0],
				  &staging_materials[curr_material_number].reflectivity[1],
				  &staging_materials[curr_material_number].reflectivity[2]);
			  break;

		  default:
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        
        }
        break;

      default:
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;
    }
  }

  for (size_t s = 0; s < staging_materials.size(); ++s)
  {
	  m_material_numbers_by_name.insert(std::make_pair(staging_materials[s].name, (int)m_material_params.size()));
	  m_material_params.push_back(staging_materials[s]);
  }
}




//  
//  loadInfoFromObj is based on:
//
//    GLM library.  Wavefront .obj file format reader/writer/manipulator.          
//                                                                               
//    Written by Nate Robins, 1997.                                                
//    email: ndr@pobox.com                                                         
//    www: http://www.pobox.com/~ndr                                               
//
void MeshBase::loadInfoFromObj(const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial)
{
  // TODO: It sure would be nice to clean up or refactor the old GLM-style 
  // parsing paragraphs into helper functions, or even a class in its own right.

  ScopedFile file( openObjFile(filename) );
  
  m_num_vertices            = 0;
  m_num_normals             = 0;
  m_num_colors              = 0;
  m_num_texture_coordinates = 0;
  m_num_triangles           = 0;
  

  // Make a default group
  std::string curr_group_name      = default_group_name;
  MeshGroup*  curr_group           = &getOrAddGroup( curr_group_name );
  std::string curr_group_base_name = curr_group_name;
  int         curr_subgroup = 0;

  int material_count = 0;
  std::string curr_material_name = default_material_name;
  if (insertDefaultMaterial)
  {
	  curr_material_name = defaultMaterial.name;
	  m_material_numbers_by_name[curr_material_name] = material_count;
	  m_material_params.push_back(defaultMaterial);
	  ++material_count;
  }

  int       v, n, t;
  char      buf[2048];
  char		buf2[2048];

  while(fscanf(file, "%s", buf) != EOF) {
    switch(buf[0]) {
      case '#':     /* comment */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;

      case 'v':     /* v, vn, vt */
        switch(buf[1]) {
          case '\0':  { /* vertex */
                        /* eat up rest of line */
                        fgets(buf, sizeof(buf), file);

                        float vx,vy,vz;
                        int   val = -1;
                        sscanf(buf,"%f %f %f %d",&vx, &vy, &vz, &val);
                        if( val >= 0 ) {
                          ++m_num_colors;
                        }

                        ++m_num_vertices;
                        break;
                      }
          case 'n':   /* normal */
                      /* eat up rest of line */
                      fgets(buf, sizeof(buf), file);
                      ++m_num_normals;
                      break;
          case 't':   /* texcoord */
                      /* eat up rest of line */
                      fgets(buf, sizeof(buf), file);
                      ++m_num_texture_coordinates;
                      break;
          default:
                      printf("meshLoaderLoadInfoFromObj(): Unknown token \"%s\".\n", buf);
                      /* Could error out here, but we'll just skip it for now.*/
                      break;
        }
        break;

      case 'm': /* "mtllib <name>" */
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s %s", buf2, buf2);
        m_material_library_name = buf2;
		{
			std::string dir = directoryOfFilePath(filename);
			std::stringstream ss_material_library_name;
			ss_material_library_name << dir << m_material_library_name;
			loadMaterials(ss_material_library_name.str());
		}
        break;

      case 'u': /* "usemtl <name>" */
        /* We need to create groups with their own materials */
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s %s", buf2, buf2);
        curr_material_name = buf2;

        if( m_material_numbers_by_name.find(curr_material_name)
            == m_material_numbers_by_name.end() )
        {
        sscanf(buf, "%s %s", buf2, buf2);
          m_material_numbers_by_name[curr_material_name] = material_count;
          ++material_count;
        }

        curr_group = &getOrAddGroup( groupMaterialName( m_grouping, curr_group_base_name, curr_material_name, curr_subgroup )   );
        break;

      case 'o': /* "o <object name>" */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;

      case 'g': /* "g <group name>" */
        fgets(buf, sizeof(buf2), file);
        sscanf(buf, "%s", buf2);
        curr_group_base_name = buf2;
        curr_group = &getOrAddGroup( groupMaterialName( m_grouping, curr_group_base_name, curr_material_name, curr_subgroup )   );

        curr_subgroup++;
        break;

      case 'f': /* face */
        v = n = t = 0;
        fscanf(file, "%s", buf);
        /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
        if( strstr(buf, "//" )) {
          /* v//n */
          sscanf(buf, "%d//%d", &v, &n);
          fscanf(file, "%d//%d", &v, &n);
          fscanf(file, "%d//%d", &v, &n);
          ++m_num_triangles;
          ++curr_group->num_triangles;
          while(fscanf(file, "%d//%d", &v, &n) > 0) {
            ++m_num_triangles;
            ++curr_group->num_triangles;
          }
        }
        else if( sscanf(buf, "%d/%d/%d", &v, &t, &n ) == 3) {
          /* v/t/n */
          fscanf(file, "%d/%d/%d", &v, &t, &n);
          fscanf(file, "%d/%d/%d", &v, &t, &n);
          ++m_num_triangles;
          ++curr_group->num_triangles;
          while(fscanf(file, "%d/%d/%d", &v, &t, &n) > 0) {
            ++m_num_triangles;
            ++curr_group->num_triangles;
          }
        }
        else if( sscanf(buf, "%d/%d", &v, &t ) == 2) {
          /* v/t */
          fscanf(file, "%d/%d", &v, &t);
          fscanf(file, "%d/%d", &v, &t);
          ++m_num_triangles;
          ++curr_group->num_triangles;
          while(fscanf(file, "%d/%d", &v, &t) > 0) {
            ++m_num_triangles;
            ++curr_group->num_triangles;
          }
        }
        else {
          /* v */
          fscanf(file, "%d", &v);
          fscanf(file, "%d", &v);
          ++m_num_triangles;
          ++curr_group->num_triangles;
          while(fscanf(file, "%d", &v) > 0) {
            ++m_num_triangles;
            ++curr_group->num_triangles;
          }
        }
        break;

      default:
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;
    }
  }
  
  // Prune out groups with 0 triangles;
  forEachGroup( PruneEmptyGroupsFunctor(m_mesh_groups) );
}

//  
//  loadDataFromObj is based on:
//
//    GLM library.  Wavefront .obj file format reader/writer/manipulator.          
//                                                                               
//    Written by Nate Robins, 1997.                                                
//    email: ndr@pobox.com                                                         
//    www: http://www.pobox.com/~ndr                                               
//
void MeshBase::loadDataFromObj( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial )
{
  ScopedFile file( openObjFile(filename) );

  const int vtri_size = getVertexTriangleSize();
  const int ntri_size = getNormalTriangleSize();
  const int ttri_size = getTextureTriangleSize();

  #define VERTEX_INDICES vtri_size
  #define NORMAL_INDICES ntri_size
  #define TEX_INDICES    ttri_size
		
  int vertices_index            = 0;
  int normals_index             = 0;
  int colors_index              = 0;
  int texture_coordinates_index = 0;
  int triangles_index           = 0;

  float*         vertices            = m_vertex_data;
  float*         normals             = m_normal_data;
  unsigned char* colors              = m_color_data;
  float*         texture_coordinates = m_texture_coordinate_data;

  // 0 as index stride signals compact triangle indices
  if (m_vertex_index_stride		== 0) m_vertex_index_stride		= 3;
  if (m_normal_index_stride		== 0) m_normal_index_stride		= 3;
  if (m_color_index_stride		== 0) m_color_index_stride		= 3;
  if (m_texture_index_stride	== 0) m_texture_index_stride	= 3;
  
  // 0 as a stride signals compact data
  int v_stride = m_vertex_stride == 0 ? 3 : m_vertex_stride;
  int n_stride = m_normal_stride == 0 ? 3 : m_normal_stride;
  int c_stride = m_color_stride  == 0 ? 3 : m_color_stride;
  int t_stride = m_texture_coordinate_stride == 0
                                    ? 2 : m_texture_coordinate_stride;

  bool        is_loading_curr_group = false;
  std::string curr_group_name       = default_group_name;
  std::string curr_group_base_name  = default_group_name;
  MeshGroup*  curr_group_data       = 0;
  int         curr_subgroup         = 0;

  MeshGroupMap::const_iterator default_group_iter
    = m_mesh_groups.find(default_group_name);
  if( default_group_iter == m_mesh_groups.end() ) {
    is_loading_curr_group = false;
    curr_group_data = 0;
  }
  else {
    is_loading_curr_group = true;
    curr_group_data       = &m_mesh_groups[default_group_name];
  }

  std::string curr_material_name = default_material_name;
  int curr_material_number = -1;
  if (insertDefaultMaterial)
  {
	  curr_material_name	= defaultMaterial.name;
	  curr_material_number	= 0;
  }

  // Init. the current group triangle index to 0 for each group, so the current
  // triangle index of each can be tracked.
  std::map<std::string, int> groups_triangles_index;
  forEachGroup( GroupCurrentIndexInitFunctor(groups_triangles_index) );

  // For improved speed, so that we don't need a lookup in groups_triangles_index
  // for every single face--only when the group changes.
  int* curr_group_triangles_index = &groups_triangles_index[default_group_name];

  bool uses_vertices = ( m_num_vertices > 0 && vertices != 0 );
  bool uses_normals  = ( m_num_normals  > 0 && normals  != 0 );
  bool uses_colors   = ( m_num_colors   > 0 && colors   != 0 );
  bool uses_texture_coordinates
                     = ( m_num_texture_coordinates > 0
                         && texture_coordinates != 0 );

  unsigned int DEFAULT_TRIANGLE_MASK = 0u;

  int   v[3], n[3], t[3];
  float f[3];
  char  buf[2048];
  char  buf2[2048];

  while(fscanf(file, "%s", buf) != EOF) {
    switch(buf[0]) {
      case '#':       /* comment */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;

      case 'v':       /* v, vn, vt */
        switch(buf[1]) {
          case '\0':      /* vertex */
            if( !uses_colors ) {
              fscanf( file, "%f %f %f", &f[0], &f[1], &f[2] );
              if( uses_vertices ) {
                for( int i = 0; i < 3; ++i ) {
                  vertices[v_stride*vertices_index + i] = f[i];
                }
                ++vertices_index;
              }
            }
            else {
              int c[3];
              fscanf( file, "%f %f %f %d %d %d",
                      &f[0], &f[1], &f[2],
                      &c[0], &c[1], &c[2] );
              if( uses_vertices ) {
                for( int i = 0; i < 3; ++i ) {
                  vertices[v_stride*vertices_index + i] = f[i];
                }
                ++vertices_index;
              }
              if( uses_colors ) {
                for( int i = 0; i < 3; ++i ) {
                  colors[c_stride*colors_index + i] = static_cast<unsigned char>( c[i] );
                  ++colors_index;
                }
              }
            }
            break;

          case 'n':       /* normal */
            fscanf( file, "%f %f %f",
                    &f[0], &f[1], &f[2] );
            if( uses_normals ) {
              for( int i = 0; i < 3; ++i ) {
                normals[n_stride*normals_index + i] = f[i];
              }
              ++normals_index;
            }
            break;

          case 't':       /* texcoord */
            fscanf( file, "%f %f",
                    &f[0], &f[1] );
            if( uses_texture_coordinates ) {
              for( int i = 0; i < 2; ++i ) {
                texture_coordinates[t_stride*texture_coordinates_index + i] = f[i];
              }
              ++texture_coordinates_index;
            }
            break;
        }
        break;

      case 'u': /* "usemtl <name>" */
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s %s", buf2, buf2);
        curr_material_name   = buf2;
        curr_material_number = m_material_numbers_by_name.at(curr_material_name);

		curr_group_name = groupMaterialName( m_grouping, curr_group_base_name, curr_material_name, curr_subgroup );
        // Set up a valid current group only if the new group hasn't been excluded from loading
        if( m_mesh_groups.find(curr_group_name) != m_mesh_groups.end() ) {
          is_loading_curr_group = true;
          curr_group_data = &m_mesh_groups[curr_group_name];
          curr_group_triangles_index = &groups_triangles_index[curr_group_name];

          curr_group_data->material_number = curr_material_number;
        }
        else {
          is_loading_curr_group = false;
          curr_group_data = 0;
          curr_group_triangles_index = 0;
        }
        break;

      case 'o': /* "o <object name>" */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;

      case 'g': /* "g <group name>" */
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s", buf2);

        curr_group_base_name = buf2;
        curr_group_name = groupMaterialName( m_grouping, curr_group_base_name, curr_material_name, curr_subgroup );

        curr_subgroup++;

        // Set up a valid current group only if the new group hasn't been excluded from loading
        if( m_mesh_groups.find( curr_group_name ) != m_mesh_groups.end() ) {
          is_loading_curr_group = true;
          curr_group_data = &m_mesh_groups[curr_group_name];
          curr_group_triangles_index = &groups_triangles_index[curr_group_name];
          curr_group_data->material_number = curr_material_number;
        }
        else {
          is_loading_curr_group = false;
          curr_group_data = 0;
          curr_group_triangles_index = 0;
        }
        break;

      case 'm': /* "mtllib <material library name>" */
        fgets(buf, sizeof(buf), file);
        sscanf(buf, "%s %s", buf2, buf2);
        break;

      case 'f':       /* face */

        #define NEWEST_INDEX(indices, vertex_offset, stride) \
                curr_group_data->indices[stride*(*curr_group_triangles_index) + (vertex_offset)]
        #define PREVIOUS_INDEX(indices, vertex_offset, stride) \
                curr_group_data->indices[stride*(*curr_group_triangles_index - 1) + (vertex_offset)]

        for (int i = 0; i < 3; ++i) {
          v[i] = n[i] = t[i] = 0;
        }

        fscanf(file, "%s", buf);
        /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
        if( strstr(buf, "//" )) {
          /* v//n */
          sscanf(buf,  "%d//%d", &v[0], &n[0]);
          fscanf(file, "%d//%d", &v[1], &n[1]);
          fscanf(file, "%d//%d", &v[2], &n[2]);
          // We still need to advance through the file, but don't store
          // what is parsed for groups we've been told not to load
          if( is_loading_curr_group ) {
            if( uses_vertices ) {
              for (int i = 0; i < 3; ++i) {
                NEWEST_INDEX(vertex_indices, i, VERTEX_INDICES)
                  = (v[i] >= 0) ? v[i] - 1 : (vertices_index + v[i]);
              }
			  NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
            }
            if( uses_normals ) {
              for (int i = 0; i < 3; ++i) {
                NEWEST_INDEX(normal_indices, i, NORMAL_INDICES)
                  = (n[i] >= 0) ? n[i] - 1 : (normals_index + n[i]);
              }
            }
            if( uses_texture_coordinates ) {
              for (int i = 0; i < 3; ++i) {
                NEWEST_INDEX(texture_coordinate_indices, i, 3)
                  = MESH_ATTRIBUTE_NOT_PROVIDED;
              }
			}

			// write the material number
			if (curr_group_data->material_indices)
				curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;

			++(*curr_group_triangles_index);
            ++triangles_index;
          }
          // Load face as a triangle fan when there are more than three indices
          while(fscanf(file, "%d//%d", &v[0], &n[0]) > 0) {
            if( is_loading_curr_group ) {
              if( uses_vertices ) {
                NEWEST_INDEX(vertex_indices, 0, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 0, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 1, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 2, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 2, VERTEX_INDICES) = (v[0] >= 0)
                                                ? v[0] - 1: (vertices_index + v[0]);
				NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
              }
              if( uses_normals ) {
                NEWEST_INDEX(normal_indices, 0, NORMAL_INDICES) = PREVIOUS_INDEX(normal_indices, 0, NORMAL_INDICES);
                NEWEST_INDEX(normal_indices, 1, NORMAL_INDICES) = PREVIOUS_INDEX(normal_indices, 2, NORMAL_INDICES);
                NEWEST_INDEX(normal_indices, 2, NORMAL_INDICES) = (n[0] >= 0)
                                                ? n[0] - 1: (normals_index + n[0]);
              }
              if( uses_texture_coordinates ) {
                for (int i = 0; i < 3; ++i) {
                  NEWEST_INDEX(texture_coordinate_indices, i, TEX_INDICES)
                    = MESH_ATTRIBUTE_NOT_PROVIDED;
                }
              }

			  // write the material number
			  if (curr_group_data->material_indices)
				  curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;

			  ++(*curr_group_triangles_index);
              ++triangles_index;
            }
          }
        }
        else if( sscanf(buf, "%d/%d/%d", &v[0], &t[0], &n[0] ) == 3) {
          /* v/t/n */
          fscanf(file, "%d/%d/%d", &v[1], &t[1], &n[1]);
          fscanf(file, "%d/%d/%d", &v[2], &t[2], &n[2]);
          if( is_loading_curr_group ) {
            if( uses_vertices ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(vertex_indices, i, VERTEX_INDICES)
                  = (v[i] >= 0) ? v[i] - 1 : (vertices_index + v[i]);
              }
			NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
            }
            if( uses_normals ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(normal_indices, i, NORMAL_INDICES)
                  = (n[i] >= 0) ? n[i] - 1 : (normals_index + n[i]);
              }
            }
            if( uses_texture_coordinates ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(texture_coordinate_indices, i, TEX_INDICES)
                  = (t[i] >= 0) ? t[i] - 1 : (texture_coordinates_index + t[i]);
              }
            }

			// write the material number
			if (curr_group_data->material_indices)
				curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;
			
			++(*curr_group_triangles_index);
            ++triangles_index;
          }
          // Load face as a triangle fan when there are more than three indices
          while(fscanf(file, "%d/%d/%d", &v[0], &t[0], &n[0]) > 0) {
            if( is_loading_curr_group ) {
              if( uses_vertices ) {
                NEWEST_INDEX(vertex_indices, 0, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 0, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 1, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 2, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 2, VERTEX_INDICES) = (v[0] >= 0)
                                                ? v[0] - 1 : (vertices_index + v[0]);
				NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
              }
              if( uses_normals ) {
                NEWEST_INDEX(normal_indices, 0, NORMAL_INDICES) = PREVIOUS_INDEX(normal_indices, 0, NORMAL_INDICES);
                NEWEST_INDEX(normal_indices, 1, NORMAL_INDICES) = PREVIOUS_INDEX(normal_indices, 2, NORMAL_INDICES);
                NEWEST_INDEX(normal_indices, 2, NORMAL_INDICES) = (n[0] >= 0)
                                                ? n[0] - 1 : (normals_index + n[0]);
              }
              if( uses_texture_coordinates ) {
                NEWEST_INDEX(texture_coordinate_indices, 0, TEX_INDICES) = PREVIOUS_INDEX(texture_coordinate_indices, 0, TEX_INDICES);
                NEWEST_INDEX(texture_coordinate_indices, 1, TEX_INDICES) = PREVIOUS_INDEX(texture_coordinate_indices, 2, TEX_INDICES);
                NEWEST_INDEX(texture_coordinate_indices, 2, TEX_INDICES) = (t[0] >= 0)
                                                ? t[0] - 1 : (texture_coordinates_index + t[0]);
              }

			  // write the material number
			  if (curr_group_data->material_indices)
				  curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;

			  ++(*curr_group_triangles_index);
              ++triangles_index;
            }
          }
        }
        else if( sscanf(buf, "%d/%d", &v[0], &t[0] ) == 2) {
          /* v/t */
          fscanf(file, "%d/%d", &v[1], &t[1]);
          fscanf(file, "%d/%d", &v[2], &t[2]);
          if( is_loading_curr_group ) {
            if( uses_vertices ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(vertex_indices, i, VERTEX_INDICES)
                  = (v[i] >= 0) ? v[i] - 1 : (vertices_index + v[i]);
              }
			  NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
            }
            if( uses_normals ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(normal_indices, i, NORMAL_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
              }
            }
            if( uses_texture_coordinates ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(texture_coordinate_indices, i, TEX_INDICES)
                  = (t[i] >= 0) ? t[i] - 1 : (texture_coordinates_index + t[i]);
              }
            }

			// write the material number
			if (curr_group_data->material_indices)
				curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;

			++(*curr_group_triangles_index);
            ++triangles_index;
          }
          // Load face as triangle fan when more than three indices
          while(fscanf(file, "%d/%d", &v[0], &t[0]) > 0) {
            if( is_loading_curr_group ) {
              if( uses_vertices ) {
                NEWEST_INDEX(vertex_indices, 0, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 0, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 1, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 2, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 2, VERTEX_INDICES) = (v[0] >= 0)
                                                ? v[0] - 1 : (vertices_index + v[0]);
				NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
              }
              if( uses_normals ) {
                for( int i = 0; i < 3; ++i ) {
                  NEWEST_INDEX(normal_indices, i, NORMAL_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
                }
              }
              if( uses_texture_coordinates ) {
                NEWEST_INDEX(texture_coordinate_indices, 0, TEX_INDICES) = PREVIOUS_INDEX(texture_coordinate_indices, 0, TEX_INDICES);
                NEWEST_INDEX(texture_coordinate_indices, 1, TEX_INDICES) = PREVIOUS_INDEX(texture_coordinate_indices, 2, TEX_INDICES);
                NEWEST_INDEX(texture_coordinate_indices, 2, TEX_INDICES) = (t[0] >= 0)
                                                ? t[0] - 1 : (texture_coordinates_index + t[0]);
              }
			  
			  // write the material number
			  if (curr_group_data->material_indices)
				  curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;
			  
			  ++(*curr_group_triangles_index);
              ++triangles_index;
            }
          }
        }
        else {
          /* v */
          sscanf(buf, "%d", &v[0]);
          fscanf(file, "%d", &v[1]);
          fscanf(file, "%d", &v[2]);
          if( is_loading_curr_group ) {
            if( uses_vertices ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(vertex_indices, i, VERTEX_INDICES)
                  = (v[i] >= 0) ? v[i] - 1 : (vertices_index + v[i]);
              }
			  NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
            }
            if( uses_normals ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(normal_indices, i, NORMAL_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
              }
            }
            if( uses_texture_coordinates ) {
              for( int i = 0; i < 3; ++i ) {
                NEWEST_INDEX(texture_coordinate_indices, i, TEX_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
              }
            }
			
			// write the material number
			if (curr_group_data->material_indices)
				curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;
			
			++(*curr_group_triangles_index);
            ++triangles_index;
          }
          // Load face with more than three indices as a triangle fan
          while(fscanf(file, "%d", &v[0]) > 0) {
            if( is_loading_curr_group ) {
              if( uses_vertices ) {
                NEWEST_INDEX(vertex_indices, 0, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 0, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 1, VERTEX_INDICES) = PREVIOUS_INDEX(vertex_indices, 2, VERTEX_INDICES);
                NEWEST_INDEX(vertex_indices, 2, VERTEX_INDICES) = (v[0] >= 0)
                                                ? v[0] - 1 : (vertices_index + v[0]);
				NEWEST_INDEX(vertex_indices, 3, VERTEX_INDICES) = DEFAULT_TRIANGLE_MASK;
              }
              if( uses_normals ) {
                for( int i = 0; i < 3; ++i ) {
                  NEWEST_INDEX(normal_indices, i, NORMAL_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
                }
              }
              if( uses_texture_coordinates ) {
                for( int i = 0; i < 3; ++i ) {
                  NEWEST_INDEX(texture_coordinate_indices, i, TEX_INDICES) = MESH_ATTRIBUTE_NOT_PROVIDED;
                }
              }
			  
			  // write the material number
			  if (curr_group_data->material_indices)
				  curr_group_data->material_indices[*curr_group_triangles_index] = curr_material_number;
			  
			  ++(*curr_group_triangles_index);
              ++triangles_index;
            }
          }
        }

        #undef PREVIOUS_INDEX
        #undef NEWEST_INDEX

        break;

      default:
        /* eat up rest of line */
        fgets(buf, sizeof(buf), file);
        break;
    }
  }

#if 0
  /* announce the memory requirements */
  printf(" Memory: %d bytes\n",
      vertices_index  * 3*sizeof(float) +
      numnormals   * 3*sizeof(float) * (numnormals ? 1 : 0) +
      numtexcoords * 3*sizeof(float) * (numtexcoords ? 1 : 0) +
      numtriangles * sizeof(GLMtriangle));
#endif

  // It happens that in a .obj, all the color indices are identical to the
  // vertex indices, so we simply copy them
  if( uses_colors ) {
    forEachGroup( VertexIndexToColorIndexCopyFunctor() );
  }
}


void MeshBase::loadFromPly( const std::string& filename, bool insertDefaultMaterial, const MeshMaterialParams& defaultMaterial )
{
  // create a default material
  if (insertDefaultMaterial)
  {
	  m_material_numbers_by_name.insert(std::make_pair(defaultMaterial.name, 0u));
	  m_material_params.push_back(defaultMaterial);
  }

  preProcess();
  loadInfoFromPly( filename, insertDefaultMaterial );
  allocateData();
  startWritingData();
  loadDataFromPly( filename );
  computeAabb();
  postProcess();
  finishWritingData();

  if (insertDefaultMaterial)
  {
	  // set to default material
	  for (int i = 0; i < m_num_triangles; ++i)
		  m_material_indices[i] = 0;
  }
}


void MeshBase::loadInfoFromPly( const std::string& filename, bool insertDefaultMaterial )
{
  p_ply ply = ply_open( filename.c_str(), 0);
  if( !ply ) {
    throw MeshException( "Error opening ply file during first pass (" + filename + ")" );
  }

  if( !ply_read_header( ply ) ) {
    throw MeshException( "Error parsing ply header during first pass (" + filename + ")" );
  }

  // Simply get the counts without setting real callbacks; that's for the second pass
  int num_vertices  = ply_set_read_cb( ply, "vertex", "x", NULL, NULL, 0 ); 
  int num_normals   = ply_set_read_cb( ply, "vertex", "nx", NULL, NULL, 3 );
  int num_textures  = ply_set_read_cb( ply, "vertex", "s", NULL, NULL, 6 );
  if (num_textures == 0)
      num_textures  = ply_set_read_cb( ply, "vertex", "u", NULL, NULL, 6 );
  int num_triangles = ply_set_read_cb( ply, "face", "vertex_indices", NULL, NULL, 0 );

  m_num_vertices			= num_vertices;
  m_num_normals				= num_normals;
  m_num_texture_coordinates	= num_textures;
  m_num_triangles			= num_triangles;

  initSingleGroup();
  MeshGroup& group = getFirstGroup();
  group.num_triangles = num_triangles;
  group.material_number = insertDefaultMaterial ? 0 : -1;

  ply_close(ply);
}


void MeshBase::loadDataFromPly( const std::string& filename )
{
  p_ply ply = ply_open( filename.c_str(), 0);
  if( !ply ) {
    throw MeshException( "Error opening ply file during second pass (" + filename + ")" );
  }

  if( !ply_read_header( ply ) ) {
    throw MeshException( "Error parsing ply header during second pass ("
                         + filename + ")" );
  }

  MeshGroup& group = getFirstGroup();

  PlyData data(
	  m_vertex_data, m_vertex_stride,
	  m_normal_data, m_normal_stride,
	  m_texture_coordinate_data, m_texture_coordinate_stride,
	  group.vertex_indices, group.normal_indices, group.texture_coordinate_indices,
	  m_vertex_index_stride, m_normal_index_stride, m_texture_index_stride );

  ply_set_read_cb( ply, "vertex", "x", plyVertexLoadDataCB, &data, 0);
  ply_set_read_cb( ply, "vertex", "y", plyVertexLoadDataCB, &data, 1);
  ply_set_read_cb( ply, "vertex", "z", plyVertexLoadDataCB, &data, 2);

  if (m_num_normals)
  {
	  ply_set_read_cb( ply, "vertex", "nx", plyVertexLoadDataCB, &data, 3);
	  ply_set_read_cb( ply, "vertex", "ny", plyVertexLoadDataCB, &data, 4);
	  ply_set_read_cb( ply, "vertex", "nz", plyVertexLoadDataCB, &data, 5);
  }

  if (m_num_texture_coordinates)
  {
	  ply_set_read_cb( ply, "vertex", "s", plyVertexLoadDataCB, &data, 6);
	  ply_set_read_cb( ply, "vertex", "t", plyVertexLoadDataCB, &data, 7);
	  ply_set_read_cb( ply, "vertex", "u", plyVertexLoadDataCB, &data, 6);
	  ply_set_read_cb( ply, "vertex", "v", plyVertexLoadDataCB, &data, 7);
  }

  ply_set_read_cb( ply, "face", "vertex_indices", plyFaceLoadDataCB, &data, 0);

  if( !ply_read( ply ) ) {
    throw MeshException( "Error parsing ply file (" + filename + ")" );
  }
  ply_close(ply);
}


void MeshBase::initSingleGroup()
{
  m_mesh_groups.clear();
  getOrAddGroup( default_group_name );
}


MeshGroup& MeshBase::getFirstGroup()
{
  if( m_mesh_groups.size() < 1 ) {
    throw MeshException( "There is no first group to return." );
  }
  return m_mesh_groups.begin()->second;
}


void MeshBase::computeAabb()
{
  if( !m_vertex_data ) {
    throw MeshException( "The mesh is not ready for bounding box computation." );
  }

  optix::Aabb aabb;
  for( int i = 0; i < m_num_vertices; ++i ) {
    aabb.include(reinterpret_cast<const optix::float3*>(m_vertex_data)[i]);
  }

  reinterpret_cast<optix::float3*>(m_bbox_min)[0] = aabb.m_min;
  reinterpret_cast<optix::float3*>(m_bbox_max)[0] = aabb.m_max;
}


/**
 * Wraps the appropriate logic to employ when adding a new group, since
 * certain initializations and checks are important to do this properly.
 */
MeshGroup& MeshBase::getOrAddGroup( const std::string& name )
{
  MeshGroupMap::iterator groups_iter = m_mesh_groups.find( name );
  if( groups_iter == m_mesh_groups.end() ) {
    // Add a new group
    MeshGroup& new_group = m_mesh_groups[name];
    new_group.name = name;
    new_group.num_triangles = 0;
    return new_group;
  }
  else {
    // Return the pre-existing group
    return groups_iter->second;
  }
}

