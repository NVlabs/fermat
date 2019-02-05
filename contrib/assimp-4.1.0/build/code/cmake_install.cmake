# Install script for directory: C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/Assimp")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/Debug/assimp-vc140-mt.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/Release/assimp-vc140-mt.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/MinSizeRel/assimp-vc140-mt.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/RelWithDebInfo/assimp-vc140-mt.lib")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/Debug/assimp-vc140-mt.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/Release/assimp-vc140-mt.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/MinSizeRel/assimp-vc140-mt.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/RelWithDebInfo/assimp-vc140-mt.dll")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp" TYPE FILE FILES
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/anim.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/ai_assert.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/camera.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/color4.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/color4.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/../include/assimp/config.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/defs.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Defines.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/cfileio.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/light.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/material.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/material.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/matrix3x3.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/matrix3x3.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/matrix4x4.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/matrix4x4.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/mesh.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/postprocess.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/quaternion.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/quaternion.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/scene.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/metadata.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/texture.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/types.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/vector2.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/vector2.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/vector3.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/vector3.inl"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/version.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/cimport.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/importerdesc.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Importer.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/DefaultLogger.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/ProgressHandler.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/IOStream.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/IOSystem.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Logger.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/LogStream.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/NullLogger.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/cexport.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Exporter.hpp"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/DefaultIOStream.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/DefaultIOSystem.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/SceneCombiner.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp/Compiler" TYPE FILE FILES
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Compiler/pushpack1.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Compiler/poppack1.h"
    "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/code/../include/assimp/Compiler/pstdint.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/Debug/assimp-vc140-mt.pdb")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/p4research/research/jpantaleoni/Fermat/contrib/assimp-4.1.0/build/code/RelWithDebInfo/assimp-vc140-mt.pdb")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
endif()

