/*
 * Copyright (c) 2010-2018, NVIDIA Corporation
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

/*! \file mappings.h
 *   \brief Defines various spherical mappings.
 */

#pragma once

#include <cugar/basic/numbers.h>
#include <cugar/linalg/vector.h>
#include <algorithm>

namespace cugar {

///
/// \page spherical_page Spherical Functions Module
///\par
/// This \ref spherical_functions "module" provides several constructs to evaluate and manipulate various types
/// of spherical functions and mappings:
///
/// - \ref spherical_harmonics
/// - \ref octahedral_functions
/// - \ref spherical_mappings
///

/*! \addtogroup spherical_functions Spherical
 */

/*! \addtogroup spherical_mappings Spherical Mappings
 *  \ingroup spherical_functions
 *  \{
 */

/// maps a point in spherical coordinates to the unit sphere
///
/// \param uv   uv coordinates
CUGAR_HOST CUGAR_DEVICE Vector3f from_spherical_coords(const Vector2f& uv);

/// computes the spherical coordinates of a 3d point
///
/// \param vec  3d direction
CUGAR_HOST CUGAR_DEVICE Vector2f to_spherical_coords(const Vector3f& vec);

/// map a point on [0,1]^2 to a uniformly distributed point on a disk of radius 1
///
/// \param uv   uv coordinates
CUGAR_HOST CUGAR_DEVICE Vector2f square_to_unit_disk(const Vector2f uv);

/// diskx, disky is point on radius 1 disk.  x, y is point on [0,1]^2
///
/// \param disk     disk point
CUGAR_HOST CUGAR_DEVICE Vector2f unit_disk_to_square(const Vector2f disk);

/// maps the unit square to the hemisphere with a cosine-weighted distribution
///
/// \param uv   uv coordinates
CUGAR_HOST CUGAR_DEVICE Vector3f square_to_cosine_hemisphere(const Vector2f& uv);

/// inverts the square to cosine-weighted hemisphere mapping
///
/// \param dir  3d direction
CUGAR_HOST CUGAR_DEVICE Vector2f cosine_hemisphere_to_square(const Vector3f& dir);

/// maps the unit square to the sphere with a uniform distribution
///
/// \param uv   uv coordinates
CUGAR_HOST CUGAR_DEVICE Vector3f uniform_square_to_sphere(const Vector2f& uv);

/// maps the sphere to a unit square with a uniform distribution
///
/// \param dir  3d direction
CUGAR_HOST CUGAR_DEVICE Vector2f uniform_sphere_to_square(const Vector3f& vec);

/// maps normalized 3d input on +Z hemisphere to the 2d hemioct representation
/// described in "A Survey of Efficient Representations for Independent Unit Vectors"
/// by Cigolle et al.
/// Output is on [-1, 1]^2.
//
CUGAR_HOST CUGAR_DEVICE Vector2f hemisphere_to_hemioct(Vector3f v);

/// maps a 2d vector to a 3d vector on the +Z hemisphere, using the 2d hemioct representation
/// described in "A Survey of Efficient Representations for Independent Unit Vectors"
/// by Cigolle et al.
///
CUGAR_HOST CUGAR_DEVICE Vector3f hemioct_to_hemisphere(Vector2f e);

/// maps normalized 3d input to the 2d oct representation described in
/// "A Survey of Efficient Representations for Independent Unit Vectors"
/// by Cigolle et al.
/// Output is on [-1, 1]^2.
///
CUGAR_HOST CUGAR_DEVICE Vector2f sphere_to_oct(Vector3f v);

/// maps a 2d vector to a 3d vector on the sphere, using the 2d oct representation
/// described in "A Survey of Efficient Representations for Independent Unit Vectors"
/// by Cigolle et al.
///
CUGAR_HOST CUGAR_DEVICE Vector3f oct_to_sphere(Vector2f e);

/*! \}
 */

} // namespace cugar

#include <cugar/spherical/mappings_inline.h>
