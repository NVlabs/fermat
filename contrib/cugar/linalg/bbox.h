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

/*! \file bbox.h
 *   \brief Defines an axis-aligned bounding box class.
 */

#pragma once

#ifdef min
#undef min
#endif
#ifdef max
#unded max
#endif

#include <cugar/basic/numbers.h>
#include <cugar/linalg/vector.h>
#include <limits>
#include <algorithm>

namespace cugar {

///@addtogroup LinalgModule
///@{

///@defgroup BboxModule Bounding Boxes
/// This module defines everything related to bounding boxes
///@{

///
/// Axis-Aligned Bounding Bbox class, templated over an arbitrary vector type
///
template <typename Vector_t>
struct Bbox
{
	typedef typename Vector_t::value_type	value_type;
	typedef typename Vector_t				vector_type;

    /// empty constructor
    ///
	CUGAR_HOST CUGAR_DEVICE Bbox();

    /// point constructor
    ///
    /// \param v    point
	CUGAR_HOST CUGAR_DEVICE Bbox(
		const Vector_t& v);

    /// min/max constructor
    ///
    /// \param v1   min corner
    /// \param v2   max corner
    CUGAR_HOST CUGAR_DEVICE Bbox(
		const Vector_t& v1,
		const Vector_t& v2);

    /// merging constructor
    ///
    /// \param bb1  first bbox
    /// \param bb2  second bbox
	CUGAR_HOST CUGAR_DEVICE Bbox(
		const Bbox<Vector_t>& bb1,
		const Bbox<Vector_t>& bb2);

    /// copy constructor
    ///
    /// \param bb   bbox to copy
    CUGAR_HOST CUGAR_DEVICE Bbox(
		const Bbox<Vector_t>& bb);

    /// insert a point
    ///
    /// \param v    point to insert
	CUGAR_HOST CUGAR_DEVICE void insert(const Vector_t& v);

    /// insert a bbox
    ///
    /// \param v    bbox to insert
	CUGAR_HOST CUGAR_DEVICE void insert(const Bbox& v);

    /// clear bbox
    ///
	CUGAR_HOST CUGAR_DEVICE void clear();

    /// const corner indexing operator
    ///
    /// \param i    corner to retrieve
	CUGAR_HOST CUGAR_DEVICE const Vector_t& operator[](const size_t i) const	{ return (&m_min)[i]; }

    /// corner indexing operator
    ///
    /// \param i    corner to retrieve
	CUGAR_HOST CUGAR_DEVICE Vector_t& operator[](const size_t i)				{ return (&m_min)[i]; }

    /// copy operator
    ///
    /// \param bb   bbox to copy
    CUGAR_HOST CUGAR_DEVICE Bbox<Vector_t>& operator=(const Bbox<Vector_t>& bb);

    Vector_t m_min; ///< min corner
	Vector_t m_max; ///< max corner
};

typedef Bbox<Vector2f> Bbox2f;
typedef Bbox<Vector3f> Bbox3f;
typedef Bbox<Vector4f> Bbox4f;
typedef Bbox<Vector2d> Bbox2d;
typedef Bbox<Vector3d> Bbox3d;
typedef Bbox<Vector4d> Bbox4d;
typedef Bbox<Vector2i> Bbox2i;
typedef Bbox<Vector3i> Bbox3i;
typedef Bbox<Vector4i> Bbox4i;
typedef Bbox<Vector2u> Bbox2u;
typedef Bbox<Vector3u> Bbox3u;
typedef Bbox<Vector4u> Bbox4u;

/// compute the area of a 2d bbox
///
/// \param bbox     bbox object
inline CUGAR_HOST_DEVICE float area(const Bbox2f& bbox);

/// compute the area of a 3d bbox
///
/// \param bbox     bbox object
inline CUGAR_HOST_DEVICE float area(const Bbox3f& bbox);

/// point-in-bbox inclusion predicate
///
/// \param bbox     bbox object
/// \param p        point to test for inclusion
template <typename Vector_t>
inline CUGAR_HOST_DEVICE bool contains(const Bbox<Vector_t>& bbox, const Vector_t& p);

/// bbox-in-bbox inclusion predicate
///
/// \param bbox     bbox object
/// \param p        candidate to test for inclusion
template <typename Vector_t>
inline CUGAR_HOST_DEVICE bool contains(const Bbox<Vector_t>& bbox, const Bbox<Vector_t>& candidate);

/// point-to-bbox squared distance
///
/// \param bbox     bbox object
/// \param p        point
template <typename Vector_t>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE float sq_distance(const Bbox<Vector_t>& bbox, const Vector_t& p);

/// returns the largest axis of a bbox
///
/// \param bbox     bbox object
template <typename Vector_t>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE size_t largest_axis(const Bbox<Vector_t>& bbox);

/// returns the delta between corners
///
/// \param bbox     bbox object
template <typename Vector_t>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE Vector_t extents(const Bbox<Vector_t>& bbox);

/// a functor to measure bbox areas
///
template <uint32 DIM>
struct bbox_area_functor
{
	typedef Vector<float,DIM>	vector_type;
	typedef Bbox<vector_type>	bbox_type;
	typedef bbox_type			argument_type;
	typedef float				result_type;

	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	result_type operator() (const bbox_type& bbox) const { return area(bbox); }
};

///@} BboxModule
///@} LinalgModule

} // namespace cugar

#include <cugar/linalg/bbox_inline.h>
