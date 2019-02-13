/*
 * Copyright (c) 2010-2016, NVIDIA Corporation
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

 /*! \file lambert.h
 *   \brief Defines the Lambert BSDF
 *
 * This module provides a Lambert BSDF implementation
 */

#pragma once

#include <cugar/linalg/vector.h>
#include <cugar/basic/numbers.h>
#include <cugar/spherical/mappings.h>


namespace cugar {

/*! \addtogroup BSDFModule BSDF
 *  \{
 */

///
/// Spherical measure type
///
enum SphericalMeasure
{
	kSolidAngle				= 0,
	kProjectedSolidAngle	= 1
};

///
/// A simplified notion of the local differential geometry at a surface
///
struct DifferentialGeometry
{
	Vector3f normal_s;		// shading normal
	Vector3f normal_g;		// geometric normal
	Vector3f tangent;		// local tangent
	Vector3f binormal;		// local binormal

	///\par
	/// transform to the local coordinate system
	///
	CUGAR_HOST_DEVICE
	Vector3f to_local(const Vector3f v) const
	{
		return Vector3f(
			dot( v, tangent ),
			dot( v, binormal ),
			dot( v, normal_s ) );
	}

	///\par
	/// transform back to world space
	///
	CUGAR_HOST_DEVICE
	Vector3f from_local(const Vector3f v) const
	{
		return
			v.x * tangent +
			v.y * binormal +
			v.z * normal_s;
	}

	///\par
	/// cosine of theta of a vector expressed in local coordinates
	///
	CUGAR_HOST_DEVICE
	float cos_theta_l(const Vector3f v) const { return v.z; }

	///\par
	/// cosine of theta of a vector expressed in global coordinates
	///
	CUGAR_HOST_DEVICE
	float cos_theta_g(const Vector3f v) const { return dot( v, normal_s ); }
};

/*! \}
 */

} // namespace cugar
