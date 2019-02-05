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

#include <types.h>
#include <ray.h>
#include <vertex.h>
#include <mesh_utils.h>
#include <texture.h>
#include <edf.h>
#include <vtl.h>
#include <cugar/basic/algorithms.h>
#include <cugar/spherical/mappings.h>

///@addtogroup Fermat
///@{

///@addtogroup LightsModule
///@{

enum class LightType
{
	kPoint			= 0,
	kDisk			= 1,
	kRectangle		= 2,
	kDirectional	= 3,
	kMesh			= 4,
	kVTL			= 5,
};

/// Represent a VPL
///
struct VPL : VertexGeometryId
{
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	VPL() {}

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	VPL(const uint32 _prim_id, const float2 _uv, const float _e) 
	{
		VertexGeometryId::prim_id = _prim_id;
		VertexGeometryId::uv = _uv;
		E = _e;
	}

	float E;

	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	static float pdf(const float4 E) { return cugar::max3(fabsf(E.x), fabsf(E.y), fabsf(E.z)); }
};

/// Base light interface.
///
/// This class implements "software" inheritance by dispatching its methods to implementations
/// provided by its derived classes.
/// This is useful to allow this class to avoid virtual method calls, and hence allow it to
/// work both on the host and the device.
/// Note that since the methods are manually dispatched, all calls can be inlined.
///
struct Light
{
	LightType type;

#if !defined(OPTIX_COMPILATION)
	FERMAT_HOST_DEVICE
	Light() : type(LightType::kPoint) {}
#else
	FERMAT_HOST_DEVICE
	Light() {}
#endif

	FERMAT_HOST_DEVICE
	Light(LightType _type) : type(_type) {}

	/// sample a point on the light source
	///
	/// \param Z				the input random numbers
	/// \param prim_id		the output primitive index, in case the light is made of a mesh
	/// \param uv			the output uv coordinates on the sampled primitive
	/// \param geom			the output sample's differential geometry
	/// \param pdf			the output sample's area pdf
	/// \param edf			the output sample's EDF
	///
	/// \return true iff the pdf is singular
	FERMAT_HOST_DEVICE
	bool sample(
		const float*		Z,
		uint32_t*			prim_id,
		cugar::Vector2f*	uv,
		VertexGeometry*		geom,
		float*				pdf,
		Edf*				edf) const;

	/// sample a point on the light source given a shading point (or <i>receiver</i>)
	///
	/// \param p			the input shading point
	/// \param Z			the input random numbers
	/// \param prim_id		the output primitive index, in case the light is made of a mesh
	/// \param uv			the output uv coordinates on the sampled primitive
	/// \param geom			the output sample's differential geometry
	/// \param pdf			the output sample's area pdf
	/// \param edf			the output sample's EDF
	///
	/// \return true iff the pdf is singular
	FERMAT_HOST_DEVICE
	bool sample(
		const cugar::Vector3f	p,
		const float*			Z,
		uint32_t*				prim_id,
		cugar::Vector2f*		uv,
		VertexGeometry*			geom,
		float*					pdf,
		Edf*					edf) const;

	/// intersect the given ray with the light source
	///
	/// \param ray			the input ray
	/// \param uv			the output uv coordinates on the sampled primitive
	/// \param t			the output ray intersection distance
	///
	FERMAT_HOST_DEVICE
	void intersect(const Ray ray, float2* uv, float* t) const;

	/// map a (prim,uv) pair to a surface element
	///
	/// \param prim_id		the input primitive index, in case the light is made of a mesh
	/// \param uv			the input uv coordinates on the sampled primitive
	/// \param geom			the output sample's differential geometry
	/// \param pdf			the output sample's area pdf
	/// \param edf			the output sample's EDF
	///
	FERMAT_HOST_DEVICE
	void map(const uint32_t prim_id, const cugar::Vector2f& uv, VertexGeometry* geom, float* pdf, Edf* edf) const;

	/// map a (prim,uv) pair and a (precomputed) surface element to the corresponding edf/pdf
	///
	/// \param prim_id		the input primitive index, in case the light is made of a mesh
	/// \param uv			the input uv coordinates on the sampled primitive
	/// \param geom			the input sample's differential geometry
	/// \param pdf			the output sample's area pdf
	/// \param edf			the output sample's EDF
	///
	FERMAT_HOST_DEVICE
	void map(const uint32_t prim_id, const cugar::Vector2f& uv, const VertexGeometry& geom, float* pdf, Edf* edf) const;
};

/// Disk-light class
///
struct DiskLight : public Light
{
	cugar::Vector3f	 pos;
	cugar::Vector3f	 dir;
	cugar::Vector3f	 u;
	cugar::Vector3f	 v;
	cugar::Vector3f  color;
	float			 radius;

#if !defined(OPTIX_COMPILATION)
	FERMAT_HOST_DEVICE
	DiskLight() : Light(LightType::kDisk) {}
#endif
	/// sample a surface element on the light source
	///
	FERMAT_HOST_DEVICE
	bool sample_impl(
		const float*		Z,
		uint32_t*			prim_id,
		cugar::Vector2f*	uv,
		VertexGeometry*		geom,
		float*				pdf,
		Edf*				edf) const
	{
		*prim_id = 0;

		*uv = cugar::Vector2f(Z[0], Z[1]);

		map_impl(*prim_id, *uv, geom, pdf, edf);
		return false;
	}

	/// intersect the given ray with the light source
	///
	FERMAT_HOST_DEVICE
	void intersect_impl(const Ray ray, float2* uv, float* t) const
	{
		// TODO
		*t = -1;
	}

	/// map a (prim,uv) pair to a surface element
	///
	FERMAT_HOST_DEVICE
	void map_impl(const uint32_t prim_id, const cugar::Vector2f& uv, VertexGeometry* geom, float* pdf, Edf* edf) const
	{
		const float2 disk = cugar::square_to_unit_disk( uv );

		geom->position = pos +
			u * disk.x * radius +
			v * disk.y * radius;

		geom->normal_g = geom->normal_s = cugar::normalize(dir);
		geom->tangent = cugar::orthogonal(geom->normal_g);
		geom->binormal = cugar::cross(geom->normal_g, geom->tangent);

		*pdf = 1.0f / (M_PIf * radius*radius);

		// TODO: write the EDF
	}

	/// map a (prim,uv) pair to a surface element
	///
	FERMAT_HOST_DEVICE
	void map_impl(const uint32_t prim_id, const cugar::Vector2f& uv, const VertexGeometry& geom, float* pdf, Edf* edf) const
	{
		*pdf = 1.0f / (M_PIf * radius*radius);

		// TODO: write the EDF
	}
};

/// Directional-light class
///
struct DirectionalLight : public Light
{
	cugar::Vector3f	 dir;
	cugar::Vector3f  color;

#if !defined(OPTIX_COMPILATION)
	FERMAT_HOST_DEVICE
	DirectionalLight() : Light(LightType::kDirectional) {}
#endif
	/// sample a surface element on the light source
	///
	FERMAT_HOST_DEVICE
	bool sample_impl(
		const float*		Z,
		uint32_t*			prim_id,
		cugar::Vector2f*	uv,
		VertexGeometry*		geom,
		float*				pdf,
		Edf*				edf) const
	{
		// sample a point on the scene's projected bounding disk in direction 'dir
		return true;
	}

	/// sample a surface element on the light source
	///
	FERMAT_HOST_DEVICE
	bool sample_impl(
		const cugar::Vector3f	p,
		const float*			Z,
		uint32_t*				prim_id,
		cugar::Vector2f*		uv,
		VertexGeometry*			geom,
		float*					pdf,
		Edf*					edf) const
	{
		const float FAR = 1.0e8f;

		geom->position = p - dir * FAR;
		geom->normal_s = geom->normal_g = dir;
		geom->tangent  = cugar::orthogonal(dir);
		geom->binormal = cugar::cross(dir,geom->tangent);
		*pdf = 1.0f;
		*edf = Edf( FAR*FAR * color ); 
		return true;
	}
};

/// Mesh-light class
///
struct MeshLight : public Light
{
#if !defined(OPTIX_COMPILATION)
	FERMAT_HOST_DEVICE
	MeshLight() : Light(LightType::kMesh) {}
#else
	FERMAT_HOST_DEVICE
	MeshLight() {}
#endif

	FERMAT_HOST_DEVICE
	MeshLight(const uint32 _n_prims, const float* _prims_cdf, const float* _prims_inv_area, MeshView _mesh, const MipMapView* _textures, const uint32 _n_vpls, const float* _vpls_cdf, const VPL* _vpls, const float _norm) :
		Light(LightType::kMesh), n_prims(_n_prims), prims_cdf(_prims_cdf), prims_inv_area(_prims_inv_area), mesh(_mesh), textures(_textures), n_vpls(_n_vpls), vpls_cdf(_vpls_cdf), vpls(_vpls), norm(_norm) {}

	/// sample a point on the light source
	///
	FERMAT_HOST_DEVICE
	bool sample_impl(
		const float*		Z,
		uint32_t*			prim_id,
		cugar::Vector2f*	uv,
		VertexGeometry*		geom,
		float*				pdf,
		Edf*				edf) const
	{
		const float one = cugar::binary_cast<float>(FERMAT_ALMOST_ONE_AS_INT);

		if (n_vpls)
		{
			// sample one of the VPLs
			const uint32 l = cugar::min( uint32(Z[2] * float(n_vpls)), n_vpls-1 );
			*prim_id = vpls[l].prim_id;
			*uv		 = vpls[l].uv;

			map_impl(*prim_id, *uv, geom, pdf, edf);
		}
		else if (n_prims)
		{
			// sample one of the primitives
			const uint32 tri_id = cugar::upper_bound_index( cugar::min( Z[2], one ), prims_cdf, n_prims);
			FERMAT_ASSERT(tri_id < n_prims);

			*prim_id = tri_id;
			*uv = cugar::Vector2f(Z[0],Z[1]);

			// make sure uv is in the triangle range
			if (uv->x + uv->y > 1.0f)
			{
				uv->x = 1.0f - uv->x;
				uv->y = 1.0f - uv->y;
			}

			map_impl(*prim_id, *uv, geom, pdf, edf);
		}
		else
		{
			*prim_id = 0;
			*uv      = cugar::Vector2f(0.0f);
			*pdf     = 1.0f;
			*edf     = Edf();
		}
		return false;
	}

	/// intersect the given ray with the light source
	///
	FERMAT_HOST_DEVICE
	void intersect_impl(const Ray ray, float2* uv, float* t) const
	{
		// TODO
	}

	/// map a (prim,uv) pair to a surface element
	///
	FERMAT_HOST_DEVICE
	void map_impl(const uint32_t prim_id, const cugar::Vector2f& uv, VertexGeometry* geom, float* pdf, Edf* edf) const
	{
		if (n_vpls || n_prims)
		{
			FERMAT_ASSERT(prim_id < uint32(mesh.num_triangles));
			setup_differential_geometry(mesh, prim_id, uv.x, uv.y, geom);

			const int material_id = mesh.material_indices[prim_id];
			MeshMaterial material = mesh.materials[material_id];

			material.emissive = cugar::Vector4f(material.emissive) * texture_lookup(geom->texture_coords, material.emissive_map, textures, cugar::Vector4f(1.0f));

			if (n_vpls)
				*pdf = VPL::pdf(material.emissive) / norm;
			else
				*pdf =	(prims_cdf[prim_id] - (prim_id ? prims_cdf[prim_id-1] : 0)) * prims_inv_area[prim_id];

			*edf = Edf(material);
		}
		else
		{
			geom->position = cugar::Vector3f(0.0f, 0.0f, 0.0f);
			geom->tangent  = cugar::Vector3f(1.0f, 0.0f, 0.0f);
			geom->binormal = cugar::Vector3f(0.0f, 1.0f, 0.0f);
			geom->normal_s = cugar::Vector3f(0.0f, 0.0f, 1.0f);
			geom->normal_g = cugar::Vector3f(0.0f, 0.0f, 1.0f);

			*pdf = 1.0f;
			*edf = Edf();
		}
	}

	/// map a (prim,uv) pair and its surface element to the corresponding EDF/pdf
	///
	FERMAT_HOST_DEVICE
	void map_impl(const uint32_t prim_id, const cugar::Vector2f& uv, const VertexGeometry& geom, float* pdf, Edf* edf) const
	{
		if (n_vpls || n_prims)
		{
			FERMAT_ASSERT(prim_id < uint32(mesh.num_triangles));
			const int material_id = mesh.material_indices[prim_id];
			MeshMaterial material = mesh.materials[material_id];

			material.emissive = cugar::Vector4f(material.emissive) * texture_lookup(geom.texture_coords, material.emissive_map, textures, cugar::Vector4f(1.0f));

			if (n_vpls)
				*pdf = VPL::pdf(material.emissive) / norm;
			else
				*pdf = (prims_cdf[prim_id] - (prim_id ? prims_cdf[prim_id - 1] : 0)) * prims_inv_area[prim_id];

			*edf = Edf(material);
		}
		else
		{
			*pdf = 1.0f;
			*edf = Edf();
		}
	}

	/// map a (prim,uv) pair to the original random numbers used to sample it
	///
	FERMAT_HOST_DEVICE
	bool invert_impl(const uint32_t prim_id, const cugar::Vector2f& uv, const float* in_Z, float* out_Z, float* out_pdf) const
	{
		if (n_prims)
		{
			FERMAT_ASSERT(prim_id < uint32(mesh.num_triangles));
			const float cdf_1 = prim_id ? prims_cdf[prim_id - 1] : 0;
			const float cdf_2 = prims_cdf[prim_id];
			const float cdf_delta = cdf_2 - cdf_1;

			out_Z[0] = uv.x;
			out_Z[1] = uv.y;
			out_Z[2] = in_Z[0] * cdf_delta + cdf_1;

			*out_pdf = prim_area(mesh, prim_id) / cdf_delta;
			//*out_pdf = 1.0f / (cdf_delta * prim_area( mesh, prim_id ));

			return cdf_delta > 0.0f; // if cdf_delta == 0 the inversion process actually works, but the pdf becomes a Dirac delta
		}
		else
		{
			out_Z[0] = out_Z[1] = out_Z[2] = 0.5f;
			*out_pdf = 1.0f;
			return true;
		}
	}

	/// map a (prim,uv) pair to the original random numbers used to sample it
	///
	FERMAT_HOST_DEVICE
	float inverse_pdf_impl(const uint32_t prim_id, const cugar::Vector2f& uv, const float* out_Z) const
	{
		if (n_prims)
		{
			FERMAT_ASSERT(prim_id < uint32(mesh.num_triangles));
			const float cdf_1 = prim_id ? prims_cdf[prim_id - 1] : 0;
			const float cdf_2 = prims_cdf[prim_id];
			const float cdf_delta = cdf_2 - cdf_1;

			// if cdf_delta == 0 the inversion process actually works, but the pdf becomes a Dirac delta
			return prim_area(mesh, prim_id) / cdf_delta;
			//return 1.0f / (cdf_delta * prim_area(mesh, prim_id));
		}
		else
			return 1.0f;
	}

	FERMAT_HOST_DEVICE
	uint32 vpl_count() const { return n_vpls; }

	FERMAT_HOST_DEVICE
	VPL get_vpl(const uint32 i) const { return vpls[i]; }

	/// map a given VPL to its surface and sampling info
	///
	FERMAT_HOST_DEVICE
	void map_vpl(
		const uint32			vpl_idx,
		uint32_t*				prim_id,
		cugar::Vector2f*		uv,
		VertexGeometry*			geom,
		float*					pdf,
		Edf*					edf) const
	{
		// sample one of the VPLs
		*prim_id = vpls[vpl_idx].prim_id;
		*uv		 = vpls[vpl_idx].uv;

		map_impl(*prim_id, *uv, geom, pdf, edf);
	}

	uint32				n_prims;
	const float*		prims_cdf;
	const float*		prims_inv_area;
	MeshView			mesh;
	const MipMapView*	textures;
	uint32				n_vpls;
	const float*		vpls_cdf;
	const VPL*			vpls;
	float				norm;
};


// sample a point on the light source
//
FERMAT_HOST_DEVICE
inline bool Light::sample(
	const float*		Z,
	uint32_t*			prim_id,
	cugar::Vector2f*	uv,
	VertexGeometry*		geom,
	float*				pdf,
	Edf*				edf) const
{
	switch (type)
	{
	case LightType::kDisk:
		return reinterpret_cast<const DiskLight*>(this)->sample_impl( Z, prim_id, uv, geom, pdf, edf );
	case LightType::kMesh:
		return reinterpret_cast<const MeshLight*>(this)->sample_impl( Z, prim_id, uv, geom, pdf, edf );
	case LightType::kDirectional:
		return reinterpret_cast<const DirectionalLight*>(this)->sample_impl( Z, prim_id, uv, geom, pdf, edf );
	}
	return true;
}

// sample a point on the light source
//
FERMAT_HOST_DEVICE
inline bool Light::sample(
	const cugar::Vector3f	p,
	const float*			Z,
	uint32_t*				prim_id,
	cugar::Vector2f*		uv,
	VertexGeometry*			geom,
	float*					pdf,
	Edf*					edf) const
{
	switch (type)
	{
	case LightType::kDisk:
		return reinterpret_cast<const DiskLight*>(this)->sample_impl( Z, prim_id, uv, geom, pdf, edf ); // NOTE: for now using the generic, non-anchored implementation
	case LightType::kMesh:
		return reinterpret_cast<const MeshLight*>(this)->sample_impl( Z, prim_id, uv, geom, pdf, edf ); // NOTE: for now using the generic, non-anchored implementation
	case LightType::kDirectional:
		return reinterpret_cast<const DirectionalLight*>(this)->sample_impl( p, Z, prim_id, uv, geom, pdf, edf );
	}
	return true;
}

// intersect the given ray with the light source
//
FERMAT_HOST_DEVICE
inline void Light::intersect(const Ray ray, float2* uv, float* t) const
{
	switch (type)
	{
	case LightType::kDisk:
		reinterpret_cast<const DiskLight*>(this)->intersect_impl( ray, uv, t );
		break;
	case LightType::kMesh:
		reinterpret_cast<const MeshLight*>(this)->intersect_impl(ray, uv, t);
		break;
	}
}

// map a (prim,uv) pair to a surface element
//
FERMAT_HOST_DEVICE
inline void Light::map(const uint32_t prim_id, const cugar::Vector2f& uv, VertexGeometry* geom, float* pdf, Edf* edf) const
{
	switch (type)
	{
	case LightType::kDisk:
		reinterpret_cast<const DiskLight*>(this)->map_impl( prim_id, uv, geom, pdf, edf );
		break;
	case LightType::kMesh:
		reinterpret_cast<const MeshLight*>(this)->map_impl( prim_id, uv, geom, pdf, edf );
		break;
	}
}

// map a (prim,uv) pair to a surface element
//
FERMAT_HOST_DEVICE
inline void Light::map(const uint32_t prim_id, const cugar::Vector2f& uv, const VertexGeometry& geom, float* pdf, Edf* edf) const
{
	switch (type)
	{
	case LightType::kDisk:
		reinterpret_cast<const DiskLight*>(this)->map_impl(prim_id, uv, geom, pdf, edf);
		break;
	case LightType::kMesh:
		reinterpret_cast<const MeshLight*>(this)->map_impl(prim_id, uv, geom, pdf, edf);
		break;
	}
}

///@} LightsModule
///@} Fermat
