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

// ------------------------------------------------------------------------- //
//
// Declaration of the composite BSDF class.
//
// ------------------------------------------------------------------------- //

#include <types.h>
#include <cugar/linalg/vector.h>
#include <cugar/bsdf/lambert.h>
#include <cugar/bsdf/lambert_trans.h>
#include <cugar/bsdf/ggx.h>
#include <cugar/bsdf/ggx_smith.h>
#include <cugar/bsdf/ltc.h>
#include <cugar/bsdf/refraction.h>
#include <renderer_view.h>
#include <mis_utils.h>

#define DIFFUSE_ONLY		0
#define SPECULAR_ONLY		0
#define SUPPRESS_SPECULAR	0
#define SUPPRESS_DIFFUSE	0

#define USE_EFFICIENT_SAMPLER_WITH_APPROXIMATE_PDFS	1
	// When this is set to 1, a different, more efficient sampling method will be used, for which the pdf computation
	// will no longer be exact - in the sense it will no longer match the actual pdfs usded for sampling - which would
	// have to be expressed as an integral with no closed-form solution - yet, for the purpose of MIS, exact pdfs are
	// not really needed.
	// One potential problem with this efficient sampler is that it makes exact inversion impossible - in the sense
	// that the probabilities assigned to the various components are stochastic variables and cannot be computed, and
	// invertex, exactly.

///@addtogroup Fermat
///@{

///@defgroup BSDFModule
///\par
/// This module defines the Bidirectional Scattering Distribution Function employed throughout Fermat.
/// At present time, this model is rather simple, and consists of four layered components only:
///\par
/// - a diffuse reflection component
/// - a diffuse transmission component
/// - a glossy reflection component layered on top of the diffuse layer
/// - a glossy transmission component layered on top of the diffuse layer
///\par
/// The diffuse model is purely Lambertian, while the glossy components are based on the GGX model with Smith's joint masking-shadowing function
/// described in:
///> [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
///
///@{

/// transport type: defines whether we are tracing rays, or particles, i.e. whether the quantity being transported is radiance or power.
/// 
enum TransportType
{
	kRadianceTransport	= 0,
	kParticleTransport	= 1
};

//#define USE_LTC
#define USE_GGX_SMITH

///
/// Composite BSDF class
///\par
/// This class defines the Bidirectional Scattering Distribution Function employed throughout Fermat.
/// At present time, this model is rather simple, and consists of five layered components only:
///\par
/// - a diffuse reflection component
/// - a diffuse transmission component
/// - a glossy reflection component layered on top of the diffuse layer
/// - a glossy transmission component layered on top of the diffuse layer
/// - a clearcoat layer on top of all of the above
///\par
///
/// The interaction between the top clearcoat layer and the inner layers below that is modeled approximately,
/// computing only the Fresnel transmission factor to enter the clearcoat, but without modeling the change
/// due to refraction of the input and output directions a la Weidlich and Wilkie.
/// This is done because the approach from Weidlich and Wilkie simulates only first-order scattering, and results
/// in dramatic energy loss for rough materials.
/// It is also questionable whether that would be more correct, given that accounting for multiple-scattering through
/// the various layers can be modeled without altering the final lobe directions, and while only changing the statistics
/// of the inner layer distributions, as described in:
/// "Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators",
/// Laurent Belcour - in ACM Transactions on Graphics, Association for Computing Machinery, 2018.
/// While we plan to adopt the full machinery of the latter in the future, we currently just crudely approximate it.
///
/// The inner layers of the BSDF use Fresnel coefficients to determine how much light undergoes glossy reflection, and how
/// much undergoes transmission. Part of the radiance transmitted from the upper glossy layer undergoes
/// diffuse scattering. Again, the interaction between the glossy layer and the underlying diffuse layer is
/// modeled in a simplified manner, as if the layer was infinitely thin and the diffusely reflected particles
/// were not interacting again with the upper layer.
///
struct Bsdf
{
	/// component type indices
	///
	enum ComponentIndex
	{
		kDiffuseReflectionIndex		= 0u,
		kDiffuseTransmissionIndex	= 1u,
		kGlossyReflectionIndex		= 2u,
		kGlossyTransmissionIndex	= 3u,
		kClearcoatReflectionIndex	= 4u,
		kNumComponents				= 5u,
	};

	/// component type bitmasks
	///
	enum ComponentType
	{
		kAbsorption				= 0u,

		kDiffuseReflection		= 0x1u,
		kDiffuseTransmission	= 0x2u,
		kGlossyReflection		= 0x4u,
		kGlossyTransmission		= 0x8u,
		kClearcoatReflection	= 0x10u,

		kDiffuseMask			= 0x3u,
		kGlossyMask				= 0xCu,

		kReflectionMask			= 0x5u,
		kTransmissionMask		= 0xAu,
		kAllComponents			= 0xFFu
	};

	typedef cugar::LambertTransBsdf	diffuse_trans_component;
	typedef cugar::LambertBsdf		diffuse_component;
  #if defined(USE_LTC)
	typedef cugar::LTCBsdf			glossy_component;
  #elif defined(USE_GGX_SMITH)
	typedef cugar::GGXSmithBsdf		glossy_component;
  #else
	typedef cugar::GGXBsdf			glossy_component;
  #endif

	/// return the number of components
	///
	FERMAT_HOST_DEVICE
	static uint32 component_count() { return 4; }

	/// return the 0-based index corresponding to a given component
	///
	FERMAT_HOST_DEVICE
	static uint32 component_index(const ComponentType comp)
	{
		if (comp == kDiffuseReflection)		return 0;
		if (comp == kDiffuseTransmission)	return 1;
		if (comp == kGlossyReflection)		return 2;
		if (comp == kGlossyTransmission)	return 3;

		return 4;
	}

	/// return the bitmask corresponding to a given component
	///
	FERMAT_HOST_DEVICE
	static ComponentType component_mask(const ComponentIndex comp) { return ComponentType(1u << comp); }

	/// constructor
	///
	FERMAT_HOST_DEVICE
	Bsdf() :
	#if defined(USE_LTC)
		m_diffuse(0.0f), m_diffuse_trans(0.0f), m_glossy(0.0f, NULL, NULL, NULL, 0u), m_glossy_trans(1.0f,true), m_fresnel(0.0f) {}
	#else
		m_diffuse(0.0f), m_diffuse_trans(0.0f), m_glossy(1.0f), m_glossy_trans(1.0f,true), m_fresnel(0.0f) {}
	#endif

	/// copy constructor
	///
	FERMAT_HOST_DEVICE
	Bsdf(const Bsdf& bsdf) :
		m_diffuse(bsdf.m_diffuse),
		m_diffuse_trans(bsdf.m_diffuse_trans),
		m_glossy(bsdf.m_glossy),
		m_glossy_trans(bsdf.m_glossy_trans),
		m_fresnel(bsdf.m_fresnel),
		m_reflectivity(bsdf.m_reflectivity),
		m_ior(bsdf.m_ior),
		m_opacity(bsdf.m_opacity),
		m_clearcoat_ior(bsdf.m_clearcoat_ior),
		m_glossy_reflectance(bsdf.m_glossy_reflectance),
		m_transport(bsdf.m_transport) {}

	/// constructor
	///
	FERMAT_HOST_DEVICE
	Bsdf(
		const TransportType			transport,
		const RenderingContextView	renderer,
		const MeshMaterial			material,
		const float					mollification_factor = 1.0f,
		const float					mollification_bias   = 0.0f,
		const float					min_roughness		 = 0.0f) :
		m_diffuse(cugar::Vector3f(material.diffuse.x, material.diffuse.y, material.diffuse.z) / M_PIf),
		m_diffuse_trans(cugar::Vector3f(material.diffuse_trans.x, material.diffuse_trans.y, material.diffuse_trans.z) / M_PIf),
	#if defined(USE_LTC)
		m_glossy(cugar::max(material.roughness * mollification_factor + mollification_bias, min_roughness), renderer.ltc_M, renderer.ltc_Minv, renderer.ltc_A, renderer.ltc_size),
	#else
		m_glossy(cugar::max(material.roughness * mollification_factor + mollification_bias, min_roughness)),
		m_glossy_trans(material.roughness /** mollification_factor + mollification_bias*/, true, material.index_of_refraction, 1.0f),
	#endif
		m_fresnel(cugar::Vector3f(material.specular.x, material.specular.y, material.specular.z) / M_PIf),
		m_reflectivity(cugar::Vector3f(material.reflectivity.x, material.reflectivity.y, material.reflectivity.z)),
		m_ior(material.index_of_refraction),
		m_opacity(material.opacity),
		m_glossy_reflectance(renderer.glossy_reflectance),
		m_transport(transport)
	{
		const float R0	= cugar::min( cugar::max_comp( m_reflectivity ), 0.95f );
		m_clearcoat_ior = (1 + sqrtf(R0)) / (1 - sqrtf(R0));	// calculate the clearcoat's IOR based on the normal incidence reflectivity
	}

	/// constructor
	///
	FERMAT_HOST_DEVICE
	Bsdf(
		const TransportType			transport, 
		const RenderingContextView	renderer,
		const cugar::Vector3f		diffuse,
		const cugar::Vector3f		specular,
		const float					roughness,
		const cugar::Vector3f		diffuse_trans,
		const float					opacity			= 1.0f,
		const float					ior				= 1.0f) :
		m_diffuse(diffuse / M_PIf),
		m_diffuse_trans(diffuse_trans / M_PIf),
	#if defined(USE_LTC)
		m_glossy(roughness, renderer.ltc_M, renderer.ltc_Minv, renderer.ltc_A, renderer.ltc_size),
	#else
		m_glossy(roughness),
		m_glossy_trans(roughness, true, ior, 1.0f),
	#endif
		m_fresnel(specular / M_PIf),
		m_ior(ior),
		m_opacity(opacity),
		m_glossy_reflectance(renderer.glossy_reflectance),
		m_transport(transport)
	{
		const float R0	= cugar::min( cugar::max_comp( m_reflectivity ), 0.95f );
		m_clearcoat_ior = (1 + sqrtf(R0)) / (1 - sqrtf(R0));	// calculate the clearcoat's IOR based on the normal incidence reflectivity
	}

	/// return the diffuse transmission component
	///
	FERMAT_HOST_DEVICE
	const diffuse_trans_component& diffuse_trans() const { return m_diffuse_trans; }

	/// return the diffuse reflection component
	///
	FERMAT_HOST_DEVICE
	const diffuse_component& diffuse() const { return m_diffuse; }

	/// return the glossy reflection component
	///
	FERMAT_HOST_DEVICE
	const glossy_component& glossy() const { return m_glossy; }

	/// return the glossy transmission component
	///
	FERMAT_HOST_DEVICE
	const glossy_component& glossy_trans() const { return m_glossy_trans; }

	/// return the incident ior ratio
	///
	FERMAT_HOST_DEVICE
	float get_eta(const float NoV)		const { return NoV > 0.0f ? 1.0f / m_ior : m_ior; }

	/// return the incident inverse ior ratio
	///
	FERMAT_HOST_DEVICE
	float get_inv_eta(const float NoV)	const { return NoV > 0.0f ? m_ior : 1.0f / m_ior; }

	/// evaluate the BSDF f(V,L)
	///
	/// \param geometry				the local differential geometry
	/// \param in					the incoming direction
	/// \param out					the outgoing direction
	/// \param components			the components to consider
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	cugar::Vector3f f(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		const ComponentType					components	= kAllComponents) const
	{
		// compute the true component weights together with the transmitted directions and coefficients
		cugar::Vector3f Fc_1, Tc_1;
		cugar::Vector3f Fc_2, Tc_2;
		cugar::Vector3f w[4];
		component_weights( geometry, w_i, w_o, Fc_1, Tc_1, Fc_2, Tc_2, w );

		const float factor = compression_factor( geometry, w_i, w_o );

		return
			((components & kDiffuseReflection)	 ? m_diffuse.f(geometry, w_i, w_o)			* w[kDiffuseReflectionIndex]   * factor	: 0.0f) + 
			((components & kDiffuseTransmission) ? m_diffuse_trans.f(geometry, w_i, w_o)	* w[kDiffuseTransmissionIndex] * factor	: 0.0f) +
			((components & kGlossyReflection)    ? m_glossy.f(geometry, w_i, w_o)			* w[kGlossyReflectionIndex]    * factor	: 0.0f) + 
			((components & kGlossyTransmission)	 ? m_glossy_trans.f(geometry, w_i, w_o)		* w[kGlossyTransmissionIndex]  * factor : 0.0f);
	}

	/// evaluate the BSDF f(V,L) separately for all components
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void f(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		cugar::Vector3f*					f) const
	{
		// compute the true component weights together with the transmitted directions and coefficients
		cugar::Vector3f Fc_1, Tc_1;
		cugar::Vector3f Fc_2, Tc_2;
		cugar::Vector3f w[4];
		component_weights( geometry, w_i, w_o, Fc_1, Tc_1, Fc_2, Tc_2, w );

		cugar::Vector3f f_d, f_g, f_dt, f_gt;

		f_d = m_diffuse.f(geometry, w_i, w_o);
		f_g = m_diffuse_trans.f(geometry, w_i, w_o);
		f_dt = m_glossy.f(geometry, w_i, w_o);
		f_gt = m_glossy_trans.f(geometry, w_i, w_o);

		const float factor = compression_factor( geometry, w_i, w_o );

		f[kDiffuseReflectionIndex]		= f_d	* w[kDiffuseReflectionIndex]   * factor;
		f[kDiffuseTransmissionIndex]	= f_dt	* w[kDiffuseTransmissionIndex] * factor;
		f[kGlossyReflectionIndex]		= f_g	* w[kGlossyReflectionIndex]    * factor;
		f[kGlossyTransmissionIndex]		= f_gt	* w[kGlossyTransmissionIndex]  * factor;
	}

	/// evaluate the BSDF f(V,L) and its pdf p(V,L) in a single call
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void f_and_p(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		cugar::Vector3f*					f,
		float*								p,
		const cugar::SphericalMeasure		measure = cugar::kProjectedSolidAngle,
		bool								RR = true) const
	{
		// compute the true component weights together with the transmitted directions and coefficients
		cugar::Vector3f Fc_1, Tc_1;
		cugar::Vector3f Fc_2, Tc_2;
		cugar::Vector3f w[4];
		component_weights( geometry, w_i, w_o, Fc_1, Tc_1, Fc_2, Tc_2, w );

		// compute the coat reflection & transmission probabilities
		float coat_reflection_prob   = cugar::average(Fc_1);
		float coat_transmission_prob = 1.0f - coat_reflection_prob;

		// evaluate the inner layers
		cugar::Vector3f f_d, f_g, f_dt, f_gt;
		float           p_d, p_g, p_dt, p_gt;

		m_diffuse.f_and_p(geometry, w_i, w_o, f_d, p_d, measure);
		m_diffuse_trans.f_and_p(geometry, w_i, w_o, f_dt, p_dt, measure);
		m_glossy.f_and_p(geometry, w_i, w_o, f_g, p_g, measure);
		m_glossy_trans.f_and_p(geometry, w_i, w_o, f_gt, p_gt, measure);

		// compute the sampling weights for the inner layers
		float w_p[4];
		sampling_weights(geometry, w_i, w_p);

		// modulate the sampling weights by the coat transmission probability
		normalize_sampling_weights( w_p, coat_reflection_prob, coat_transmission_prob, RR );

		p[kDiffuseReflectionIndex]		= p_d	* w_p[kDiffuseReflectionIndex];
		p[kDiffuseTransmissionIndex]	= p_dt	* w_p[kDiffuseTransmissionIndex];
		p[kGlossyReflectionIndex]		= p_g	* w_p[kGlossyReflectionIndex];
		p[kGlossyTransmissionIndex]		= p_gt	* w_p[kGlossyTransmissionIndex];

		const float factor = compression_factor( geometry, w_i, w_o );

		f[kDiffuseReflectionIndex]		= f_d	* w[kDiffuseReflectionIndex]   * factor;
		f[kDiffuseTransmissionIndex]	= f_dt	* w[kDiffuseTransmissionIndex] * factor;
		f[kGlossyReflectionIndex]		= f_g	* w[kGlossyReflectionIndex]    * factor;
		f[kGlossyTransmissionIndex]		= f_gt	* w[kGlossyTransmissionIndex]  * factor;
	}

	/// evaluate the BSDF f(V,L) and its p(V,L) in a single call
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void f_and_p(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		cugar::Vector3f&					f,
		float&								p,
		const cugar::SphericalMeasure		measure		= cugar::kProjectedSolidAngle,
		const bool							RR			= true,
		const ComponentType					components	= kAllComponents) const
	{
		// compute the true component weights together with the transmitted directions and coefficients
		cugar::Vector3f Fc_1, Tc_1;
		cugar::Vector3f Fc_2, Tc_2;
		cugar::Vector3f w[4];
		component_weights( geometry, w_i, w_o, Fc_1, Tc_1, Fc_2, Tc_2, w );

		// compute the coat reflection & transmission probabilities
		float coat_reflection_prob   = cugar::average(Fc_1);
		float coat_transmission_prob = 1.0f - coat_reflection_prob;

		// evaluate the inner layers
		cugar::Vector3f f_d(0.0f), f_g(0.0f), f_dt(0.0f), f_gt(0.0f);
		float           p_d(0.0f), p_g(0.0f), p_dt(0.0f), p_gt(0.0f);

		if (components & kDiffuseReflection)	m_diffuse.f_and_p(geometry, w_i, w_o, f_d, p_d, measure);
		if (components & kDiffuseTransmission)	m_diffuse_trans.f_and_p(geometry, w_i, w_o, f_dt, p_dt, measure);
		if (components & kGlossyReflection)		m_glossy.f_and_p(geometry, w_i, w_o, f_g, p_g, measure);
		if (components & kGlossyTransmission)	m_glossy_trans.f_and_p(geometry, w_i, w_o, f_gt, p_gt, measure);

		// compute the sampling weights for the inner layers
		float w_p[4];
		sampling_weights(geometry, w_i, w_p);

		// modulate the sampling weights by the coat transmission probability
		normalize_sampling_weights( w_p, coat_reflection_prob, coat_transmission_prob, RR, components );

		p = p_d		* w_p[kDiffuseReflectionIndex]	 +
			p_dt	* w_p[kDiffuseTransmissionIndex] +
			p_g		* w_p[kGlossyReflectionIndex]	 +
			p_gt	* w_p[kGlossyTransmissionIndex]; 
		
		const float factor = compression_factor( geometry, w_i, w_o );

		f = f_d		* w[kDiffuseReflectionIndex]   * factor +
			f_dt	* w[kDiffuseTransmissionIndex] * factor +
			f_g		* w[kGlossyReflectionIndex]    * factor +
			f_gt	* w[kGlossyTransmissionIndex]  * factor;
	}

	/// evaluate the total projected probability p(V,L) = p(L|V)
	///
	/// \param geometry				the local differential geometry
	/// \param w_i					the incoming direction
	/// \param w_o					the outgoing direction
	/// \param measure				the spherical measure to use
	/// \param RR					indicate whether to use Russian-Roulette or not
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	float p(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		const cugar::SphericalMeasure		measure		= cugar::kProjectedSolidAngle,
		const bool							RR			= true,
		const ComponentType					components	= kAllComponents) const
	{
		// evaluate the clearcoat's transmission
		cugar::Vector3f H_c;
		cugar::Vector3f Fc_1;
		cugar::Vector3f Tc_1;
		float			eta_c;
		float			cos_theta_i;

		if (!clearcoat_transmission(
			geometry,
			w_i,
			H_c,
			cos_theta_i,
			eta_c,
			Fc_1,
			Tc_1 ))
		{
			// total internal reflection - no chance to enter the clearcoat layer
			return 0.0f;
		}

		// compute the coat reflection & transmission probabilities
		float coat_reflection_prob   = cugar::average(Fc_1);
		float coat_transmission_prob = 1.0f - coat_reflection_prob;

		// compute the sampling weights for the inner layers
		float w_p[4];
		sampling_weights(geometry, w_i, w_p);

		// modulate the sampling weights by the coat transmission probability
		normalize_sampling_weights( w_p, coat_reflection_prob, coat_transmission_prob, RR, components );

		float p_d(0.0f), p_g(0.0f), p_dt(0.0f), p_gt(0.0f);

		if (components & kDiffuseReflection)	p_d  = m_diffuse.p(geometry, w_i, w_o, measure);
		if (components & kDiffuseTransmission)	p_dt = m_diffuse_trans.p(geometry, w_i, w_o, measure);
		if (components & kGlossyReflection)		p_g  = m_glossy.p(geometry, w_i, w_o, measure);
		if (components & kGlossyTransmission)	p_gt = m_glossy_trans.p(geometry, w_i, w_o, measure);

		return
			p_d		* w_p[kDiffuseReflectionIndex]	 +
			p_dt	* w_p[kDiffuseTransmissionIndex] +
			p_g		* w_p[kGlossyReflectionIndex]	 +
			p_gt	* w_p[kGlossyTransmissionIndex]; 
	}

	/// evaluate the sampling weights for the inner layers
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void sampling_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				V,
		float&								diffuse_refl_prob,
		float&								diffuse_trans_prob,
		float&								glossy_refl_prob,
		float&								glossy_trans_prob) const
	{
		const float NoV_signed = cugar::dot(geometry.normal_s, V);
		const float NoV        = fabsf( NoV_signed );

		cugar::Vector3f r_coeff;
		cugar::Vector3f t_coeff;

		if (m_ior == 0) // suppress the glossy layer
		{
			r_coeff = cugar::Vector3f(0.0f);
			t_coeff = cugar::Vector3f(1.0f);
		}
		else
		{
			r_coeff = glossy_reflectance( NoV_signed );
			t_coeff = cugar::Vector3f(1.0f - cugar::max_comp(r_coeff));
		}

	#if DIFFUSE_ONLY
		// disable Fresnel mixing - allow diffuse only
		r_coeff = cugar::Vector3f(0.0f);
		t_coeff = cugar::Vector3f(1.0f);
	#elif SPECULAR_ONLY
		// disable Fresnel mixing - allow specular only
		r_coeff = cugar::Vector3f(1.0f);
		t_coeff = cugar::Vector3f(0.0f);
	#endif
	#if SUPPRESS_SPECULAR
		r_coeff = cugar::Vector3f(0.0f);
	#endif
	#if SUPPRESS_DIFFUSE
		t_coeff = cugar::Vector3f(0.0f);
	#endif

		glossy_refl_prob	= cugar::max_comp( r_coeff );
		glossy_trans_prob	= (1 - m_opacity) * cugar::max_comp( t_coeff );
		diffuse_refl_prob	= m_opacity * cugar::max_comp( t_coeff * m_diffuse.color ) * M_PIf;
		diffuse_trans_prob	= m_opacity * cugar::max_comp( t_coeff * m_diffuse_trans.color ) * M_PIf;
	}

	/// evaluate the sampling weights for the inner layers
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void sampling_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				V,
		float*								w) const
	{
		sampling_weights( geometry, V, w[kDiffuseReflectionIndex], w[kDiffuseTransmissionIndex], w[kGlossyReflectionIndex], w[kGlossyTransmissionIndex] );
	}

	/// normalize the sampling weights of the inner layers given the coat sampling probability
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void normalize_sampling_weights(
		float*				w_p,
		float&				coat_reflection_prob,
		float&				coat_transmission_prob,
		const bool			RR,
		const ComponentType	components = kAllComponents) const
	{
		// set to zero unwanted components
		if ((components & kDiffuseReflection)	== 0)	w_p[kDiffuseReflectionIndex]	= 0;
		if ((components & kDiffuseTransmission)	== 0)	w_p[kDiffuseTransmission]		= 0;
		if ((components & kGlossyReflection)	== 0)	w_p[kGlossyReflection]			= 0;
		if ((components & kGlossyTransmission)	== 0)	w_p[kGlossyTransmission]		= 0;
		if ((components & kClearcoatReflection)	== 0) { coat_reflection_prob = 0; coat_transmission_prob = 1; }

		// modulate the sampling weights by the coat transmission probability
		w_p[kDiffuseReflectionIndex]	*= coat_transmission_prob;
		w_p[kDiffuseTransmissionIndex]	*= coat_transmission_prob;
		w_p[kGlossyReflectionIndex]		*= coat_transmission_prob;
		w_p[kGlossyTransmissionIndex]	*= coat_transmission_prob;

		if (RR == false)
		{
			const float inv_sum = 1.0f / (
				w_p[kDiffuseReflectionIndex]	+
				w_p[kDiffuseTransmissionIndex]	+
				w_p[kGlossyReflectionIndex]		+
				w_p[kGlossyTransmissionIndex]	+
				coat_reflection_prob);

			w_p[kDiffuseReflectionIndex]	*= inv_sum;
			w_p[kDiffuseTransmissionIndex]	*= inv_sum;
			w_p[kGlossyReflectionIndex]		*= inv_sum;
			w_p[kGlossyTransmissionIndex]   *= inv_sum;
			coat_reflection_prob			*= inv_sum;
			coat_transmission_prob			*= inv_sum;
		}
	}

	/// compute the Fresnel reflection and transmission weights
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void fresnel_weights(
		const float							VoH,
		const float							eta,
		cugar::Vector3f&					r_coeff,
		cugar::Vector3f&					t_coeff) const
	{
		if (eta == 0.0f) // NOTE: eta == m_ior == 0 signals the complete suppression of glossy reflections
		{
			r_coeff = cugar::Vector3f(0.0f);
			t_coeff = cugar::Vector3f(1.0f);
		}
		else
		{
			r_coeff = cugar::fresnel_schlick(VoH, eta, m_fresnel);
			t_coeff = cugar::Vector3f(1.0f - cugar::max_comp(r_coeff));
		}

	#if DIFFUSE_ONLY
		// disable Fresnel mixing - allow diffuse only
		r_coeff = cugar::Vector3f(0.0f);
		t_coeff = cugar::Vector3f(1.0f);
	#elif SPECULAR_ONLY
		// disable Fresnel mixing - allow specular only
		r_coeff = cugar::Vector3f(1.0f);
		t_coeff = cugar::Vector3f(0.0f);
	#endif
	#if SUPPRESS_SPECULAR
		r_coeff = cugar::Vector3f(0.0f);
	#endif
	#if SUPPRESS_DIFFUSE
		t_coeff = cugar::Vector3f(0.0f);
	#endif
	}

	/// compute the Fresnel reflection and transmission weights
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void fresnel_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				V,
		const cugar::Vector3f				L,
		cugar::Vector3f&					r_coeff,
		cugar::Vector3f&					t_coeff) const
	{
		float eta	  = 0.0f;
		float inv_eta = 0.0f;
		float VoH	  = 0.0f;

		if (m_ior)
		{
			const cugar::Vector3f N = geometry.normal_s;

			eta     = dot(N, V) > 0.0f ? 1.0f / m_ior : m_ior;
			inv_eta = dot(N, V) > 0.0f ? m_ior        : 1.0f / m_ior;

			// recover the microfacet H
			const cugar::Vector3f H = microfacet(V, L, N, inv_eta);

			VoH = dot(V,H);
		}
		// NOTE: eta == m_ior == 0 signals the complete suppression of glossy reflections

		fresnel_weights( VoH, eta, r_coeff, t_coeff );
	}
	
	/// <i>deprecated</i>
	/// evaluate the Fresnel weight for the glossy component - after the clearcoat
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void inner_component_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				V,
		const cugar::Vector3f				L,
		cugar::Vector3f&					diffuse_refl_coeff,
		cugar::Vector3f&					diffuse_trans_coeff,
		cugar::Vector3f&					glossy_refl_coeff,
		cugar::Vector3f&					glossy_trans_coeff) const
	{
		cugar::Vector3f w[4];
		inner_component_weights( geometry, V, L, w );

		glossy_refl_coeff   = w[kGlossyReflectionIndex];
		glossy_trans_coeff  = w[kGlossyTransmissionIndex];
		diffuse_refl_coeff  = w[kDiffuseReflectionIndex];
		diffuse_trans_coeff = w[kDiffuseTransmissionIndex];
	}

	/// evaluate the component weights of the inner layers - after the clearcoat
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void inner_component_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				V,
		const cugar::Vector3f				L,
		cugar::Vector3f*					w) const
	{
		cugar::Vector3f r_coeff, t_coeff;
		fresnel_weights( geometry, V, L, r_coeff, t_coeff );

		// compute an additional weight for the diffuse (or matte) component, necessary for energy conservation,
		// as explained in:
		//   "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling",
		//   Csaba Kelemen and László Szirmay-Kalos
		const float diffuse_weight =
			(1.0f - glossy_reflectance( dot(geometry.normal_s,V) )) *
			(1.0f - glossy_reflectance( dot(geometry.normal_s,L) ));

		w[kGlossyReflectionIndex]		= r_coeff;
		w[kGlossyTransmissionIndex]		= t_coeff * (1 - m_opacity);
		w[kDiffuseReflectionIndex]		= t_coeff * m_opacity * diffuse_weight;
		w[kDiffuseTransmissionIndex]	= t_coeff * m_opacity * diffuse_weight;
	}

	/// evaluate the component weights of all layers - including the clearcoat
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void component_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		cugar::Vector3f&					Fc_1,
		cugar::Vector3f&					Tc_1,
		cugar::Vector3f&					Fc_2,
		cugar::Vector3f&					Tc_2,
		cugar::Vector3f*					w) const
	{
		// calculate the transmitted incident vector seen from the inner layers
		cugar::Vector3f H_c;
		float			eta_c;
		float			cos_theta_i;

		if (!clearcoat_transmission(
			geometry,
			w_i,
			H_c,
			cos_theta_i,
			eta_c,
			Fc_1,
			Tc_1 ))
		{
			// total internal reflection - no chance to enter the clearcoat layer
			w[kGlossyReflectionIndex]		= 0.0f;
			w[kGlossyTransmissionIndex]		= 0.0f;
			w[kDiffuseReflectionIndex]		= 0.0f;
			w[kDiffuseTransmissionIndex]	= 0.0f;
			return;
		}

		// Here we blatantly ignore the second refraction to exit the clearcoat layer - as accounting for that explicitly
		// would lead to severe energy loss due to the lack of multiple-scattering.
		// A much better alternative we plan to adopt is the adding-doubling algorithm proposed by L.Belcour.
		Fc_2 = 0.0f;
		Tc_2 = 1.0f - Fc_2;

		inner_component_weights( geometry, w_i, w_o, w );

		w[kGlossyReflectionIndex]		*= Tc_1 * Tc_2;
		w[kGlossyTransmissionIndex]		*= Tc_1 * Tc_2;
		w[kDiffuseReflectionIndex]		*= Tc_1 * Tc_2;
		w[kDiffuseTransmissionIndex]	*= Tc_1 * Tc_2;
	}

	/// evaluate the component weights of all layers - including the clearcoat
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	void component_weights(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o,
		cugar::Vector3f*					w) const
	{
		cugar::Vector3f Fc_1;
		cugar::Vector3f Tc_1;
		cugar::Vector3f Fc_2;
		cugar::Vector3f Tc_2;

		component_weights( geometry, w_i, w_o, Fc_1, Tc_1, Fc_2, Tc_2, w );
	}

	/// sample a given component
	///
	/// \param geometry				the local differential geometry
	/// \param z					the input random numbers, relative to the given component
	/// \param in					the incoming direction
	/// \param out_comp				the given component
	/// \param out					the outgoing direction
	/// \param out_p				the output solid angle pdf
	/// \param out_p_proj			the output projected solid angle pdf
	/// \param out_g				the output sample value = f/p_proj
	/// \param RR					indicate whether to use Russian-Roulette or not
	/// \param evaluate_full_bsdf	ndicate whether to evaluate the full BSDF, or just an unbiased estimate
	/// \param components			the components to consider
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool sample_component(
		const cugar::DifferentialGeometry&	geometry,
		const float							z[3],
		const cugar::Vector3f				in,
		const ComponentType					out_comp,
		cugar::Vector3f&					out,
		float&								out_p,
		float&								out_p_proj,
		cugar::Vector3f&					out_g,
		const bool							evaluate_full_bsdf = false) const
	{
		cugar::Vector3f g(0.0f);
		float			p(0.0f);
		float           p_proj(0.0f);

		if (out_comp == kDiffuseReflection)
		{
			// sample the diffuse component
			const Bsdf::diffuse_component component = diffuse();

			component.sample(cugar::Vector3f(z[0], z[1], z[2]), geometry, in, out, g, p, p_proj);
		}
		else if (out_comp == kGlossyReflection)
		{
			// sample the glossy component
			const Bsdf::glossy_component component = glossy();

			component.sample(cugar::Vector3f(z[0], z[1], z[2]), geometry, in, out, g, p, p_proj);
		}
		else if (out_comp == kDiffuseTransmission)
		{
			// sample the diffuse component
			const Bsdf::diffuse_trans_component component = diffuse_trans();

			component.sample(cugar::Vector3f(z[0], z[1], z[2]), geometry, in, out, g, p, p_proj);
		}
		else if (out_comp == kGlossyTransmission)
		{
			// sample the glossy component
			const Bsdf::glossy_component& component = glossy_trans();

			component.sample(cugar::Vector3f(z[0], z[1], z[2]), geometry, in, out, g, p, p_proj);
		}

		if (out_comp != kAbsorption)
		{
			// re-evaluate the entire BSDF
			if (evaluate_full_bsdf)
			{
				g = this->f(geometry, in, out, kAllComponents) / p_proj;
			}
			else
			{
				// evaluate the true weights, now that the output direction is known
				cugar::Vector3f w[4];
				component_weights(geometry, in, out, w);

				// re-weight the bsdf value
				g *=
					(out_comp & kGlossyReflection)   ?	w[kGlossyReflectionIndex] :
					(out_comp & kGlossyTransmission) ?	w[kGlossyTransmissionIndex] :
					(out_comp & kDiffuseReflection)  ?	w[kDiffuseReflectionIndex] :
														w[kDiffuseTransmissionIndex];
			}

			const float factor = compression_factor( geometry, in, out );

			out_p		= p;
			out_p_proj	= p_proj;
			out_g		= g * factor;
			return true;
		}
		else
		{
			out			= cugar::Vector3f(0.0f);
			out_p		= 0.0f;
			out_p_proj	= 0.0f;
			out_g		= cugar::Vector3f(0.0f);
			return false;
		}
	}

	/// sample an outgoing direction
	///
	/// \param geometry				the local differential geometry
	/// \param z					the incoming direction
	/// \param in					the incoming direction
	/// \param out_comp				the output component
	/// \param out					the outgoing direction
	/// \param out_p				the output solid angle pdf
	/// \param out_p_proj			the output projected solid angle pdf
	/// \param out_g				the output sample value = f/p_proj
	/// \param RR					indicate whether to use Russian-Roulette or not
	/// \param evaluate_full_bsdf	ndicate whether to evaluate the full BSDF, or just an unbiased estimate
	/// \param components			the components to consider
	FERMAT_HOST_DEVICE FERMAT_FORCEINLINE
	bool sample(
		const cugar::DifferentialGeometry&	geometry,
		const float							z[3],
		const cugar::Vector3f				in,
		ComponentType&						out_comp,
		cugar::Vector3f&					out,
		float&								out_p,
		float&								out_p_proj,
		cugar::Vector3f&					out_g,
		bool								RR					= true,
		bool								evaluate_full_bsdf	= false,
		const ComponentType					components			= kAllComponents) const
	{
		cugar::Vector3f g(0.0f);
		float			p(0.0f);
		float           p_proj(0.0f);
		float			z_norm(0.0f);
		float			p_comp(0.0f);

		cugar::Vector3f w_i = in;
		cugar::Vector3f w_o;
		
		//
		// 1. compute the Fresnel term Fc due to the coating layer
		// 2. use Fc to derive sampling probabilities of all layers (i.e. multiplying their pdf by (1-Fc))
		//   2b. normalize all probabilities if RR is disabled 
		// 3. sample reflection from the coating layer with probability Fc
		// 4. choose an inner layer with the above probabilities
		// 5. refract the output from the selected inner layer through the coating layer
		//

		// We assume the clearcoat is actually made of two layers, C_o and C_i: one with a relative index of refraction (eta)
		// of eta with respect to the outer medium, and another adjacent one with a relative index of refraction
		// of eta with respect to the inner medium.
		// i.e. ior(C_o)/ior(O) = eta
		// i.e. ior(C_i)/ior(I) = eta
		// Now, this means that if the material's IOR specifies the ratio ior(I)/ior(O), we have:
		//
		// ior(C_i)/ior(C_o) = ior(I)/ior(O)
		//
		// We suppose the substrate GGX/diffuse layer sits at the surface between C_o and C_i, implying that
		// the transmissive component of that layer sees exactly the same ratio eta = ior(I)/ior(O).
		// In practice, this means that rays that go through all the layers will simply see a combined refraction
		// of eta.
		// In theory of course modeling the individual interactions would be significant for multiple scattering
		// within the medium, yet that would be rather expensive and require stochastic evaluation, e.g. using
		// the scheme proposed in:
		// "Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs", Yu Guo, Milos Hasan, Shuang Zhao.
		// A more convenient approach is the one taken by:
		// "Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators",
		// Laurent Belcour - in ACM Transactions on Graphics, Association for Computing Machinery, 2018.
		// While we plan to adopt the full machinery of the latter in the future, we currently just approximate it.

		cugar::Vector3f H_c;
		cugar::Vector3f Fc_1;
		cugar::Vector3f Tc_1;
		float			eta_c;
		float			cos_theta_i;

		if (!clearcoat_transmission(
			geometry,
			in,
			H_c,
			cos_theta_i,
			eta_c,
			Fc_1,
			Tc_1 ))
		{
			// total internal reflection - no chance to enter the clearcoat layer
			out			= cugar::Vector3f(0.0f);
			out_p		= 0.0f;
			out_p_proj	= 0.0f;
			out_g		= cugar::Vector3f(0.0f);
			return false;
		}

		float coat_reflection_prob   = cugar::average(Fc_1);
		float coat_transmission_prob = 1.0f - coat_reflection_prob;

		// evaluate the sampling weights for the inner layers
		float w_p[4];
		sampling_weights(geometry, in, w_p);

	#if USE_EFFICIENT_SAMPLER_WITH_APPROXIMATE_PDFS
		//
		// The following scheme is orders of magnitude more efficient at sampling the glossy/specular layer near normal incidence,
		// as the a-priori glossy sampling weights calculated using the *average* glossy reflectance tend to be way too low for high
		// values of cos(theta).
		//

		// sample the GGX microfacet distribution
		glossy_component::distribution_type glossy_distribution = m_glossy.distribution();
		
		// move to local coordinates
		const cugar::Vector3f V_local = geometry.to_local( w_i );

		const cugar::Vector3f H_local = glossy_distribution.sample( cugar::Vector3f(z[0], z[1], z[2]), V_local );

		// move to world coordinates
		const cugar::Vector3f H = geometry.from_local( H_local );

		// compute the Fresnel coefficients
		cugar::Vector3f r_coeff, t_coeff;

		// compute eta based on the surface side
		const float eta = V_local.z > 0.0f ? 1.0f / m_ior : m_ior;

		fresnel_weights( dot(V_local, H_local), eta, r_coeff, t_coeff );

		// assign component probabilities
		w_p[kGlossyReflectionIndex]		= (w_p[kGlossyReflectionIndex]		+	cugar::max_comp( r_coeff ) ) * 0.5f;
		w_p[kGlossyTransmissionIndex]	= (w_p[kGlossyTransmissionIndex]	+	(1 - m_opacity) * cugar::max_comp( t_coeff )) * 0.5f;
		w_p[kDiffuseReflectionIndex]	= (w_p[kDiffuseReflectionIndex]		+	m_opacity * cugar::max_comp( t_coeff * m_diffuse.color ) * M_PIf) * 0.5f;
		w_p[kDiffuseTransmissionIndex]	= (w_p[kDiffuseTransmissionIndex]	+	m_opacity * cugar::max_comp( t_coeff * m_diffuse_trans.color ) * M_PIf) * 0.5f;
	#endif

		// modulate the sampling weights by the cleacoat sampling probability
		normalize_sampling_weights( w_p, coat_reflection_prob, coat_transmission_prob, RR, components );

		// select a BSDF component with z[2]
		if (z[2] < w_p[kDiffuseReflectionIndex])
		{
			// sample the diffuse reflection component
			const Bsdf::diffuse_component component = diffuse();

			z_norm = z[2] / w_p[kDiffuseReflectionIndex];
			p_comp = w_p[kDiffuseReflectionIndex];

			component.sample(cugar::Vector3f(z[0], z[1], z_norm), geometry, w_i, w_o, g, p, p_proj);

			out_comp = kDiffuseReflection;
		}
		else if (z[2] < w_p[kDiffuseReflectionIndex] +
						w_p[kGlossyReflectionIndex])
		{
			// sample the glossy reflection component
			const Bsdf::glossy_component component = glossy();

			z_norm = (z[2] - w_p[kDiffuseReflectionIndex]) / w_p[kGlossyReflectionIndex];
			p_comp = w_p[kGlossyReflectionIndex];

		#if !USE_EFFICIENT_SAMPLER_WITH_APPROXIMATE_PDFS
			component.sample(cugar::Vector3f(z[0], z[1], z_norm), geometry, w_i, w_o, g, p, p_proj);
		#else
			// sample L given H
			component.sample( geometry, H, w_i, w_o, g, p, p_proj );
		#endif
			out_comp = kGlossyReflection;
		}
		else if (z[2] < w_p[kDiffuseReflectionIndex]	+
						w_p[kGlossyReflectionIndex]		+
						w_p[kDiffuseTransmissionIndex])
		{
			// sample the diffuse transmission component
			const Bsdf::diffuse_trans_component component = diffuse_trans();

			z_norm = (z[2] - w_p[kDiffuseReflectionIndex] - w_p[kGlossyReflectionIndex]) / w_p[kDiffuseTransmissionIndex];
			p_comp = w_p[kDiffuseTransmissionIndex];

			component.sample(cugar::Vector3f(z[0], z[1], z_norm), geometry, w_i, w_o, g, p, p_proj);

			out_comp = kDiffuseTransmission;
		}
		else if (z[2] <	w_p[kDiffuseReflectionIndex]	+
						w_p[kGlossyReflectionIndex]		+
						w_p[kDiffuseTransmissionIndex]	+
						w_p[kGlossyTransmissionIndex])
		{
			// sample the glossy transmission component
			const Bsdf::glossy_component& component = glossy_trans();

			z_norm = (z[2] - w_p[kDiffuseReflectionIndex] - w_p[kGlossyReflectionIndex] - w_p[kDiffuseTransmissionIndex]) / w_p[kGlossyTransmissionIndex];
			p_comp = w_p[kGlossyTransmissionIndex];

		#if !USE_EFFICIENT_SAMPLER_WITH_APPROXIMATE_PDFS
			component.sample(cugar::Vector3f(z[0], z[1], z_norm), geometry, w_i, w_o, g, p, p_proj);
		#else
			// sample L given H
			component.sample( geometry, H, w_i, w_o, g, p, p_proj );
		#endif

			out_comp = kGlossyTransmission;
		}
		else if (z[2] < w_p[kDiffuseReflectionIndex]	+
						w_p[kGlossyReflectionIndex]		+
						w_p[kDiffuseTransmissionIndex]	+
						w_p[kGlossyTransmissionIndex]	+
						coat_reflection_prob)
		{
			// sample the clearcoat reflection
			z_norm = (z[2] - w_p[kDiffuseReflectionIndex] - w_p[kGlossyReflectionIndex] - w_p[kDiffuseTransmissionIndex] - w_p[kGlossyTransmissionIndex]) / coat_reflection_prob;
			p_comp = coat_reflection_prob;

			// compute the reflection direction
			out    = 2 * cos_theta_i * H_c - in;
			g      = Fc_1 / p_comp;					// NOTE: we pre-divide by p_comp here, unlike for other components
			p_proj = cugar::float_infinity();
			p      = cugar::float_infinity();

			out_comp = kClearcoatReflection;
		}
		else
		{
			out_comp = kAbsorption;
		}

		if (out_comp != kAbsorption && 
			out_comp != kClearcoatReflection)
		{
			// Here we blatantly ignore the second refraction to exit the clearcoat layer - as accounting for that explicitly
			// would lead to severe energy loss due to the lack of multiple-scattering.
			// A much better alternative we plan to adopt is the adding-doubling algorithm proposed by L.Belcour.
			cugar::Vector3f Fc_2 = 0.0f;
			cugar::Vector3f Tc_2 = 1.0f - Fc_2;

			// modulate g by both clearcoat transmission terms
			g *= Tc_1 * Tc_2;

			// write out the output direction
			out = w_o;
		}

		// TODO: chose the channel based on our new BSDF factorization heuristic
		if (out_comp != kAbsorption)
		{
			// re-evaluate the entire BSDF and its sampling probabilities
			if (out_comp != kClearcoatReflection)
			{
				if (evaluate_full_bsdf)
				{
					float p_d	= (components & kDiffuseReflection)		?       diffuse().p(geometry, in, out, cugar::kProjectedSolidAngle) : 0.0f;
					float p_dt	= (components & kDiffuseTransmission)	? diffuse_trans().p(geometry, in, out, cugar::kProjectedSolidAngle) : 0.0f;
					float p_g	= (components & kGlossyReflection)		?        glossy().p(geometry, in, out, cugar::kProjectedSolidAngle) : 0.0f;
					float p_gt	= (components & kGlossyTransmission)	?  glossy_trans().p(geometry, in, out, cugar::kProjectedSolidAngle) : 0.0f;

					p_proj = 
						p_d  * w_p[kDiffuseReflectionIndex]	* coat_transmission_prob	+
						p_dt * w_p[kDiffuseTransmissionIndex]	* coat_transmission_prob	+
						p_g  * w_p[kGlossyReflectionIndex]		* coat_transmission_prob	+
						p_gt * w_p[kGlossyTransmissionIndex]	* coat_transmission_prob;

					p		= p_proj * fabsf( dot(out, geometry.normal_s) );
					g		= this->f(geometry, in, out, components) / p_proj;
				}
				else
				{
					// evaluate the true weights, now that the output direction is known
					cugar::Vector3f w[4];
					inner_component_weights(geometry, in, out, w);

					// re-weight the bsdf value
					g *=
						(out_comp & kGlossyReflection)   ?	w[kGlossyReflectionIndex] :
						(out_comp & kGlossyTransmission) ?	w[kGlossyTransmissionIndex] :
						(out_comp & kDiffuseReflection)  ?	w[kDiffuseReflectionIndex] :
															w[kDiffuseTransmissionIndex];

					g		/= p_comp;
					p		*= p_comp;
					p_proj	*= p_comp;
				}
			}

			const float factor = compression_factor( geometry, in, out );

			out_p		= p;
			out_p_proj	= p_proj;
			out_g		= g * factor;
			return true;
		}
		else
		{
			out			= cugar::Vector3f(0.0f);
			out_p		= 0.0f;
			out_p_proj	= 0.0f;
			out_g		= cugar::Vector3f(0.0f);
			return false;
		}
	}

	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	bool clearcoat_transmission(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f	w_i,
		cugar::Vector3f&		H,					// the output clearcoat microfacet
		float&					cos_theta_i,		// the output cosine of the incident angle - dot(w_i,H)
		float&					eta,				// the output clearcoat relative IOR
		cugar::Vector3f&		Fc_1,				// the output Fresnel reflection coefficient - R12
		cugar::Vector3f&		Tc_1) const			// the output Fresnel transmission coefficient - T12
	{
		const float R0			= cugar::min( cugar::max_comp( m_reflectivity ), 0.95f );
		const float eta_c		= 1.0f / m_clearcoat_ior;

		H			= geometry.normal_s; // for now, assume the clearcoat to be perfectly specular
		cos_theta_i = dot(w_i, H);

		// compute the refracted direction - which will be the input to the internal layers
		cugar::Vector3f w_t;
		float Fc_1_s;
		if (!cugar::refract(w_i, H, cos_theta_i, eta_c, &w_t, &Fc_1_s))
		{
			// total internal reflection - no chance to enter the clearcoat layer
			Fc_1 = 1.0f;
			Tc_1 = 0.0f;
			return false;
		}

		// use the true Fresnel coefficient to interpolate between m_reflectivity at normal incidence and pure white at grazing angles
		Fc_1 = cugar::lerp( m_reflectivity, cugar::Vector3f(1.0f), cugar::max(Fc_1_s - R0, 0.0f) / (1 - R0) );
		Tc_1 = 1.0f - Fc_1;
		return true;
	}

	/// apply the radiance compression factor of (eta_t / eta_i)^2
	///
	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	float compression_factor(
		const cugar::DifferentialGeometry&	geometry,
		const cugar::Vector3f				w_i,
		const cugar::Vector3f				w_o) const
	{
		if (m_transport == kRadianceTransport && m_ior)
		{
			// apply the radiance compression factor of (eta_t / eta_i)^2
			const float NoV = dot(w_i, geometry.normal_s);
			const float NoL = dot(w_o, geometry.normal_s);
			if (NoV * NoL < 0.0f)
				return cugar::sqr( NoV > 0.0f ? m_ior : 1.0f / m_ior );
		}
		return 1.0f;
	}

	FERMAT_FORCEINLINE FERMAT_HOST_DEVICE
	float glossy_reflectance(const float cos_theta) const
	{
		const uint32 S = 32;

		const float eta = cos_theta > 0.0f ?  1.0f / m_ior : m_ior;

		const uint32 cos_theta_i = cugar::min( S-1u, uint32( fabsf(cos_theta) * (S-1) ) );
		const uint32 base_spec_i = cugar::min( S-1u, uint32( cugar::max_comp( m_fresnel ) * (S-1) ) );
		const uint32 eta_i       = cugar::min( S-1u, uint32( (eta / 2.0f) * (S-1) ) );
		const uint32 roughness_i = cugar::min( S-1u, uint32( m_glossy.roughness * (S-1) ) );

		const uint32 cell_i = (eta_i*S*S*S + base_spec_i*S*S + roughness_i*S + cos_theta_i);

		return m_glossy_reflectance[cell_i];
	}

	diffuse_component		m_diffuse;
	diffuse_trans_component m_diffuse_trans;
	glossy_component		m_glossy;
	glossy_component		m_glossy_trans;
	cugar::Vector3f			m_fresnel;
	cugar::Vector3f			m_reflectivity;	// normal-incidence reflectivity of the clear-coat layer
	float					m_ior;
	float					m_opacity;
	float					m_clearcoat_ior;
	const float*			m_glossy_reflectance;
	TransportType			m_transport;
};


///@} BSDFModule
///@} Fermat
