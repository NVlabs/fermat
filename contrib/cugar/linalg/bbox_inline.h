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

namespace cugar {

template <typename Vector_t>
Bbox<Vector_t>::Bbox() :
	m_min( field_traits<value_type>::max() ),
	m_max( field_traits<value_type>::min() )
{
}
template <typename Vector_t>
Bbox<Vector_t>::Bbox(const Vector_t& v) :
	m_min( v ),
	m_max( v )
{
}
template <typename Vector_t>
Bbox<Vector_t>::Bbox(const Vector_t& v1, const Vector_t& v2) :
	m_min( v1 ),
	m_max( v2 )
{
}
template <typename Vector_t>
Bbox<Vector_t>::Bbox(const Bbox<Vector_t>& bb1, const Bbox<Vector_t>& bb2)
{
	for (uint32 i = 0; i < m_min.dimension(); i++)
	{
        m_min[i] = ::cugar::min( bb1[0][i], bb2[0][i] );
		m_max[i] = ::cugar::max( bb1[1][i], bb2[1][i] );
	}
}
template <typename Vector_t>
Bbox<Vector_t>::Bbox(const Bbox<Vector_t>& bb) :
    m_min( bb.m_min ),
    m_max( bb.m_max )
{
}

template <typename Vector_t>
void Bbox<Vector_t>::insert(
	const Vector_t& v)
{
	for (uint32 i = 0; i < m_min.dimension(); i++)
	{
		m_min[i] = ::cugar::min( m_min[i], v[i] );
		m_max[i] = ::cugar::max( m_max[i], v[i] );
	}
}
template <typename Vector_t>
void Bbox<Vector_t>::insert(
	const Bbox& bbox)
{
	for (uint32 i = 0; i < m_min.dimension(); i++)
	{
        m_min[i] = ::cugar::min( m_min[i], bbox.m_min[i] );
		m_max[i] = ::cugar::max( m_max[i], bbox.m_max[i] );
	}
}
template <typename Vector_t>
void Bbox<Vector_t>::clear()
{
	for (uint32 i = 0; i < m_min.dimension(); i++)
	{
		m_min[i] = field_traits<value_type>::max();
		m_max[i] = field_traits<value_type>::min();
	}
}

template <typename Vector_t>
Bbox<Vector_t>& Bbox<Vector_t>::operator=(const Bbox<Vector_t>& bb)
{
    m_min = bb.m_min;
    m_max = bb.m_max;
    return *this;
}

template <typename Vector_t>
uint32 largest_axis(const Bbox<Vector_t>& bbox)
{
	typedef typename Vector_t::value_type value_type;

	const Vector_t edge( bbox[1] - bbox[0] );
	uint32 axis = 0;
	value_type v = edge[0];

	for (uint32 i = 1; i < edge.dimension(); i++)
	{
		if (v < edge[i])
		{
			v = edge[i];
			axis = i;
		}
	}
	return axis;
}

/// returns the delta between corners
///
/// \param bbox     bbox object
template <typename Vector_t>
CUGAR_FORCEINLINE CUGAR_HOST_DEVICE Vector_t extents(const Bbox<Vector_t>& bbox)
{
    return bbox[1] - bbox[0];
}

// compute the area of a 2d bbox
//
// \param bbox     bbox object
float area(const Bbox2f& bbox)
{
    const Vector2f edge = bbox[1] - bbox[0];
    return edge[0] * edge[1];
}

// compute the area of a 3d bbox
//
// \param bbox     bbox object
float area(const Bbox3f& bbox)
{
    const Vector3f edge = bbox[1] - bbox[0];
    return edge[0] * edge[1] + edge[2] * (edge[0] + edge[1]);
}

// point-in-bbox inclusion predicate
//
// \param bbox     bbox object
// \param p        point to test for inclusion
template <typename Vector_t>
bool contains(const Bbox<Vector_t>& bbox, const Vector_t& p)
{
    for (uint32 i = 0; i < p.dimension(); ++i)
    {
        if (p[i] < bbox[0][i] ||
            p[i] > bbox[1][i])
            return false;
    }
    return true;
}

// bbox-in-bbox inclusion predicate
//
// \param bbox     bbox object
// \param c        candidate to test for inclusion
template <typename Vector_t>
bool contains(const Bbox<Vector_t>& bbox, const Bbox<Vector_t>& c)
{
    for (uint32 i = 0; i < c[0].dimension(); ++i)
    {
        if (c[0][i] < bbox[0][i] ||
            c[1][i] > bbox[1][i])
            return false;
    }
    return true;
}

// point-to-bbox squared distance
//
// \param bbox     bbox object
// \param p        point
template <typename Vector_t>
float sq_distance(const Bbox<Vector_t>& bbox, const Vector_t& p)
{
    float r = 0.0f;
    for (uint32 i = 0; i < p.dimension(); ++i)
    {
        const float dist = cugar::max( bbox[0][i] - p[i], 0.0f ) +
                           cugar::max( p[i] - bbox[1][i], 0.0f );

        r += dist*dist;
    }
    return r;
}

} // namespace cugar
