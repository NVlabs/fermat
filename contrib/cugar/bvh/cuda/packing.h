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

/*! \file packing.h
 *   \brief Defines utility functions to pack a set of bvh nodes and their
 *          bboxes into a single set of 4d bboxes.
 */

#pragma once

#include <cugar/bvh/bvh.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugar {
namespace cuda {

/*! \addtogroup bvh Boundary Volume Hierarchies
 *  \{
 */

/// utility functor to pack a tuple formed by a (4d) bbox and a node into
/// a single bbox.
struct bvh_packing_functor
{
    typedef Bbox4f                         result_type;
    typedef thrust::tuple<Bbox4f,Bvh_node> argument_type;

    CUGAR_HOST_DEVICE Bbox4f operator() (const argument_type arg) const
    {
        Bbox4f bbox = thrust::get<0>(arg);
        bbox[0][3] = binary_cast<float>( thrust::get<1>(arg).m_packed_data );
        bbox[1][3] = binary_cast<float>( thrust::get<1>(arg).m_skip_node );
        return bbox;
    }
};

///
/// Pack a set of 3d bvh nodes and their bboxes into a single set of (4d) bboxes,
/// where the 3rd components of the bbox min and max are used to pack the tree
/// topology.
/// The input and output bbox arrays can be the same.
///
/// \param n_nodes          node count
/// \param nodes            input nodes
/// \param bboxes           input bboxes
/// \param packed_nodes     output bboxes
///
/// The following code snipped illustrates an example usage:
///
/// \code
///
/// thrust::device_vector<Bvh_node> bvh_nodes;
/// thrust::device_vector<Bbox4f>   bvh_bboxes;
/// uint32 node_count;
/// ... // build a bvh and compute its bboxes here
///
/// // pack the bvh
/// cuda::pack(
///     node_count,
///     bvh_nodes.begin(),
///     bvh_bboxes.begin(),
///     bvh_bboxes.begin() );
///
/// \endcode
///
template <typename Node_iterator, typename Bbox_iterator, typename Output_iterator>
void pack(
    const uint32    n_nodes,
    Node_iterator   nodes,
    Bbox_iterator   bboxes,
    Output_iterator packed_nodes)
{
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(bboxes,nodes)),
        thrust::make_zip_iterator(thrust::make_tuple(bboxes,nodes)) + n_nodes,
        packed_nodes,
        bvh_packing_functor() );
}

/*! \}
 */

} // namespace cuda
} // namespace cugar
