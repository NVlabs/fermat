/*
 * Fermat
 *
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <uv_bvh_view.h>
#include <cugar/bvh/bvh.h>
#include <cugar/basic/vector.h>

/// A 2d UV-space BVH, useful for finding out the triangle covering a given UV coordinate
///
template <typename domain_tag>
struct UVBvh
{
    UVBvh() {}

    template <typename U>
    UVBvh(UVBvh<U>& bvh) : nodes(bvh.nodes), bboxes(bvh.bboxes), index(bvh.index) {}

    UVBvhView view() const { return UVBvhView( cugar::raw_pointer(nodes), cugar::raw_pointer(bboxes), cugar::raw_pointer(index) ); }

	cugar::vector<domain_tag, UVBvh_node>		nodes;
    cugar::vector<domain_tag, cugar::Bbox2f>    bboxes;
	cugar::vector<domain_tag, uint32_t>	        index;
};

typedef UVBvh<cugar::host_tag>    HostUVBvh;
typedef UVBvh<cugar::device_tag>  DeviceUVBvh;

void build(HostUVBvh* bvh, const MeshStorage& mesh);
void build(HostUVBvh* bvh, const cugar::vector<cugar::host_tag,VTL>& vtls);

void output_uv_tris(const MeshStorage& mesh);
