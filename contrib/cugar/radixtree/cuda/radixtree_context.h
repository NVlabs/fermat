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

/*! \file radixtree_context.h
 *   \brief Defines the context class for the binary tree generate() function.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/vector.h>

namespace cugar {
namespace cuda {

///@addtogroup bintree
///@{

///@addtogroup radixtree
///@{

///
/// A context class for generate_radix_tree() function.
///
struct Radixtree_context
{
    struct Split_task
    {
        CUGAR_HOST_DEVICE Split_task() {}
        CUGAR_HOST_DEVICE Split_task(const uint32 parent, const uint32 begin, const uint32 end, const uint32 level)
            : m_parent( parent ), m_begin( begin ), m_end( end ), m_level( level ) {}
        CUGAR_HOST_DEVICE Split_task(const uint4 u)
            : m_parent(u.x), m_begin(u.y), m_end(u.z), m_level(u.w) {}

        CUGAR_HOST_DEVICE operator uint4() const { return make_uint4(m_parent,m_begin,m_end,m_level); }

        uint32 m_parent;
        uint32 m_begin;
        uint32 m_end;
        uint32 m_level;
    };
    struct Counters
    {
        uint32 node_counter;
        uint32 leaf_counter;
        uint32 task_counter;
        uint32 work_counter;
    };

    caching_device_vector<Split_task>   m_task_queues;
    caching_device_vector<Counters>     m_counters;
    caching_device_vector<uint32>       m_skip_nodes;
    uint32                              m_nodes;
    uint32                              m_leaves;
};

///@} radixtree
///@} bintree

} // namespace cuda
} // namespace cugar
