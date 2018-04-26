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

#include <cugar/basic/functors.h>
#include <cugar/basic/algorithms.h>
#include <cugar/basic/utils.h>
#include <stack>

namespace cugar {
namespace bintree {

    struct Split_task
    {
        CUGAR_HOST_DEVICE Split_task() {}
        CUGAR_HOST_DEVICE Split_task(const uint32 id, const uint32 begin, const uint32 end, const uint32 level, const uint32 parent)
            : m_node( id ), m_begin( begin ), m_end( end ), m_parent( parent ), m_level( level ) {}

        uint32 m_node;
        uint32 m_begin;
        uint32 m_end;
        uint32 m_parent : 26;
        uint32 m_level  : 6;
    };

    // find the most significant bit smaller than start by which code0 and code1 differ
    template <typename Integer>
    CUGAR_FORCEINLINE CUGAR_HOST_DEVICE int32 find_leading_bit_difference(
        const  int32  start_level,
        const Integer code0,
        const Integer code1)
    {
        int32 level = start_level;

        while (level >= 0)
        {
            const Integer mask = Integer(1u) << level;

            if ((code0 & mask) !=
                (code1 & mask))
                break;

            --level;
        }
        return level;
    }

} // namespace bintree

template <typename Tree, typename Integer>
void generate_radix_tree(
    const uint32   n_codes,
    const Integer* codes,
    const uint32   bits,
    const uint32   max_leaf_size,
    const bool     keep_singletons,
	const bool     middle_splits,
    Tree&          tree)
{
    typedef bintree::Split_task Split_task;

    tree.reserve_nodes( n_codes * 2 );
    tree.reserve_leaves( n_codes );

    std::vector<Split_task> task_queues[2];
    std::vector<uint32>     skip_queues[2];

    task_queues[0].push_back( Split_task(0,0,n_codes,bits-1,uint32(-1)) );
    skip_queues[0].push_back(  uint32(-1) );

    uint32 node_count = 1;
    uint32 leaf_count = 0;

    typename Tree::context_type context = tree.get_context();

    uint32 in_queue  = 0;
    uint32 out_queue = 1;

    while (task_queues[ in_queue ].size())
    {
        task_queues[ out_queue ].erase(
            task_queues[ out_queue ].begin(),
            task_queues[ out_queue ].end() );
            
        for (uint32 task_id = 0; task_id < task_queues[ in_queue ].size(); ++task_id)
        {
            const Split_task task  = task_queues[ in_queue ][ task_id ];
            const uint32 skip_node = skip_queues[ in_queue ][ task_id ];

            const uint32 node   = task.m_node;
            const uint32 begin  = task.m_begin;
            const uint32 end    = task.m_end;
                  uint32 level  = task.m_level;
                  uint32 parent = task.m_parent;

            if (!keep_singletons)
            {
                level = bintree::find_leading_bit_difference(
                    level,
                    codes[ begin ],
                    codes[ end-1 ] );
            }

            uint32 output_count = 0;
            uint32 split_index;

            // check whether the input node really needs to be split
            if (end - begin > max_leaf_size && level != uint32(-1))
            {
                // find the "partitioning pivot" using a binary search
                split_index = find_pivot(
                    codes + begin,
                    end - begin,
                    mask_and<Integer>( Integer(1u) << level ) ) - codes;

                output_count = (split_index == begin || split_index == end) ? 1u : 2u;
            }

            const uint32 node_offset = node_count;
            const uint32 first_end   = (output_count == 1) ? end       : split_index;
            const uint32 first_skip  = (output_count == 1) ? skip_node : node_offset+1;

            if (output_count >= 1) { task_queues[ out_queue ].push_back( Split_task( node_offset+0, begin, first_end, level-1, node ) ); skip_queues[ out_queue ].push_back( first_skip ); }
            if (output_count == 2) { task_queues[ out_queue ].push_back( Split_task( node_offset+1, split_index, end, level-1, node ) ); skip_queues[ out_queue ].push_back( skip_node ); }

            const bool generate_leaf = output_count == 0;

            // count how many leaves we need to generate
            const uint32 leaf_index = leaf_count;

            node_count += output_count;
            leaf_count += generate_leaf ? 1 : 0;

            // write the parent node
            context.write_node(
                node,
                parent,
                output_count ? split_index != begin : false,
                output_count ? split_index != end   : false,
                output_count ? node_offset          : leaf_index,
                skip_node,
                level,
                begin,
                end,
                output_count ? split_index : uint32(-1) );

            // make a leaf if necessary
            if (generate_leaf)
                context.write_leaf( leaf_index, node, begin, end );
        }

        std::swap( in_queue, out_queue );
    }
}

} // namespace cugar
