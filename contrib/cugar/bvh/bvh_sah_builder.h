/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
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

#pragma once

#include <cugar/bvh/bvh.h>
#include <algorithm>
#include <stack>
#include <limits>

namespace cugar {

///@addtogroup bvh
///@{

///
/// An SAH-based bvh builder for 3d bboxes
///
class Bvh_sah_builder
{
public:
    typedef Vector3f            vector_type;
    typedef Bbox3f              bbox_type;
    typedef Bvh<3u>             bvh_type;
    typedef Bvh_node            bvh_node_type;

    struct Stats
    {
        uint32 m_max_depth;
    };

    /// constructor
    Bvh_sah_builder() :
        m_max_leaf_size( 4u ),
        m_force_splitting( true ),
        m_force_alignment( false ),
        m_single_axis_threshold( 1024*1024*64 ) {}

    /// set bvh parameters
    void set_max_leaf_size(const uint32 max_leaf_size) { m_max_leaf_size = max_leaf_size; }

    /// set force splitting
    void set_force_splitting(const bool flag) { m_force_splitting = flag; }

    /// set force 'max leaf size'-aligned splits
    void set_force_alignment(const bool flag) { m_force_alignment = flag; }

    /// set single axis test threshold
    void set_single_axis_threshold(const uint32 v) { m_single_axis_threshold = v; }

    /// build
    ///
    /// Iterator is supposed to dereference to a Bbox3f
    ///
    /// \param begin            first point
    /// \param end              last point
    /// \param bvh              output bvh
    template <typename Iterator>
    void build(
        Iterator        begin,
        Iterator        end,
        bvh_type*       bvh,
        Stats*          stats = NULL);

    /// build
    ///
    /// Iterator is supposed to dereference to a Bbox3f
    ///
    /// \param begin            first point
    /// \param end              last point
    /// \param bvh              output bvh
    template <typename Iterator, typename CostIterator>
    void build(
        Iterator        begin,
        Iterator        end,
        CostIterator    cost_begin,
        bvh_type*       bvh,
        Stats*          stats = NULL);

    /// remapped point index
    uint32 index(const uint32 i) const { return m_entities[m_indices[0][i]].m_index; }

private:
    void build(
        const uint32    n_entities,
        bvh_type*       bvh,
        Stats*          stats);

    class Predicate;
    class IndexSortPredicate;

    struct Entity
    {
        bbox_type   m_bbox;
        uint32      m_index;
        float       m_cost;
    };

    struct Node
    {
        bbox_type m_bbox;
        uint32    m_begin;
        uint32    m_end;
        uint32    m_node;
        uint32    m_depth;
    };
    typedef std::stack<Node> Node_stack;

    float area(const Bbox3f& bbox);

    void compute_bbox(
        const uint32      begin,
        const uint32      end,
        bbox_type&          bbox);

    bool find_largest_axis_split(
        const Node& node,
        uint32&   pivot,
        bbox_type&  l_bb,
        bbox_type&  r_bb);

    bool find_best_split(
        const Node& node,
        uint32&   pivot,
        bbox_type&  l_bb,
        bbox_type&  r_bb);

    struct Bvh_partitioner;

    uint32                  m_max_leaf_size;
    bool                    m_force_splitting;
    bool                    m_force_alignment;
    uint32                  m_single_axis_threshold;
    std::vector<int>        m_indices[3];           // indices into entities, one vector per dimension
    std::vector<Entity>     m_entities;
    std::vector<bbox_type>  m_bboxes;
    std::vector<float>      m_costs;
    std::vector<int>        m_tmpInt;               // temp array used during build
    std::vector<uint8>      m_tag_bits;
};

///@}  bvh

namespace deprecated {

///
/// An SAH-based bvh builder for 3d bboxes
///
class Bvh_sah_builder
{
public:
    typedef Vector3f            vector_type;
    typedef Bbox3f              bbox_type;
    typedef Bvh<3u>             bvh_type;
    typedef Bvh_node            bvh_node_type;

    struct Stats
    {
        uint32 m_max_depth;
    };

    /// constructor
    Bvh_sah_builder() :
        m_max_leaf_size( 4u ),
        m_force_splitting( true ),
        m_force_alignment( false ),
        m_partial_build( false ),
        m_single_axis_threshold( 1024*1024*64 ) {}

    /// set bvh parameters
    void set_max_leaf_size(const uint32 max_leaf_size) { m_max_leaf_size = max_leaf_size; }

    /// set force splitting
    void set_force_splitting(const bool flag) { m_force_splitting = flag; }

    /// set force 'max leaf size'-aligned splits
    void set_force_alignment(const bool flag) { m_force_alignment = flag; }

    /// set partial build
    void set_partial_build(const bool flag) { m_partial_build = flag; }

    /// set single axis test threshold
    void set_single_axis_threshold(const uint32 v) { m_single_axis_threshold = v; }

    /// build
    ///
    /// Iterator is supposed to dereference to a Bbox3f
    ///
    /// \param begin            first point
    /// \param end              last point
    /// \param bvh              output bvh
    template <typename Iterator>
    void build(
        Iterator        begin,
        Iterator        end,
        Bvh_type*       bvh);

    /// remapped point index
    uint32 index(const uint32 i) const { return m_entities[i].m_index; }

private:
    class Predicate;
    class IndexSortPredicate;

    struct Entity
    {
        bbox_type   m_bbox;
        uint32      m_index;
    };

    struct Node
    {
        bbox_type m_bbox;
        uint32    m_begin;
        uint32    m_end;
        uint32    m_node;
        uint32    m_depth;
    };
    typedef std::stack<Node> Node_stack;

    float area(const Bbox3f& bbox);

    void compute_bbox(
        const uint32      begin,
        const uint32      end,
        bbox_type&          bbox);

    bool find_best_split(
        const Node& node,
        uint32&   pivot,
        bbox_type&  l_bb,
        bbox_type&  r_bb);

    struct Bvh_partitioner;

    uint32                  m_max_leaf_size;
    bool                    m_force_splitting;
    bool                    m_force_alignment;
    bool                    m_partial_build;
    uint32                  m_single_axis_threshold;
    std::vector<Entity>     m_entities;
    std::vector<bbox_type>  m_bboxes;
};

} // namespace deprecated

} // namespace cugar

#include <cugar/bvh/bvh_sah_builder_inline.h>
