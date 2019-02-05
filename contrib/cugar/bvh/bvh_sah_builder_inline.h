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

//
// IndexSortPredicate implementation.
//

// Predicate to sort indices based on their entity's AABB center.
class Bvh_sah_builder::IndexSortPredicate
{
public:
    IndexSortPredicate(const std::vector<Entity>& entities, int dim)
      : m_entities( entities ),
        m_dim( dim )
    {}

    bool operator() (int lhs, int rhs) const
    {
        const Bbox3f& left  = m_entities[lhs].m_bbox;
        const Bbox3f& right = m_entities[rhs].m_bbox;
        const float leftval  =  left.m_min[m_dim]  + left.m_max[m_dim];
        const float rightval = right.m_min[m_dim] + right.m_max[m_dim];
        return leftval < rightval;
    }

private:
    const std::vector<Entity>&  m_entities;
    int                         m_dim;
};

inline void Bvh_sah_builder::build(
    const uint32    n_entities,
    bvh_type*       bvh,
    Stats*          stats)
{
   std::vector<bvh_node_type>& nodes = bvh->m_nodes;

    // initialize tag bits
    m_tag_bits.resize( n_entities, false );

    // Initialize indices.
    m_tmpInt.resize( n_entities );

    m_indices[0].resize( n_entities );
    for(uint32 i = 0; i < n_entities; ++i)
        m_indices[0][i] = static_cast<int>( i );

    for(int dim = 1; dim < 3; ++dim)
        m_indices[dim] = m_indices[0];

    // Sort index arrays by their dimension.
    IndexSortPredicate pred_x( m_entities, 0 );
    IndexSortPredicate pred_y( m_entities, 1 );
    IndexSortPredicate pred_z( m_entities, 2 );
    std::sort( m_indices[0].begin(), m_indices[0].end(), pred_x );
    std::sort( m_indices[1].begin(), m_indices[1].end(), pred_y );
    std::sort( m_indices[2].begin(), m_indices[2].end(), pred_z );

    // create root
    Bbox3f root_bbox;
    compute_bbox( 0u, n_entities, root_bbox );

    Node_stack node_stack;
    {
        Node node;
        node.m_bbox  = root_bbox;
        node.m_begin = 0u;
        node.m_end   = n_entities;
        node.m_depth = 0u;
        node.m_node  = 0u;

        node_stack.push( node );
    }

    if (stats)
        stats->m_max_depth = 0u;

    bvh_node_type root = bvh_node_type( Bvh_node::kInternal, 0u, uint32(-1) );
    nodes.erase( nodes.begin(), nodes.end() );
    nodes.push_back( root );
    bvh->m_bboxes.resize( 1u );
    bvh->m_bboxes[0] = root_bbox;

    while (node_stack.empty() == false)
    {
        const Node node = node_stack.top();
        node_stack.pop();

        const uint32 node_index = node.m_node;
        const bbox_type bbox    = node.m_bbox;

        // set the node bbox
        bvh->m_bboxes[ node_index ] = bbox;

        if (stats)
            stats->m_max_depth = std::max( stats->m_max_depth, node.m_depth );

        if (node.m_end - node.m_begin <= m_max_leaf_size)
        {
            //
            // Make a leaf node
            //
            bvh_node_type& bvh_node = nodes[ node_index ];
            bvh_node = bvh_node_type( node.m_begin, node.m_end );
        }
        else
        {
            //
            // Make a split node
            //

            uint32    middle;
            bbox_type l_bb, r_bb;

            if (find_best_split( node, middle, l_bb, r_bb ) == false)
            {
                if (m_force_splitting)
                {
                    middle = (node.m_begin + node.m_end) / 2;
                    compute_bbox( node.m_begin, middle, l_bb );
                    compute_bbox( middle, node.m_end, r_bb );
                }
                else
                {
                    // unsuccessful split: make a leaf node
                    bvh_node_type& bvh_node = nodes[ node_index ];
                    bvh_node = bvh_node_type( node.m_begin, node.m_end );
                    continue;
                }
            }

            // alloc space for children
            const uint32 left_node_index = uint32( nodes.size() );
            nodes.resize( left_node_index + 2 );
            bvh->m_bboxes.resize( left_node_index + 2 );

            bvh_node_type& bvh_node = nodes[ node_index ];
            bvh_node = bvh_node_type( left_node_index );

            // push left and right children in processing queue
            Node right_node;
            right_node.m_bbox  = r_bb;
            right_node.m_begin = middle;
            right_node.m_end   = node.m_end;
            right_node.m_depth = node.m_depth + 1u;
            right_node.m_node  = left_node_index + 1u;
            node_stack.push( right_node );

            Node left_node;
            left_node.m_bbox  = l_bb;
            left_node.m_begin = node.m_begin;
            left_node.m_end   = middle;
            left_node.m_depth = node.m_depth + 1u;
            left_node.m_node  = left_node_index;
            node_stack.push( left_node );
        }
    }
}

template <typename Iterator>
void Bvh_sah_builder::build(
    const Iterator  begin,
    const Iterator  end,
    bvh_type*       bvh,
    Stats*          stats)
{
    const uint32 n_entities = uint32( end - begin );

    m_entities.resize( n_entities );
    uint32 i = 0;
    for (Iterator it = begin; it != end; ++it)
    {
        m_entities[i].m_bbox  = *it;
        m_entities[i].m_index = i;
        m_entities[i].m_cost  = 1.0f;

        i++;
    }

    build( n_entities, bvh, stats );
}
// build
//
// Iterator is supposed to dereference to a Bbox3f
//
// \param begin            first point
// \param end              last point
// \param bvh              output bvh
template <typename Iterator, typename CostIterator>
void Bvh_sah_builder::build(
    Iterator        begin,
    Iterator        end,
    CostIterator    cost_begin,
    bvh_type*       bvh,
    Stats*          stats)
{
    const uint32 n_entities = uint32( end - begin );

    m_entities.resize( n_entities );
    uint32 i = 0;
    for (Iterator it = begin; it != end; ++it)
    {
        m_entities[i].m_bbox  = *it;
        m_entities[i].m_index = i;
        m_entities[i].m_cost  = cost_begin[i];

        i++;
    }

    build( n_entities, bvh, stats );
}

// Predicate to sort indices based on their entity's AABB center.
class Bvh_sah_builder::Predicate
{
public:
    Predicate(int dim) : m_dim( dim ) {}

    bool operator() (const Entity& lhs, const Entity& rhs) const
    {
        const float leftval  = lhs.m_bbox[0][m_dim] + lhs.m_bbox[1][m_dim];
        const float rightval = rhs.m_bbox[0][m_dim] + rhs.m_bbox[1][m_dim];

        if (leftval == rightval)
            return lhs.m_index < rhs.m_index;

        return leftval < rightval;
    }

private:
    int m_dim;
};

inline float Bvh_sah_builder::area(const Bbox3f& bbox)
{
    const Vector3f edge = bbox[1] - bbox[0];
    return edge[0] * edge[1] + edge[2] * (edge[0] + edge[1]);
}

inline bool Bvh_sah_builder::find_largest_axis_split(
    const Node& node,
    uint32&     pivot,
    bbox_type&  left_bb,
    bbox_type&  right_bb)
{
    const uint32 begin = node.m_begin;
    const uint32 end   = node.m_end;
    const uint32 n_entities = end - begin;
    if (m_bboxes.size() < n_entities+1)
        m_bboxes.resize( n_entities+1 );
    if (m_costs.size() < n_entities+1)
        m_costs.resize( n_entities+1 );

    float min_cost = std::numeric_limits<float>::max();
    int   min_dim  = -1;

    const vector_type edge = node.m_bbox[1] - node.m_bbox[0];
    const int dim = max_element( edge );

    Predicate predicate( dim );

    m_bboxes[0] = bbox_type();
    m_costs[0]  = 0.0f;

    std::vector<int>& indices = m_indices[dim];

    // Accumulate right hand side bboxes.
    for(uint32 i = 1; i <= n_entities; ++i)
        m_bboxes[i] = bbox_type( m_bboxes[i-1], m_entities[ indices[ end - i ] ].m_bbox );

    for(uint32 i = 1; i <= n_entities; ++i)
        m_costs[i] = m_costs[i-1] + m_entities[ indices[ end - i ] ].m_cost;

    // Loop over possible splits and find the cheapest one.
    bbox_type lbb, rbb;
    float     lc = 0.0f, rc;

    for(uint32 num_left = 1; num_left < n_entities; ++num_left)
    {
        const int num_right = n_entities - num_left;

        rbb = m_bboxes[num_right];
        lbb.insert( m_entities[ indices[ begin + num_left - 1 ] ].m_bbox );

        rc  = m_costs[num_right];
        lc += m_entities[ indices[ begin + num_left - 1 ] ].m_cost;

        if (m_force_alignment && (num_left % m_max_leaf_size != 0))
            continue;

        const float sa_left  = area( lbb );
        const float sa_right = area( rbb );

        const float cost = sa_left*lc + sa_right*rc;

        if(cost < min_cost)
        {
            min_cost = cost;
            min_dim  = dim;
            pivot    = begin + num_left;
            left_bb  = lbb;
            right_bb = rbb;
        }
    }

    if (min_dim == -1)
        return false;

    // Compute the cost for the current node.
    const float curr_cost = area( node.m_bbox ) * n_entities;

    // Don't split if the split is more expensive than the current node.
    if (m_force_splitting == false && curr_cost * 1.5f < min_cost /*TODO: remove magic constant*/ )
        return false;

    // Tag the entities whether they're left or right according to the winner split.
    const int* indices_ptr = &m_indices[min_dim][0];
    for (uint32 i = begin; i < pivot; ++i)
        m_tag_bits[indices_ptr[i]] = false;
    for (uint32 i = pivot; i < end; ++i)
        m_tag_bits[indices_ptr[i]] = true;

    // Rearrange the indices of the non-winner dimensions to sorted left/right.
    for (int d = 1; d <= 2; ++d)
    {
        const int dim = (min_dim+d) % 3;
        int leftpos  = begin;
        int rightpos = pivot;

        const int* indices_ptr = &m_indices[dim][0];
        for (uint32 i = begin; i < end; ++i)
        {
            const int index = indices_ptr[i];

            if (m_tag_bits[index])
                m_tmpInt[rightpos++] = index;
            else
                m_tmpInt[leftpos++] = index;
        }

        // Copy the working array back to indices.
        memcpy( &m_indices[dim][begin], &m_tmpInt[begin], sizeof(int)*n_entities );
    }
    return true;
}

inline bool Bvh_sah_builder::find_best_split(
    const Node& node,
    uint32&     pivot,
    bbox_type&  left_bb,
    bbox_type&  right_bb)
{
    const uint32 begin = node.m_begin;
    const uint32 end   = node.m_end;
    const uint32 n_entities = end - begin;

    if (n_entities > m_single_axis_threshold)
        return find_largest_axis_split( node, pivot, left_bb, right_bb );

    if (m_bboxes.size() < n_entities+1)
        m_bboxes.resize( n_entities+1 );
    if (m_costs.size() < n_entities+1)
        m_costs.resize( n_entities+1 );

    float min_cost = std::numeric_limits<float>::max();
    int   min_dim  = -1;

    // Find the least costly split for each dimension.
    for(int dim = 0; dim < 3; ++dim)
    {
        Predicate predicate( dim );

        m_bboxes[0] = bbox_type();
        m_costs[0]  = 0.0f;

        std::vector<int>& indices = m_indices[dim];

        // Accumulate right hand side bboxes.
        for(uint32 i = 1; i <= n_entities; ++i)
            m_bboxes[i] = bbox_type( m_bboxes[i-1], m_entities[ indices[ end - i ] ].m_bbox );

        for(uint32 i = 1; i <= n_entities; ++i)
            m_costs[i] = m_costs[i-1] + m_entities[ indices[ end - i ] ].m_cost;

        // Loop over possible splits and find the cheapest one.
        bbox_type lbb, rbb;
        float     lc = 0.0f, rc;

        for(uint32 num_left = 1; num_left < n_entities; ++num_left)
        {
            const int num_right = n_entities - num_left;

            rbb = m_bboxes[num_right];
            lbb.insert( m_entities[ indices[ begin + num_left - 1 ] ].m_bbox );

            rc  = m_costs[num_right];
            lc += m_entities[ indices[ begin + num_left - 1 ] ].m_cost;

            if (m_force_alignment && (num_left % m_max_leaf_size != 0))
                continue;

            const float sa_left  = area( lbb );
            const float sa_right = area( rbb );

            const float cost = sa_left*lc + sa_right*rc;

            if(cost < min_cost)
            {
                min_cost = cost;
                min_dim  = dim;
                pivot    = begin + num_left;
                left_bb  = lbb;
                right_bb = rbb;
            }
        }
    }

    if (min_dim == -1)
        return false;

    // Compute the cost for the current node.
    const float curr_cost = area( node.m_bbox ) * n_entities;

    // Don't split if the split is more expensive than the current node.
    if (m_force_splitting == false && curr_cost * 1.5f < min_cost /*TODO: remove magic constant*/ )
        return false;

    // Tag the entities whether they're left or right according to the winner split.
    const int* indices_ptr = &m_indices[min_dim][0];
    for (uint32 i = begin; i < pivot; ++i)
        m_tag_bits[indices_ptr[i]] = false;
    for (uint32 i = pivot; i < end; ++i)
        m_tag_bits[indices_ptr[i]] = true;

    // Rearrange the indices of the non-winner dimensions to sorted left/right.
    for (int d = 1; d <= 2; ++d)
    {
        const int dim = (min_dim+d) % 3;
        int leftpos  = begin;
        int rightpos = pivot;

        const int* indices_ptr = &m_indices[dim][0];
        for (uint32 i = begin; i < end; ++i)
        {
            const int index = indices_ptr[i];

            if (m_tag_bits[index])
                m_tmpInt[rightpos++] = index;
            else
                m_tmpInt[leftpos++] = index;
        }

        // Copy the working array back to indices.
        memcpy( &m_indices[dim][begin], &m_tmpInt[begin], sizeof(int)*n_entities );
    }
    return true;
}

inline void Bvh_sah_builder::compute_bbox(
    const uint32      begin,
    const uint32      end,
    bbox_type&          bbox)
{
    bbox = bbox_type();
    for (uint32 i = begin; i < end; i++)
        bbox.insert( m_entities[m_indices[0][i]].m_bbox );
}

namespace deprecated {

template <typename Iterator>
void Bvh_sah_builder::build(
	const Iterator	begin,
	const Iterator	end,
	bvh_type*		bvh)
{
	m_entities.resize( end - begin );
	uint32 i = 0;
	for (Iterator it = begin; it != end; ++it)
	{
		m_entities[i].m_bbox  = *it;
		m_entities[i].m_index = i;

		i++;
	}

	// create root
	Node_stack node_stack;
	{
		Node node;
		compute_bbox( 0u, uint32(end - begin), node.m_bbox );
		node.m_begin = 0u;
		node.m_end   = uint32(end - begin);
		node.m_depth = 0u;
		node.m_node  = 0u;

		node_stack.push( node );
	}

	Bvh_node root;
	root.set_type( Bvh_node::kLeaf );
	root.set_index( 0u );
	bvh->m_nodes.push_back( root );
	bvh->m_bboxes.resize( 1u );

	while (node_stack.empty() == false)
	{
		const Node node = node_stack.top();
		node_stack.pop();
		
		const uint32 node_index = node.m_node;
		const bbox_type bbox    = node.m_bbox;

		bvh->m_bboxes[ node_index ] = bbox;

		if (node.m_end - node.m_begin < m_max_leaf_size)
		{
			//
			// Make a leaf node
			//
			Bvh_node& bvh_node = bvh->m_nodes[ node_index ];
            bvh_node = Bvh_node( Bvh_node::kLeaf, node.m_begin, node.m_end );
		}
		else
		{
			//
			// Make a split node
			//

			uint32		middle;
			bbox_type	l_bb, r_bb;

			if (find_best_split( node, middle, l_bb, r_bb ) == false)
			{
				// unsuccessful split: make a leaf node
				Bvh_node& bvh_node = bvh->m_nodes[ node_index ];
                bvh_node = Bvh_node( Bvh_node::kLeaf, node.m_begin, node.m_end );
				continue;
			}

			// alloc space for children
			const uint32 left_node_index = uint32( bvh->m_nodes.size() );
			bvh->m_nodes.resize( left_node_index + 2 );
			bvh->m_bboxes.resize( left_node_index + 2 );

			Bvh_node& bvh_node = bvh->m_nodes[ node_index ];
            bvh_node = Bvh_node( Bvh_node::kInternal, left_node_index, node.m_end - node.m_begin );

			// push left and right children in processing queue
			Node right_node;
			right_node.m_bbox  = r_bb;
			right_node.m_begin = middle;
			right_node.m_end   = node.m_end;
			right_node.m_depth = node.m_depth + 1u;
			right_node.m_node  = left_node_index + 1u;
			node_stack.push( right_node );

			Node left_node;
			left_node.m_bbox  = l_bb;
			left_node.m_begin = node.m_begin;
			left_node.m_end   = middle;
			left_node.m_depth = node.m_depth + 1u;
			left_node.m_node  = left_node_index;
			node_stack.push( left_node );
		}
	}
}


// Predicate to sort indices based on their entity's AABB center.
class Bvh_sah_builder::Predicate
{
public:
	Predicate(int dim) : m_dim( dim ) {}

	bool operator() (const Entity& lhs, const Entity& rhs) const
	{
		const float leftval  = lhs.m_bbox[0][m_dim] + lhs.m_bbox[1][m_dim];
		const float rightval = rhs.m_bbox[0][m_dim] + rhs.m_bbox[1][m_dim];

		return leftval < rightval;
	}

private:
	int m_dim;
};

inline float Bvh_sah_builder::area(const Bbox3f& bbox)
{
	const Vector3f edge = bbox[1] - bbox[0];
	return edge[0] * edge[1] + edge[2] * (edge[0] + edge[1]);
}

inline bool Bvh_sah_builder::find_best_split(
	const Node&	node,
	uint32&		pivot,
	bbox_type&	left_bb,
	bbox_type&	right_bb)
{
	float		min_cost = std::numeric_limits<float>::max();
	int			min_dim  = -1;

	const uint32 begin = node.m_begin;
	const uint32 end   = node.m_end;
	const uint32 n_entities = end - begin;
	if (m_bboxes.size() < n_entities+1)
		m_bboxes.resize( n_entities+1 );

	// Find the least costly split for each dimension.
	for(int dim = 0; dim < 3; ++dim)
	{
		Predicate predicate( dim );

		// Sort along current dimension.
		std::sort(
			&m_entities[0] + begin,
			&m_entities[0] + end,
			predicate );

		m_bboxes[0].clear();

		// Accumulate right hand side bboxes.
		for(uint32 i = 1; i <= n_entities; ++i)
		{
			m_bboxes[i] = m_bboxes[i-1];
			m_bboxes[i].insert( m_entities[ end - i ].m_bbox );
		}

		// Loop over possible splits and find the cheapest one.
		bbox_type lbb, rbb;

		for(uint32 num_left = 1; num_left < n_entities; ++num_left)
		{
			const int num_right = n_entities - num_left;

			rbb = m_bboxes[num_right];
			lbb.insert( m_entities[ begin + num_left - 1 ].m_bbox );

			const float sa_left  = area( lbb );
			const float sa_right = area( rbb );

			const float cost = sa_left*num_left + sa_right*num_right;

			if(cost < min_cost)
			{
				min_cost = cost;
				min_dim  = dim;
				pivot    = begin + num_left;
				left_bb  = lbb;
				right_bb = rbb;
			}
		}
	}

	// Compute the cost for the current node.
	const float curr_cost = area( node.m_bbox ) * n_entities;

	// Don't split if the split is more expensive than the current node.
	if( curr_cost * 1.5f < min_cost)
		return false;

	// Resort if necessary.
	if(min_dim != 2)
	{
		Predicate predicate( min_dim );
		std::sort(
			&m_entities[0] + begin,
			&m_entities[0] + end,
			predicate );
	}
	return true;
}

inline void Bvh_sah_builder::compute_bbox(
	const uint32		begin,
	const uint32		end,
	bbox_type&			bbox)
{
	bbox.clear();
	for (uint32 i = begin; i < end; i++)
		bbox.insert( m_entities[i].m_bbox );
}

} // namespace deprecated

} // namespace cugar