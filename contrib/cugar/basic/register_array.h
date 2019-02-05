/*
 * cugar
 * Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
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

/*! \file register_array.h
 *   \brief A CUDA-compatible, fixed-size priority queue
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/iterator.h>

namespace cugar {

/// \page register_array_page Register Arrays
///
/// This module implements a form of arrays that, by only using static indexing under the hood,
/// can be easily placed in registers.
/// The ability to be placed in registers is achieved by avoiding dynamic indexing,
/// replacing that by an explicit O(log(N)) binary-search.
///
/// - register_array
///

///@addtogroup Basic
///@{

///@defgroup RegisterArraysModule Register Arrays
/// This module implements a statically sized array using registers as backing storage.
/// The ability to be placed in registers is achieved by avoiding dynamic indexing,
/// replacing that by an explicit O(log(N)) binary-search.
///@{
    

///
/// A statically sized array using registers as backing storage.
/// The ability to be placed in registers is achieved by avoiding dynamic indexing,
/// replacing that by an explicit O(log(N)) binary-search.
///
/// \tparam Iterator	the base iterator type
/// \tparam SIZE		the size of the array
///
template <typename Iterator, uint32 SIZE>
struct register_array
{
	typedef iterator_traits<Iterator>::reference	reference_type;
    static const uint32								DIM = SIZE;

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array() {}

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array(Iterator it) : r(it) {}

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE reference_type operator[](const uint32 i) const { select( r, i ); }

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static reference select(const Iterator& it, const uint32 i) { return it[i]; }

	/// automatic conversion to iterator
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE operator Iterator() const { return r; }

	Iterator r;
};
 
///
/// A statically sized array using registers as backing storage
///
/// \tparam Iterator	the base iterator type
///
template <typename Iterator>
struct register_array<Iterator,4>
{
	typedef iterator_traits<Iterator>::reference	reference_type;
    static const uint32								DIM = 4;

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array() {}

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array(const Iterator& it) : r(it) {}

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE reference_type operator[](const uint32 i) const { select( r, i ); }

	/// automatic conversion to iterator
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE operator Iterator() const { return r; }

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static reference select(const Iterator& it, const uint32 i)
	{
		return i < 2 ?  (i == 0 ? it[0] : it[1]) :
						(i == 2 ? it[2] : it[3]);
	}

	Iterator r;
};

///
/// A statically sized array using registers as backing storage
///
/// \tparam Iterator	the base iterator type
///
template <typename Iterator>
struct register_array<Iterator,5>
{
	typedef iterator_traits<Iterator>::reference	reference_type;
    static const uint32								DIM = 5;

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array() {}

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array(const Iterator& it) : r(it) {}

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE reference_type operator[](const uint32 i) const { select( r, i ); }

	/// automatic conversion to iterator
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE operator Iterator() const { return r; }

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static reference select(const Iterator& it, const uint32 i)
	{
		return i == 4 ? it[4] :
			i < 2 ? (i == 0 ? it[0] : it[1]) :
					(i == 2 ? it[2] : it[3]);
	}

	Iterator r;
};

///
/// A statically sized array using registers as backing storage
///
/// \tparam Iterator	the base iterator type
///
template <typename Iterator>
struct register_array<Iterator,8>
{
	typedef iterator_traits<Iterator>::reference	reference_type;
    static const uint32								DIM = 8;

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array() {}

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array(const Iterator& it) : r(it) {}

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE reference_type operator[](const uint32 i) const { select( r, i ); }

	/// automatic conversion to iterator
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE operator Iterator() const { return r; }

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static reference select(const Iterator& it, const uint32 i)
	{
		return i < 4 ?
				(i < 2 ? (i == 0 ? it[0] : it[1]) :
						 (i == 2 ? it[2] : it[3])) :
				(i < 6 ? (i == 4 ? it[4] : it[5]) :
						 (i == 6 ? it[6] : it[7]));
	}

	Iterator r;
};

///
/// A statically sized array using registers as backing storage
///
/// \tparam Iterator	the base iterator type
///
template <typename Iterator>
struct register_array<Iterator,16>
{
	typedef iterator_traits<Iterator>::reference	reference_type;
    static const uint32								DIM = 16;

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array() {}

	/// constructor
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE register_array(const Iterator& it) : r(it) {}

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE reference_type operator[](const uint32 i) const { select( r, i ); }

	/// automatic conversion to iterator
	///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE operator Iterator() const { return r; }

	/// indexing operator
    ///
	CUGAR_FORCEINLINE CUGAR_HOST_DEVICE
	static reference select(const Iterator& it, const uint32 i)
	{
		return i < 8 ?
			(i < 4 ?
				(i < 2 ? (i == 0 ? it[0] : it[1]) :
						 (i == 2 ? it[2] : it[3])) :
				(i < 6 ? (i == 4 ? it[4] : it[5]) :
						 (i == 6 ? it[6] : it[7]))) :
			(i < 12 ?
				(i < 10 ? (i == 8  ? it[8]  : it[9]) :
						  (i == 10 ? it[10] : it[11])) :
				(i < 14 ? (i == 12 ? it[12] : it[13]) :
						  (i == 14 ? it[14] : it[15])));
	}
	Iterator r;
};

/// a free function to perform dynamic indexing on vectors / iterators using a statically compiled binary search
///
template <uint32 SIZE, typename iterator_type>
iterator_traits<iterator_type>::reference dynamic_index(const iterator_type& T, const uint32 i)
{
	return register_array<iterator_type,SIZE>::select(T, i);
}

///@} RegisterArraysModule
///@} Basic

} // namespace cugar
