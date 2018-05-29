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

/*! \file sort.h
 *   \brief Define CUDA based sort primitives.
 */

#pragma once

#include <cugar/basic/types.h>
#include <cugar/basic/atomics.h>
#include <cugar/basic/numbers.h>
#include <cugar/basic/cuda/pointers.h>

namespace cugar {
namespace cuda {

/// \page hashmaps_page Hash Maps Module
///\par
/// This module contains containers to build block- and device-wide  hash-sets and hash-maps.
///

///@addtogroup Basic
///@{

///@addtogroup CUDAModule
///@{

///@defgroup HashMapsModule   Hash Maps
/// This module contains containers to build block- and device-wide  hash-sets and hash-maps.
///@{

/// This class implements a device-side Hash Set, allowing arbitrary threads from potentially
/// different CTAs of a cuda kernel to add new entries at the same time.
///
/// The constructor of the class accepts pointers to the arrays representing the underlying data structure:
///
/// - an array of hashed keys
/// - an array of unique set entries
/// - a pointer to a counter, keeping track of the number of set entries
///
/// Before starting to use the HashSet, the array of hash slots has to be initialized to INVALID_KEY.
///
/// The class provides two methods:
///
///     - HashSet::insert
///     - HashSet::has
///
/// It's important to notice that calls to the two methods cannot be mixed without interposing a global
/// synchronization between them.
///
template <typename KeyT, typename HashT, KeyT INVALID_KEY = 0xFFFFFFFF>
struct HashSet
{
    /// empty constructor
    ///
    CUGAR_DEVICE
    HashSet() {}

    /// HashSet constructor
    ///
    /// \param  _table_size         the size of the hash table (needs to be a power of 2)
    /// \param  _hash               a pointer to the array of _table_size hashed keys - needs to be initialized to INVALID_KEY before first use
    /// \param  _unique             a pointer to the array where inserted entries will be stored - needs to be appropriately sized to contain all unique entries
    /// \param  _count              a pointer to a single counter, used to keep track of how many unique entries have been inserted
    ///
    CUGAR_DEVICE
    HashSet(const uint32 _table_size, KeyT* _hash, KeyT* _unique, uint32* _count) :
        table_size(_table_size),
        hash(_hash),
        unique(_unique),
        count(_count)
    {}

    /// insert an element with its hashing value
    ///
    /// \param key          the element to insert
    /// \param hash_code    the hashing value
    ///
    CUGAR_DEVICE
    void insert(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old  = INVALID_KEY;

        do
        {
            slot = (slot + skip) & (table_size - 1);
            old = atomicCAS( &hash[slot], INVALID_KEY, key );
        } while (old != INVALID_KEY && old != key);

        // assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            unique[ unique_id ] = key;
        }
    }

	/// return the size of the hash set
    ///
	CUGAR_DEVICE
    uint32 size() const { return *count; }

    /// return the i-th unique element in the set
    ///
	CUGAR_DEVICE
    KeyT get_unique(const uint32 i) const { return unique[i]; }

    uint32    table_size;
    KeyT*     hash;
    KeyT*     unique;
    uint32*   count;
};

/// This class implements a cuda block-wide Hash Set, allowing arbitrary threads from the same CTA
/// to add new entries at the same time.
///
template <typename KeyT, typename HashT, uint32 CTA_SIZE, uint32 TABLE_SIZE, KeyT INVALID_KEY = 0xFFFFFFFF>
struct BlockHashSet : public HashSet<KeyT,HashT,INVALID_KEY>
{
    struct TempStorage
    {
        KeyT   hash[TABLE_SIZE];
        KeyT   unique[TABLE_SIZE];
        uint32 count;
    };

    /// emptry constructor
    ///
    CUGAR_DEVICE
    BlockHashSet() {}

    /// constructor
    ///
    /// \param  _storage            the per-CTA storage backing this container
    ///
    CUGAR_DEVICE
    BlockHashSet(TempStorage& _storage) : HashSet( TABLE_SIZE, _storage.hash, _storage.unique, &_storage.count )
    {
        // clear the table
        const uint32 ITEMS_PER_THREAD = TABLE_SIZE / CTA_SIZE;
        for (uint32 i = 0; i < ITEMS_PER_THREAD; ++i)
            storage.hash[ CTA_SIZE * i + threadIdx.x ] = INVALID_KEY;

        // initialize the counter
        if (threadIdx.x == 0)
            storage.count = 0;

        __syncthreads();
    }
};


/// This class implements a device-side Hash Map, allowing arbitrary threads from potentially
/// different CTAs of a cuda kernel to add new entries at the same time.
///
/// The constructor of the class accepts pointers to the arrays representing the underlying data structure:
///
/// - an array of hashed keys
/// - an array of unique set entries
/// - an array of slots keeping track of the unique position where each inserted key has been mapped
/// - a pointer to a counter, keeping track of the number of set entries
///
/// Before starting to use the HashSet, the array of hash slots has to be initialized to INVALID_KEY.
///
/// The class provides two methods:
///
///     - HashMap::insert
///     - HashMap::find
///
/// It's important to notice that calls to the two methods cannot be mixed without interposing a global
/// synchronization between them.
///
template <typename KeyT, typename HashT, KeyT INVALID_KEY = 0xFFFFFFFF>
struct HashMap
{
    /// constructor
    ///
    CUGAR_DEVICE
    HashMap() {}

    /// constructor
    ///
    /// \param  _table_size         the size of the hash table (needs to be a power of 2)
    /// \param  _hash               a pointer to the array of _table_size hashing slots - needs to be initialized to INVALID_KEY before first use
    /// \param  _unique             a pointer to the array where inserted entries will be stored - needs to be appropriately sized to contain all unique entries
    /// \param  _slots              a pointer to an array of _table_size entries keeping track of the unique position where each inserted key has been mapped
    /// \param  _count              a pointer to a single counter, used to keep track of how many unique entries have been inserted
    ///
    CUGAR_DEVICE
    HashMap(const uint32 _table_size, KeyT* _hash, KeyT* _unique, uint32* _slots, uint32* _count) :
        table_size(_table_size),
        hash(_hash),
        unique(_unique),
        slots(_slots),
        count(_count) {}

    /// insert an element with its hashing value
    ///
    /// \param key          the element to insert
    /// \param hash_code    the hashing value
    ///
    CUGAR_DEVICE
    void insert(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old  = INVALID_KEY;

        do
        {
            slot = (slot + skip) & (table_size - 1);
            old = atomicCAS( &hash[slot], INVALID_KEY, key );
        } while (old != INVALID_KEY && old != key);

        // assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            unique[ unique_id ] = key;
            slots[ slot ] = unique_id;
        }
    }
    
    /// find the unique slot associated to an inserted key (NOTE: needs to be separated from insertion by a synchronization point)
    ///
    CUGAR_DEVICE
    uint32 find(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;

        do
        {
            slot = (slot + skip) & (table_size - 1);
			if (hash[slot] == INVALID_KEY)
				return 0xFFFFFFFFu;
        }
        while (hash[slot] != key);

        return slots[slot];
    }

    /// return the size of the hash set
    ///
	CUGAR_DEVICE
    uint32 size() const { return *count; }

    /// return the i-th unique element in the set
    ///
	CUGAR_DEVICE
    KeyT get_unique(const uint32 i) const { return unique[i]; }

    uint32  table_size;
    KeyT*   hash;
    KeyT*   unique;
    uint32* slots;
    uint32* count;
};

/// This class implements a cuda block-wide Hash Set, allowing arbitrary threads from the same CTA
/// to add new entries at the same time.
///
/// \code
/// __global__ void kernel()
/// {
///    typedef BlockHashMap<uint32,uint32,128,512> hash_map_type;
///
///    // allocate the temporary storage of the hash map in shared memory
///    __shared__ typename hash_map_type::TempStorage     hash_map_storage;
///
///    hash_map_type hash_map( hash_map_storage );
///
///    hash_map.insert( threadIdx.x/2, hash(threadIdx.x/2) );
///
///    // allocate some storage for the hash map values
///    __shared__ uint32 hash_map_values[512];
///
///    // initialize the hash map values to zero
///    for (uint32 i = threadIdx.x; i < 512; i += 128)
///         hash_map_values[i] = 0;
///
///    __syncthreads();
///
///    // add 1 to each entry sharing the same slot in the hash map
///    const uint32 slot = hash_map.find( threadIdx.x/2 );
///
///    atomic_add( &hash_map_values[ slot ], 1u );
/// }
/// \endcode
///
template <typename KeyT, typename HashT, uint32 CTA_SIZE, uint32 TABLE_SIZE, KeyT INVALID_KEY = 0xFFFFFFFF>
struct BlockHashMap : public HashMap<KeyT,HashT,INVALID_KEY>
{
    struct TempStorage
    {
        KeyT   hash[TABLE_SIZE];
        KeyT   unique[TABLE_SIZE];
        uint32 slots[TABLE_SIZE];
        uint32 count;
    };

    /// empty constructor
    ///
    CUGAR_DEVICE
    BlockHashMap() {}

    /// constructor
    ///
    /// \param  _storage            the per-CTA storage backing this container
    ///
    CUGAR_DEVICE
    BlockHashMap(TempStorage& _storage) : HashMap( TABLE_SIZE, _storage.hash, _storage.unique, _storage.slots, &_storage.count )
    {
        // clear the table
        const uint32 ITEMS_PER_THREAD = TABLE_SIZE / CTA_SIZE;
        for (uint32 i = 0; i < ITEMS_PER_THREAD; ++i)
            hash[ CTA_SIZE * i + threadIdx.x ] = INVALID_KEY;

        // initialize the counter
        if (threadIdx.x == 0)
            *count = 0;

        __syncthreads();
    }
};

#if 0

/// This class implements a device-side Hash Map, allowing arbitrary threads from potentially
/// different CTAs of a cuda kernel to add new entries at the same time.
///
/// The constructor of the class accepts pointers to the arrays representing the underlying data structure:
///
/// - an array of hashed keys
/// - an array of unique set entries
/// - an array of slots keeping track of the unique position where each inserted key has been mapped
/// - a pointer to a counter, keeping track of the number of set entries
///
/// Before starting to use the SynchronousHashMap, the array of hash slots has to be initialized to INVALID_KEY.
///
/// The difference between this class and the HashMap is that this class allows to call both insert() and find()
/// at the same time, without interposing a global synchronization between them: the price is higher overhead
/// (due to the need to use volatile writes and add memory fences).
///
template <typename KeyT, typename HashT, KeyT INVALID_KEY = 0xFFFFFFFF>
struct SyncFreeHashMap
{
	static const uint32 BUCKET_SIZE = 1;

	/// constructor
    ///
    CUGAR_HOST_DEVICE
    SyncFreeHashMap() {}

    /// constructor
    ///
    /// \param  _table_size         the size of the hash table (needs to be a power of 2)
    /// \param  _hash               a pointer to the array of _table_size hashing slots - needs to be initialized to INVALID_KEY before first use
    /// \param  _unique             a pointer to the array where inserted entries will be stored - needs to be appropriately sized to contain all unique entries
    /// \param  _slots              a pointer to an array of _table_size entries keeping track of the unique position where each inserted key has been mapped
    /// \param  _count              a pointer to a single counter, used to keep track of how many unique entries have been inserted
    ///
    CUGAR_HOST_DEVICE
    SyncFreeHashMap(const uint32 _table_size, KeyT* _hash, KeyT* _unique, uint32* _slots, uint32* _count) :
        hash(_hash),
        unique((volatile KeyT*)_unique),
        slots((volatile uint32*)_slots),
        count(_count),
		table_size(_table_size) {}

	/// insert an element with its hashing value
	///
	/// \param key          the element to insert
	/// \param hash_code    the hashing value
	///
	CUGAR_DEVICE
	bool try_insert(const KeyT key, const HashT hash_code, const uint32 n)
	{
		const HashT skip = (hash_code / table_size) | 1;

		HashT slot = hash_code;
		KeyT old = INVALID_KEY;

		slot = (slot + skip * n) & (table_size - 1u);
		old = atomicCAS(&hash[slot], INVALID_KEY, key);
		if (old == INVALID_KEY || old == key)
		{
			// assign compacted vertex slots
			if (old == INVALID_KEY)
			{
				const uint32 unique_id = atomic_add(count, 1);
				unique[unique_id] = key;
				slots[slot] = unique_id;
				__threadfence(); // make sure the write will eventually be visible
			}
			return true;
		}
		return false;
	}

    /// insert an element with its hashing value
    ///
    /// \param key          the element to insert
    /// \param hash_code    the hashing value
    ///
    CUGAR_DEVICE
    void insert(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old  = INVALID_KEY;

        do
        {
            slot = (slot + skip) & (table_size - 1u);
            old = atomicCAS( &hash[slot], INVALID_KEY, key );
        } while (old != INVALID_KEY && old != key);

        // assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            unique[ unique_id ] = key;
            slots[ slot ] = unique_id;
            __threadfence(); // make sure the write will eventually be visible
        }
    }

	/// insert an element with its hashing value and immediately get the unique slot where it's been inserted
	///
	/// \param key          the element to insert
	/// \param hash_code    the hashing value
	///
	CUGAR_DEVICE
    bool insert(const KeyT key, const HashT hash_code, uint32* pos)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old  = INVALID_KEY;

        do
        {
            slot = (slot + skip) & (table_size - 1u);
            old = atomicCAS( &hash[slot], INVALID_KEY, key );
        } while (old != INVALID_KEY && old != key);

        // assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            unique[ unique_id ] = key;
            slots[ slot ] = unique_id;
            __threadfence(); // make sure the write will eventually be visible
			*pos = unique_id;
			return true;	// first thread to fetch this entry
        }
        else
        {
            // loop until the slot has been written to
            while (slots[slot] == 0xFFFFFFFFu) {}

			*pos = slots[slot];
			return false;	// pre-existing entry
        }
    }

    /// find the unique slot associated to an inserted key
    ///
    CUGAR_DEVICE
    uint32 find(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;

        do
        {
            slot = (slot + skip) & (table_size - 1u);
			if (hash[slot] == INVALID_KEY)
				return 0xFFFFFFFFu;
        }
        while (hash[slot] != key);

        // loop until the slot has been written to
        while (slots[slot] == 0xFFFFFFFFu) {}

        return slots[slot];
    }

    /// return the size of the hash set
    ///
    uint32 size() const { return *count; }

    /// return the i-th unique element in the set
    ///
    KeyT get_unique(const uint32 i) const { return unique[i]; }

    KeyT*            hash;
    volatile KeyT*   unique;
    volatile uint32* slots;
    uint32*          count;
	uint32           table_size;
};

#else

#define HASH_UNCACHED_LOAD(x)    load<LOAD_VOLATILE>(x)
#define HASH_UNCACHED_STORE(x,v) store<STORE_VOLATILE>(x,v)

/// This class implements a device-side Hash Map, allowing arbitrary threads from potentially
/// different CTAs of a cuda kernel to add new entries at the same time.
///
/// The constructor of the class accepts pointers to the arrays representing the underlying data structure:
///
/// - an array of hashed keys
/// - an array of unique set entries
/// - an array of slots keeping track of the unique position where each inserted key has been mapped
/// - a pointer to a counter, keeping track of the number of set entries
///
/// Before starting to use the SynchronousHashMap, the array of hash slots has to be initialized to INVALID_KEY.
///
/// The difference between this class and the HashMap is that this class allows to call both insert() and find()
/// at the same time, without interposing a global synchronization between them: the price is higher overhead
/// (due to the need to use volatile writes and add memory fences).
///
template <typename KeyT, typename HashT, KeyT INVALID_KEY = 0xFFFFFFFF>
struct SyncFreeHashMap
{
	static const uint32 BUCKET_SIZE = 8;

	/// constructor
    ///
    CUGAR_HOST_DEVICE
    SyncFreeHashMap() {}

    /// constructor
    ///
    /// \param  _table_size         the size of the hash table (needs to be a power of 2)
    /// \param  _hash               a pointer to the array of _table_size hashing slots - needs to be initialized to INVALID_KEY before first use
    /// \param  _unique             a pointer to the array where inserted entries will be stored - needs to be appropriately sized to contain all unique entries
    /// \param  _slots              a pointer to an array of _table_size entries keeping track of the unique position where each inserted key has been mapped
    /// \param  _count              a pointer to a single counter, used to keep track of how many unique entries have been inserted
    ///
    CUGAR_HOST_DEVICE
    SyncFreeHashMap(const uint32 _table_size, KeyT* _hash, KeyT* _unique, uint32* _slots, uint32* _count) :
        hash(_hash),
        unique(_unique),
        slots(_slots),
        count(_count),
		table_size(_table_size) {}

	/// insert an element with its hashing value
	///
	/// \param key          the element to insert
	/// \param hash_code    the hashing value
	///
	CUGAR_DEVICE
	bool try_insert(const KeyT key, const HashT hash_code, const uint32 n)
	{
		const HashT skip = (hash_code / table_size) | 1;

		HashT slot = hash_code;
		KeyT old = INVALID_KEY;

		// advance by n buckets
		slot = (slot + skip * n) & (table_size - 1u);

		// look into one bucket
		uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;

		// search within the bucket
		#pragma unroll
		for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
		{
			old = atomicCAS( &hash[bucket + bucket_index], INVALID_KEY, key );
			if (old == INVALID_KEY || old == key)
			{
				slot = bucket + bucket_index;
				break;
			}
		}

		if (old == INVALID_KEY || old == key)
		{
			// assign compacted vertex slots
			if (old == INVALID_KEY)
			{
				const uint32 unique_id = atomic_add(count, 1);
				HASH_UNCACHED_STORE(&unique[unique_id], key);
				HASH_UNCACHED_STORE(&slots[slot], unique_id);
				__threadfence(); // make sure the write will eventually be visible
			}
			return true;
		}
		return false;
	}

    /// insert an element with its hashing value
    ///
    /// \param key          the element to insert
    /// \param hash_code    the hashing value
    ///
    CUGAR_DEVICE
    void insert(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old   = INVALID_KEY;

        while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			// find the bucket containing this slot
			uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;

			// search within the bucket
			#pragma unroll
			for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
			{
				old = atomicCAS( &hash[bucket + bucket_index], INVALID_KEY, key );
				if (old == INVALID_KEY || old == key)
				{
					slot = bucket + bucket_index;
					break;
				}
			}

			if (old == INVALID_KEY || old == key)
				break;

			// linear probing
			slot = slot + skip;
		}

		// assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            HASH_UNCACHED_STORE(&unique[ unique_id ], key);
            HASH_UNCACHED_STORE(&slots[ slot ], unique_id);
            __threadfence(); // make sure the write will eventually be visible
        }
    }

	/// insert an element with its hashing value and immediately get the unique slot where it's been inserted
	///
	/// \param key          the element to insert
	/// \param hash_code    the hashing value
	///
	CUGAR_DEVICE
    bool insert(const KeyT key, const HashT hash_code, uint32* pos)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT  old  = INVALID_KEY;

        while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			// find the bucket containing this slot
			uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;

			// search within the bucket
			#pragma unroll
			for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
			{
				old = atomicCAS( &hash[bucket + bucket_index], INVALID_KEY, key );
				if (old == INVALID_KEY || old == key)
				{
					slot = bucket + bucket_index;
					break;
				}
			}

			if (old == INVALID_KEY || old == key)
				break;

			// linear probing
			slot = slot + skip;
		}

        // assign compacted vertex slots
        if (old == INVALID_KEY)
        {
            const uint32 unique_id = atomic_add( count, 1 );
            HASH_UNCACHED_STORE(&unique[ unique_id ], key);
            HASH_UNCACHED_STORE(&slots[ slot ], unique_id);
            __threadfence(); // make sure the write will eventually be visible
			*pos = unique_id;
			return true;	// first thread to fetch this entry
        }
        else
        {
            // loop until the slot has been written to
            while (HASH_UNCACHED_LOAD(&slots[slot]) == 0xFFFFFFFFu) {}

			*pos = slots[slot];
			return false;	// pre-existing entry
        }
    }

    /// find the unique slot associated to an inserted key
    ///
    CUGAR_DEVICE
    uint32 find(const KeyT key, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT  old  = INVALID_KEY;

        while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			// find the bucket containing this slot
			uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;

			// search within the bucket
			#pragma unroll
			for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
			{
				old = hash[bucket + bucket_index];
				if (old == INVALID_KEY)
					return 0xFFFFFFFFu;

				if (old == key)
				{
					slot = bucket + bucket_index;
					break;
				}
			}

			if (old == key)
				break;

			// linear probing
			slot = slot + skip;
		}

        // loop until the slot has been written to
        while (HASH_UNCACHED_LOAD(&slots[slot]) == 0xFFFFFFFFu) {}

        return slots[slot];
    }

    /// return the size of the hash set
    ///
    uint32 size() const { return *count; }

    /// return the i-th unique element in the set
    ///
    KeyT get_unique(const uint32 i) const { return unique[i]; }

    KeyT*           hash;
    KeyT*			unique;
    uint32*			slots;
    uint32*         count;
	uint32          table_size;
};

/// This class implements a device-side Hash Map, allowing arbitrary threads from potentially
/// different CTAs of a cuda kernel to add new entries at the same time.
///
/// The constructor of the class accepts pointers to the arrays representing the underlying data structure:
///
/// - an array of hashed keys
/// - an array of unique set entries
/// - an array of slots keeping track of the unique position where each inserted key has been mapped
/// - a pointer to a counter, keeping track of the number of set entries
///
/// Before starting to use the SynchronousHashMap, the array of hash slots has to be initialized to INVALID_KEY.
///
/// The difference between this class and the HashMap is that this class allows to call both insert() and find()
/// at the same time, without interposing a global synchronization between them: the price is higher overhead
/// (due to the need to use volatile writes and add memory fences).
///
template <typename KeyT, typename HashT, KeyT INVALID_KEY = 0xFFFFFFFF>
struct SyncFreeDoubleKeyHashMap
{
	typedef vector_type<KeyT,2>			pair_vector;
	typedef typename pair_vector::type	pair_type;

	static const uint32 BUCKET_SIZE = 8;

	/// constructor
    ///
    CUGAR_HOST_DEVICE
    SyncFreeDoubleKeyHashMap() {}

    /// constructor
    ///
    /// \param  _table_size         the size of the hash table (needs to be a power of 2)
    /// \param  _hash1              a pointer to the array of _table_size hashing slots - needs to be initialized to INVALID_KEY before first use
    /// \param  _hash2              a pointer to the array of _table_size hashing slots - needs to be initialized to INVALID_KEY before first use
    /// \param  _unique             a pointer to the array where inserted entries will be stored - needs to be appropriately sized to contain all unique entries
    /// \param  _slots              a pointer to an array of _table_size entries keeping track of the unique position where each inserted key has been mapped
    /// \param  _count              a pointer to a single counter, used to keep track of how many unique entries have been inserted
    ///
    CUGAR_HOST_DEVICE
    SyncFreeDoubleKeyHashMap(const uint32 _table_size, KeyT* _hash1, KeyT* _hash2, KeyT* _unique, uint32* _slots, uint32* _count) :
        hash1(_hash1),
        hash2(_hash2),
        unique((volatile KeyT*)_unique),
        slots((volatile uint32*)_slots),
        count(_count),
		table_size(_table_size) {}

    /// insert an element with its hashing value
    ///
    /// \param key          the element to insert
    /// \param hash_code    the hashing value
    ///
    CUGAR_DEVICE
    void insert(const KeyT key1, const KeyT key2, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old1  = INVALID_KEY;

		while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			old1 = atomicCAS( &hash1[slot], INVALID_KEY, key1 );
			if (old1 == INVALID_KEY || old1 == key1)
			{
				// search the second key within this bucket
				KeyT old2 = INVALID_KEY;

				// find the bucket containing this slot
				//const uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;
				const uint32 bucket = slot;

				#pragma unroll
				for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
				{
					old2 = atomicCAS( &hash2[bucket + bucket_index], INVALID_KEY, key2 );
					if (old2 == INVALID_KEY || old2 == key2)
					{
						// we found a slot!
						slot = bucket + bucket_index;

						// assign compacted vertex slots
						if (old2 == INVALID_KEY)
						{
							const uint32 unique_id = atomic_add( count, 1 );
							unique[ unique_id*2 + 0 ] = key1;
							unique[ unique_id*2 + 1 ] = key2;
							slots[ slot ] = unique_id;
							__threadfence(); // make sure the write will eventually be visible
						}
						return;
					}
				}
			}

			// linear probing
			slot = slot + skip;
		}
    }

	/// insert an element with its hashing value and immediately get the unique slot where it's been inserted
	///
	/// \param key          the element to insert
	/// \param hash_code    the hashing value
	///
	CUGAR_DEVICE
    bool insert(const KeyT key1, const KeyT key2, const HashT hash_code, uint32* pos)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;
        KeyT old1  = INVALID_KEY;

		while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			old1 = atomicCAS( &hash1[slot], INVALID_KEY, key1 );
			if (old1 == INVALID_KEY || old1 == key1)
			{
				// search the second key within this bucket
				KeyT old2 = INVALID_KEY;

				//const uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;
				const uint32 bucket = slot;

				#pragma unroll
				for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
				{
					old2 = atomicCAS( &hash2[bucket + bucket_index], INVALID_KEY, key2 );
					if (old2 == INVALID_KEY || old2 == key2)
					{
						// we found a slot!
						slot = bucket + bucket_index;

						// assign compacted vertex slots
						if (old2 == INVALID_KEY)
						{
							const uint32 unique_id = atomic_add( count, 1 );
							unique[ unique_id*2 + 0 ] = key1;
							unique[ unique_id*2 + 1 ] = key2;
							slots[ slot ] = unique_id;
							__threadfence(); // make sure the write will eventually be visible
							*pos = unique_id;
							return true;	// first thread to fetch this entry
						}
						else
						{
							// loop until the slot has been written to
							while (slots[slot] == 0xFFFFFFFFu) {}

							*pos = slots[slot];
							return false;	// pre-existing entry
						}
					}
				}
			}

			// linear probing
			slot = slot + skip;
		}
    }

    /// find the unique slot associated to an inserted key
    ///
    CUGAR_DEVICE
    uint32 find(const KeyT key1, const KeyT key2, const HashT hash_code)
    {
        const HashT skip = (hash_code / table_size) | 1;

        HashT slot = hash_code;

        while (1)
        {
			// wrap around
	        slot = slot & (table_size - 1u);

			KeyT old1 = hash1[slot];
			if (old1 == INVALID_KEY)
				return 0xFFFFFFFFu;

			if (old1 == key1)
			{
				//const uint32 bucket = (slot / BUCKET_SIZE) * BUCKET_SIZE;
				const uint32 bucket = slot;

				// search within the bucket
				#pragma unroll
				for (uint32 bucket_index = 0; bucket_index < BUCKET_SIZE; ++bucket_index)
				{
					KeyT old2 = hash2[bucket + bucket_index];
					if (old2 == INVALID_KEY)
						return 0xFFFFFFFFu;

					if (old2 == key2)
					{
						// we found our slot
						slot = bucket + bucket_index;

						// loop until the slot has been written to
						while (slots[slot] == 0xFFFFFFFFu) {}

						return slots[slot];
					}
				}
			}

			// linear probing
			slot = slot + skip;
		}
        //return 0xFFFFFFFFu;
    }

    /// return the size of the hash set
    ///
    uint32 size() const { return *count; }

    /// return the i-th unique element in the set
    ///
    pair_type get_unique(const uint32 i) const { return pair_vector::make(unique[i*2],unique[i*2+1]); }

    KeyT*				hash1;
    KeyT*				hash2;
    volatile KeyT*		unique;
    volatile uint32*	slots;
    uint32*				count;
	uint32				table_size;
};

#endif

///@} HashMapsModule
///@} CUDAModule
///@} Basic

} // namespace cuda
} // namespace cugar
