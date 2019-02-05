//
//Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
//
//NVIDIA Corporation and its licensors retain all intellectual property and
//proprietary rights in and to this software, related documentation and any
//modifications thereto.  Any use, reproduction, disclosure or distribution of
//this software and related documentation without an express license agreement
//from NVIDIA Corporation is strictly prohibited.
//
//TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
//OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
//MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
//NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
//CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
//LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
//INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGES
//

#pragma once

// ------------------------------------------------------------------------- //
//
// Declaration of classes used to store intersections.
//
// ------------------------------------------------------------------------- //

#include <types.h>
#include <cugar/basic/memory_arena.h>
#include <cugar/basic/cuda/warp_atomics.h>

///@addtogroup Fermat
///@{

///@addtogroup WavefrontQueues
///@{

struct QueueDescriptor
{
	enum Type
	{
		NONE	= 0,
		UINT	= 1,
		UINT2	= 2,
		UINT4	= 3,
		FLOAT	= 4,
		FLOAT2	= 5,
		FLOAT4	= 6,
	};

	FERMAT_HOST_DEVICE
	QueueDescriptor()
	{
		for (uint32 i = 0; i < 16; ++i)
			desc[i] = NONE;
	}

	FERMAT_HOST_DEVICE
	uint32 size(const uint32 i) const
	{
		switch (desc[i])
		{
		case UINT:		return 4u;
		case UINT2:		return 8u;
		case UINT4:		return 16u;
		case FLOAT:		return 4u;
		case FLOAT2:	return 8u;
		case FLOAT4:	return 16u;
		default :		return 0u;
		};
	}

	template <typename TUserData>
	void setup(QueueDescriptor& user)
	{
		// invoke the serialization method on the user object
		serialize( *this, 0u, user );
	}

	Type desc[16];
};

template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, uint  v) { queue.desc[m]  = QueueDescriptor::UINT; }
template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, uint2 v) { queue.desc[m]  = QueueDescriptor::UINT2; }
template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, uint4 v) { queue.desc[m]  = QueueDescriptor::UINT4; }
template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, float  v) { queue.desc[m] = QueueDescriptor::FLOAT; }
template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, float2 v) { queue.desc[m] = QueueDescriptor::FLOAT2; }
template <uint32 m> void serialize_member(QueueDescriptor& queue, uint32 i, float4 v) { queue.desc[m] = QueueDescriptor::FLOAT4; }

struct WavefrontQueue
{
	typedef uint32 Entry;

	/// constructor
	///
	FERMAT_HOST_DEVICE
	WavefrontQueue() : ptr(NULL), size(NULL), capacity(0) {}

	/// setup the descriptor
	///
	FERMAT_HOST_DEVICE
	void setup(const QueueDescriptor& _desc, const uint32 _capacity)
	{
		capacity = _capacity;

		uint32 offset = 0;
		for (uint32 i = 0; i < 16; ++i)
		{
			const uint32 el_size = _desc.size(i);

			// take care of the element alignment
			offset = cugar::round_i( offset, el_size );

			// record the offset
			offsets[i] = offset;

			// increase the offset
			offset += el_size * capacity;
		}
	}

	/// return the element size
	///
	FERMAT_HOST_DEVICE
	uint32 byte_size() const { return offsets[ 15 ]; }

	/// setup the baking storage
	///
	FERMAT_HOST_DEVICE
	void alloc(uint8* _ptr, uint32* _size) { ptr = _ptr; size = _size; }

	/// return the offset of the i-th entry
	///
	template <uint32 m, typename T>
	FERMAT_HOST_DEVICE
	T* member_base() const { return reinterpret_cast<T*>(ptr + offsets[m]); }

	/// return the offset of the i-th entry
	///
	template <uint32 m, typename T>
	FERMAT_HOST_DEVICE
	const T& member(const uint32 slot) const { return member_base<m>()[slot]; }

	/// return the offset of the i-th entry
	///
	template <uint32 m, typename T>
	FERMAT_HOST_DEVICE
	T& member(const uint32 slot) { return member_base<m>()[slot]; }

	/// return a free slot
	///
	FERMAT_DEVICE
	uint32 append()
	{
		return cugar::cuda::warp_increment(size);
	}

	/// explicitly write the size
	///
	FERMAT_HOST_DEVICE
	void set_size(const uint32 _size) { *size = _size; }
		
	uint8*  ptr;
	uint32* size;
	uint32  capacity;
	uint32  offsets[16];
};

template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, uint   v) { queue.member<m,uint>(slot)   = v; }
template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, uint2  v) { queue.member<m,uint2>(slot)  = v; }
template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, uint4  v) { queue.member<m,uint4>(slot)  = v; }
template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, float  v) { queue.member<m,float>(slot)  = v; }
template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, float2 v) { queue.member<m,float2>(slot) = v; }
template <uint32 m> void serialize_member(WavefrontQueue& queue, const uint32 slot, float4 v) { queue.member<m,float4>(slot) = v; }

template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, uint&   v) { v = queue.member<m,uint>(slot); }
template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, uint2&  v) { v = queue.member<m,uint2>(slot); }
template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, uint4&  v) { v = queue.member<m,uint4>(slot); }
template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, float&  v) { v = queue.member<m,float>(slot); }
template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, float2& v) { v = queue.member<m,float2>(slot); }
template <uint32 m> void deserialize_member(const WavefrontQueue& queue, const uint32 slot, float4& v) { v = queue.member<m,float4>(slot); }

/// a ray queue is a regular queue where the header of the payload is augmented with a Ray and a Hit data structure
///
struct RayWavefrontQueue : WavefrontQueue
{
	/// setup the baking storage
	///
	FERMAT_HOST_DEVICE
	void alloc(uint8* _ptr, uint32* _size) { ptr = _ptr + sizeof(float4)*3*capacity; size = _size; ray_ptr = _ptr; }

	FERMAT_HOST_DEVICE
	Ray* rays() const { return (Ray*)ray_ptr; }

	FERMAT_HOST_DEVICE
	Hit* hits() const { return (Hit*)(ray_ptr + sizeof(float4)*2*capacity); }

	uint8* ray_ptr;
};

FERMAT_HOST_DEVICE
void serialize(RayWavefrontQueue& queue, const uint32 i, const Ray& ray)
{
	reinterpret_cast<float4*>(queue.ray_ptr)[2*i + 0] = make_float4( ray.origin.x, ray.origin.y, ray.origin.x, ray.tmin );
	reinterpret_cast<float4*>(queue.ray_ptr)[2*i + 1] = make_float4( ray.dir.x, ray.dir.y, ray.dir.x, ray.tmax );
}
FERMAT_HOST_DEVICE
void deserialize(const RayWavefrontQueue& queue, const uint32 i, Ray& ray, Hit& hit)
{
	{
		const float4 val = reinterpret_cast<const float4*>(queue.ray_ptr)[2*i + 0];
		ray.origin.x = val.x;
		ray.origin.y = val.y;
		ray.origin.z = val.z;
		ray.tmin     = val.w;
	}
	{
		const float4 val = reinterpret_cast<const float4*>(queue.ray_ptr)[2*i + 1];
		ray.dir.x = val.x;
		ray.dir.y = val.y;
		ray.dir.z = val.z;
		ray.tmax  = val.w;
	}
	{
		const float4 val = reinterpret_cast<const float4*>(queue.ray_ptr + sizeof(float4)*2*queue.capacity)[i];
		hit.t		= val.x;
		hit.triId	= val.y;
		hit.u		= val.z;
		hit.v		= val.w;
	}
}

#if 0
	struct ScatteringPayload
	{
		PixelInfo	pixel_info;
		CacheInfo	cache_info;
		CacheInfo	prev_cache_info;
		float4		weight;
		float2		cone;
		float		roughness;
	};
	template <typename TQueue, typename TQueueEntry>
	FERMAT_HOST_DEVICE
	void serialize(TQueue& queue, const TQueueEntry& i, const ScatteringPayload& payload)
	{
		serialize_member<0u>( queue, i, make_uint4(
			payload.pixel_info.packed,
			payload.cache_info.packed,
			payload.prev_cache_info.packed,
			cugar::binary_cast<uint32>(payload.roughness)) );

		serialize_member<1u>( queue, i, payload.cone );
		serialize_member<2u>( queue, i, payload.weight );
	}
	template <typename TQueue, typename TQueueEntry>
	FERMAT_HOST_DEVICE
	void deserialize(const TQueue& queue, const TQueueEntry& i, ScatteringPayload& payload)
	{
		uint4 u;
		deserialize_member<0u>( queue, i, u );
		payload.pixel_info		= PixelInfo(u.x);
		payload.cache_info		= CacheInfo(u.y);
		payload.prev_cache_info = CacheInfo(u.z);
		payload.roughness		= cugar::binary_cast<float>(u.w);

		deserialize_member<1u>( queue, i, payload.cone );
		deserialize_member<2u>( queue, i, payload.weight );
	}
#endif

///@} WavefrontQueues
///@} Fermat
