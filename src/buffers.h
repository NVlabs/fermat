/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "types.h"

#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    std::cerr << "Error on line " << __LINE__ << ":"                           \
              << cudaGetErrorString( err__ ) << std::endl;                     \
    exit(1);                                                                   \
  }                                                                            \
}

//------------------------------------------------------------------------------
#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
  {                                                                            \
  const char* err_string;                                                      \
  rtpContextGetLastErrorString( context, &err_string );                        \
  std::cerr << "Error on line " << __LINE__ << ": '"                           \
  << err_string                                                                \
  << "' (" << res__ << ")" << std::endl;                                       \
  exit(1);                                                                     \
  }                                                                            \
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
enum PageLockedState
{
  UNLOCKED,
  LOCKED
};

enum BufferType
{
	CUDA_BUFFER = RTP_BUFFER_TYPE_CUDA_LINEAR,
	HOST_BUFFER = RTP_BUFFER_TYPE_HOST,
};

template<typename T> class Buffer;
template<typename T> class ManagedBuffer;

//------------------------------------------------------------------------------
//
// Simple buffer class for buffers residing in managed memory
//
template<typename T>
class ManagedBuffer
{
public:
	ManagedBuffer(size_t count = 0)
		: m_ptr(0), m_count(0)
	{
		alloc(count);
	}

	ManagedBuffer(const ManagedBuffer<T>& src)
		: m_ptr(0), m_count(0)
	{
		this->operator=(src);
	}

	ManagedBuffer<T>& operator=(const ManagedBuffer<T>& src)
	{
		alloc(src.count());

		memcpy(m_ptr, src.ptr(), sizeInBytes());
		return *this;
	}

	// Allocate without changing type
	void alloc(size_t count)
	{
		free();

		cudaMallocManaged(&m_ptr, sizeof(T)*count);

		m_count = count;
	}

	void free()
	{
		if (m_ptr)
			cudaFree(m_ptr);

		m_ptr = 0;
		m_count = 0;
	}

	~ManagedBuffer() { free(); }

	size_t count()       const { return m_count; }
	size_t sizeInBytes() const { return m_count * sizeof(T); }
	const T* ptr()       const { return m_ptr; }
	T* ptr()                   { return m_ptr; }

protected:
	T*				m_ptr;
	size_t			m_count;
};

//------------------------------------------------------------------------------
//
// Simple buffer class for buffers on the host or CUDA device
//
template<typename T>
class Buffer
{
public:
	Buffer(size_t count = 0, BufferType type = HOST_BUFFER, PageLockedState pageLockedState = UNLOCKED)
    : m_ptr( 0 ),
	  m_device( 0 ),
      m_count( 0 ),
	  m_type( type ),
      m_pageLockedState( pageLockedState )
	{
		alloc( count, type, pageLockedState );
	}

	Buffer(const Buffer<T>& src) : Buffer(0,src.type())
	{
		this->operator=(src);
	}

	Buffer<T>& operator=(const Buffer<T>& src)
	{
		alloc(src.count());

		if (src.type() == HOST_BUFFER)
		{
			if (type() == HOST_BUFFER)
				memcpy(m_ptr, src.ptr(), sizeInBytes());
			else
				cudaMemcpy(m_ptr, src.ptr(), sizeInBytes(), cudaMemcpyHostToDevice);
		}
		else
		{
			if (type() == HOST_BUFFER)
				cudaMemcpy(m_ptr, src.ptr(), sizeInBytes(), cudaMemcpyDeviceToHost);
			else
				cudaMemcpy(m_ptr, src.ptr(), sizeInBytes(), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	Buffer<T>& operator=(const ManagedBuffer<T>& src)
	{
		alloc(src.count());

		if (type() == HOST_BUFFER)
			memcpy(m_ptr, src.ptr(), sizeInBytes());
		else
			cudaMemcpy(m_ptr, src.ptr(), sizeInBytes(), cudaMemcpyHostToDevice);

		return *this;
	}

	// Allocate without changing type
	void alloc( size_t count )
	{
		alloc( count, m_type, m_pageLockedState );
	}

	void alloc(size_t count, BufferType type, PageLockedState pageLockedState = UNLOCKED)
	{
		if (m_ptr)
			free();

		m_type = type;
		m_count = count;
		if (m_count > 0) 
		{
			if (m_type == HOST_BUFFER)
			{
				m_ptr = new T[m_count];
				if( pageLockedState == LOCKED )
					rtpHostBufferLock( m_ptr, sizeInBytes() ); // for improved transfer performance
				m_pageLockedState = pageLockedState;
			}
			else
			{
				CHK_CUDA( cudaGetDevice( &m_device ) );
				CHK_CUDA( cudaMalloc( &m_ptr, sizeInBytes() ) );
			}
		}
	}

	void resize(const size_t count)
	{
		Buffer<T> buffer( count, m_type, m_pageLockedState );
		buffer.copy_from( count < m_count ? count : m_count, m_type, m_ptr );

		swap( buffer );
	}

	void copy_from(const size_t count, const BufferType src_type, const T* src, const uint32 dst_offset = 0)
	{
		if (count == 0)
			return;

		if (m_type == HOST_BUFFER)
		{
			if (src_type == HOST_BUFFER)
				memcpy( m_ptr + dst_offset, src, sizeof(T)*count );
			else
			{
				CHK_CUDA( cudaMemcpy( m_ptr + dst_offset, src, sizeof(T)*count, cudaMemcpyDeviceToHost ) );
			}
		}
		else
		{
			if (src_type == HOST_BUFFER)
			{
				CHK_CUDA( cudaMemcpy( m_ptr + dst_offset, src, sizeof(T)*count, cudaMemcpyHostToDevice ) );
			}
			else
			{
				CHK_CUDA( cudaMemcpy( m_ptr + dst_offset, src, sizeof(T)*count, cudaMemcpyDeviceToDevice ) );
			}
		}
	}

	void clear(const uint8 byte)
	{
		if (m_type == HOST_BUFFER)
			memset(m_ptr, byte, sizeInBytes());
		else
			cudaMemset(m_ptr, byte, sizeInBytes());
	}

	void free()
	{
		if (m_ptr)
		{
			if (m_type == HOST_BUFFER)
			{
				if (m_pageLockedState == LOCKED)
					rtpHostBufferUnlock(m_ptr);
				delete[] m_ptr;
			}
			else
			{
				int oldDevice;
				CHK_CUDA(cudaGetDevice(&oldDevice));
				CHK_CUDA(cudaSetDevice(m_device));
				CHK_CUDA(cudaFree(m_ptr));
				CHK_CUDA(cudaSetDevice(oldDevice));
			}
		}

		m_ptr = 0;
		m_count = 0;
	}

	~Buffer() { free();	}

	size_t count()       const { return m_count; }
	size_t sizeInBytes() const { return m_count * sizeof(T); }
	const T* ptr()       const { return m_ptr; }
	T* ptr()                   { return m_ptr; }
	BufferType type() const { return m_type; }

    T operator[] (const size_t i) const
    {
        if (m_type == RTP_BUFFER_TYPE_HOST)
            return m_ptr[i];
        else
        {
            T t;
            cudaMemcpy( &t, m_ptr + i, sizeof(T), cudaMemcpyDeviceToHost);
            return t;
        }
    }

	void set(const size_t i, const T val)
	{
		if (m_type == HOST_BUFFER)
			m_ptr[i] = val;
		else
			cudaMemcpy(m_ptr + i, &val, sizeof(T), cudaMemcpyHostToDevice);
	}
    
    void swap(Buffer<T>& buf)
	{
		std::swap(m_type, buf.m_type);
		std::swap(m_ptr, buf.m_ptr);
		std::swap(m_device, buf.m_device);
		std::swap(m_count, buf.m_count);
		std::swap(m_pageLockedState, buf.m_pageLockedState);
	}

protected:
	BufferType		m_type;
	T*				m_ptr;
	int				m_device;
	size_t			m_count;
	PageLockedState	m_pageLockedState;
};

//------------------------------------------------------------------------------
//
// Simple buffer class for buffers on the host or CUDA device
//
template<BufferType TYPE, typename T>
class DomainBuffer : public Buffer<T>
{
public:
	DomainBuffer(size_t count = 0, PageLockedState pageLockedState = UNLOCKED)
		: Buffer(count, TYPE, pageLockedState)
	{}

	template <BufferType UTYPE>
	DomainBuffer(const DomainBuffer<UTYPE, T>& src) : Buffer(0, TYPE)
	{
		this->operator=(src);
	}

	DomainBuffer<TYPE, T>& operator=(const Buffer<T>& src)
	{
		this->Buffer<T>::operator=(src);
		return *this;
	}
	DomainBuffer<TYPE, T>& operator=(const ManagedBuffer<T>& src)
	{
		this->Buffer<T>::operator=(src);
		return *this;
	}
};


inline float3 ptr_to_float3(const float* v) { return make_float3(v[0], v[1], v[2]); }
