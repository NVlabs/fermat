/*
 * Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <cugar/basic/cuda/sort.h>
#include <cugar/basic/thrust_view.h>
#include <cugar/basic/vector.h>
#include <cub/cub.cuh>

namespace cugar {
namespace cuda {

#define USE_SMALL_SORT_KERNEL

// Fast intra-CTA radix sort kernel for small problem sizes.
#ifdef USE_SMALL_SORT_KERNEL
  template <class K, class V, int NUM_THREADS, int ELEMS_PER_THREAD, int RADIX_BITS>
  __launch_bounds__(NUM_THREADS, 1) static __global__
  void small_sort_kernel(void* d_keys, void* d_values, unsigned int num_elements)
  {
    K keys[ELEMS_PER_THREAD];
    V values[ELEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
    {
      int idx = threadIdx.x * ELEMS_PER_THREAD + i;
      if (idx < num_elements)
      {
      #if __CUDA_ARCH__ >= 350
        keys[i]   = __ldg(&((K*)d_keys)[idx]);
        values[i] = __ldg(&((V*)d_values)[idx]);
      #else
        keys[i]   = ((K*)d_keys)[idx];
        values[i] = ((V*)d_values)[idx];
      #endif
      }
      else
      {
        keys[i]   = ~(K)0;
        values[i] = ~(V)0;
      }
    }

    __syncthreads();

    cub::BlockRadixSort
      <K, NUM_THREADS, ELEMS_PER_THREAD, V, RADIX_BITS>
      ().SortBlockedToStriped(keys, values);

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
    {
      int idx = threadIdx.x + i * NUM_THREADS;
      if (idx < num_elements)
      {
        ((K*)d_keys)[idx]   = keys[i];
        ((V*)d_values)[idx] = values[i];
      }
    }
  }
#endif

  // Individual specialization of small_sort_kernel().
  struct SmallSortKernelSpec
  {
    void    (*device_func)(void* d_keys, void* d_values, unsigned int num_elements);
    size_t  key_bytes;
    size_t  value_bytes;
    int     num_threads;
    int     max_elements;
  };

  // Choose the appropriate specialization of small_sort_kernel() for the given parameters.
  static SmallSortKernelSpec choose_small_sort_kernel(size_t key_bytes, size_t value_bytes, unsigned int num_elements)
  {
#ifdef USE_SMALL_SORT_KERNEL
#  define SMALL_SORT_KERNEL_SPEC(K, V, NUM_WARPS, ELEMS_PER_THREAD, RADIX_BITS) \
    { small_sort_kernel<K, V, NUM_WARPS * 32, ELEMS_PER_THREAD, RADIX_BITS>, sizeof(K), sizeof(V), NUM_WARPS * 32, NUM_WARPS * 32 * ELEMS_PER_THREAD }

    static const SmallSortKernelSpec kernels[] =
    {
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 4,  1,  5), // 0.016 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 4,  5,  5), // 0.018 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 8,  5,  5), // 0.020 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 8,  10, 5), // 0.025 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 12, 10, 5), // 0.029 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 16, 10, 5), // 0.033 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 20, 10, 5), // 0.038 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned int,       unsigned int, 20, 19, 5), // 0.054 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 4,  1,  5), // 0.022 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 4,  5,  5), // 0.026 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 8,  5,  5), // 0.030 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 8,  10, 5), // 0.040 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 12, 10, 5), // 0.049 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 16, 10, 5), // 0.058 ms on GTX980
      SMALL_SORT_KERNEL_SPEC(unsigned long long, unsigned int, 19, 10, 5), // 0.069 ms on GTX980
    };

    for (int i = 0; i < (int)(sizeof(kernels) / sizeof(kernels[0])); i++)
    {
      const SmallSortKernelSpec& k = kernels[i];
      if (k.key_bytes == key_bytes && k.value_bytes == value_bytes && k.max_elements >= num_elements)
        return k;
    }
#endif

    SmallSortKernelSpec noKernel = {};
    return noKernel;
  }

SortEnactor::SortEnactor()
{
    m_impl = NULL; // we might want to use this later for the temp storage
}
SortEnactor::~SortEnactor()
{
}

void SortEnactor::sort(const uint32 count, SortBuffers<uint8*, uint32*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint8>  key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}

void SortEnactor::sort(const uint32 count, SortBuffers<uint16*,uint32*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint16> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint32*,uint32*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    SmallSortKernelSpec kernel = choose_small_sort_kernel(sizeof(unsigned int), sizeof(unsigned int), count);
    if (kernel.device_func)
    {
      kernel.device_func<<<1, kernel.num_threads>>>(buffers.keys[buffers.selector], buffers.values[buffers.selector], count);
      return;
    }

    cub::DoubleBuffer<uint32> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint32*,uint64*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint32> key_buffers;
    cub::DoubleBuffer<uint64> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint64*,uint32*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    SmallSortKernelSpec kernel = choose_small_sort_kernel(sizeof(unsigned long long), sizeof(unsigned int), count);
    if (kernel.device_func)
    {
      kernel.device_func<<<1, kernel.num_threads>>>(buffers.keys[buffers.selector], buffers.values[buffers.selector], count);
      return;
    }

    cub::DoubleBuffer<uint64> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint64*,uint64*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint64> key_buffers;
    cub::DoubleBuffer<uint64> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}

void SortEnactor::sort(const uint32 count, SortBuffers<uint8*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint8> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;

}
void SortEnactor::sort(const uint32 count, SortBuffers<uint16*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint16> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint32*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint32> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint64*>& buffers, const uint32 begin_bit, const uint32 end_bit)
{
    cub::DoubleBuffer<uint64> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    caching_device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( cugar::raw_pointer( d_temp ), temp_storage_bytes, key_buffers, count, begin_bit, end_bit );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}

} // namespace cuda
} // namespace cugar
