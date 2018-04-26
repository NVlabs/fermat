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

#include <b40c/segmented_scan/enactor.cuh>
#include <nih/basic/types.h>
#include <nih/basic/numbers.h>

namespace nih {
namespace cuda {

namespace sah {

///
/// Binary functor to merge a pair of quantized bboxes
///
struct merge_op
{
    typedef uint2 result_type;

    FORCE_INLINE NIH_HOST_DEVICE uint2 operator() (const uint2 op1, const uint2 op2) const
    {
        return make_uint2(
            nih::min( op1.x & (1023u <<  0), op2.x & (1023u <<  0) ) |
            nih::min( op1.x & (1023u << 10), op2.x & (1023u << 10) ) |
            nih::min( op1.x & (1023u << 20), op2.x & (1023u << 20) ),
            nih::max( op1.y & (1023u <<  0), op2.y & (1023u <<  0) ) |
            nih::max( op1.y & (1023u << 10), op2.y & (1023u << 10) ) |
            nih::max( op1.y & (1023u << 20), op2.y & (1023u << 20) ) );
    }

    // identity operator
	FORCE_INLINE NIH_HOST_DEVICE uint2 operator()() { return make_uint2( 0xFFFFFFFFu, 0 ); }
};

void inclusive_segmented_scan(
    uint2* dest,
    uint2* src,
    uint32* flags,
    const uint32 n_objects)
{
    b40c::segmented_scan::Enactor segmented_scan_enactor;

    segmented_scan_enactor.Scan<false>(
        dest,
        src,
        flags,
        int(n_objects), sah::merge_op(), sah::merge_op());
}

void exclusive_segmented_scan(
    uint2* dest,
    uint2* src,
    uint32* flags,
    const uint32 n_objects)
{
    b40c::segmented_scan::Enactor segmented_scan_enactor;

    segmented_scan_enactor.Scan<true>(
        dest,
        src,
        flags,
        int(n_objects), sah::merge_op(), sah::merge_op());
}

} // namespace sah

} // namespace cuda
} // namespace nih
