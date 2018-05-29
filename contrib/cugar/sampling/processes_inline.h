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


namespace cugar {

//
// Evaluate a DIM-dimensional Brownian bridge of length L at time t,
// using a black-box Gaussian generator.
//
template <typename Generator, uint32 DIM>
Vector<float,DIM> brownian_bridge(
    const uint32        L,
    const uint32        t,
    Generator           gaussian)
{
    // check whether the origin is being retrived
    if (t == 0) return Vector<float,DIM>(0.0f);

    Vector<float,DIM> p1(0.0f);
    Vector<float,DIM> p2 = gaussian.next() * sqrtf(float(L));

    // check whether the last point is being retrived
    if (t == L) return p2;

    // perform a binary search to get to instant t
    const uint32 log_L = log2( L );

    uint32 t1 = 0;
    uint32 t2 = L;

    for (uint32 l = 0; l < log_L; ++l)
    {
        const uint32 M  = 1u << l;
        const uint32 DT = L/M;

        const Vector<float,DIM> pm = (p1 + p2)*0.5f + gaussian.next() * sqrtf(float(L/(2*M)));

        const uint32 tm = (t1 + t2)/2;

        if (t == im)
        {
            p2 = pm;
            break;
        }

        if (t < im)
        {
            p2 = pm;
            t2 = tm;
        }
        else
        {
            p1 = pm;
            t1 = tm;
        }
    }
    return p2;
}

//
// A simple utility function to generate a DIM-dimensional Gaussian point
// using the i-th point of a sample sequence, shifted by a Cranley-Patterson
// rotation.
//
template <uint32 DIM, typename Distribution, typename Sampler_type>
Vector<float,DIM> generate_point(
    Sampler_type&             sampler,
    Distribution&             gaussian)
{
    Vector<float,DIM> pt;
    for (uint32 d = 0; d < DIM; ++d)
        pt[d] = gaussian.next( sampler );

    return pt;
}

//
// Evaluate the i/N-th DIM-dimensional Brownian bridge of length L at time t.
// The bridges are created using L copies of a DIM-dimensional Sobol sequence,
// first permuted and then shifted through Cranley-Patterson rotations.
// The vector of permutations must contain L permutations of the indices [0,N-1],
// and the vector of rotations must contain L * (DIM + (DIM & 1)) entries.
//
template <uint32 DIM, typename Sequence, typename Distribution>
Vector<float,DIM> brownian_bridge(
    Distribution&               gaussian,
    const float                 sigma,
    const uint32                N,
    const uint32                i,
    const uint32                L,
    const uint32                t,
    const Sequence&             sequence)
{
    // check whether the origin is being retrived
    if (t == 0) return Vector<float,DIM>(0.0f);

    const uint32 EVEN_DIM = DIM + (DIM & 1);

    typedef typename Sequence::Sampler_type Sampler_type;

    Vector<float,DIM> p1(0.0f);
    Vector<float,DIM> p2 = generate_point<DIM>( sequence.instance( i, 0 ), gaussian ) * sqrtf(float(L)) * sigma;

    // check whether the last point is being retrived
    if (t == L) return p2;

    // perform a binary search to get to instant t
    const uint32 log_L = log2( L );

    uint32 t1 = 0;
    uint32 t2 = L;

    for (uint32 l = 0; l < log_L; ++l)
    {
        const uint32 M  = 1u << l;
        const uint32 DT = L/M;

        const uint32 tm = (t1 + t2)/2;

        const Vector<float,DIM> pm = (p1 + p2)*0.5f +
            generate_point<DIM>( sequence.instance( i, tm ), gaussian ) * sqrtf(float(L/(2*M))) * sigma;

        if (t == tm)
        {
            p2 = pm;
            break;
        }

        if (t < tm)
        {
            p2 = pm;
            t2 = tm;
        }
        else
        {
            p1 = pm;
            t1 = tm;
        }
    }
    return p2;
}

} // namespace cugar
