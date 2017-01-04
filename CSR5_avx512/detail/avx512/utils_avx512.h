#ifndef UTILS_AVX512_H
#define UTILS_AVX512_H

#include "common_avx512.h"

struct anonymouslib_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }
    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};

template<typename iT>
iT binary_search_right_boundary_kernel(const iT *d_row_pointer,
                                       const iT  key_input,
                                       const iT  size)
{
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

// exclusive scan using a single thread
template<typename T>
void scan_single(T            *s_scan,
                 const int     l)
{
    T old_val, new_val;

    old_val = s_scan[0];
    s_scan[0] = 0;
    for (int i = 1; i < l; i++)
    {
        new_val = s_scan[i];
        s_scan[i] = old_val + s_scan[i-1];
        old_val = new_val;
    }
}

// inclusive prefix-sum scan
inline __m512d hscan_avx512(__m512d scan512d, __m512d zero512d)
{
    register __m512d t0, t1;

    t0 = _mm512_permutex_pd(scan512d, 0xB1); //_mm512_swizzle_pd(scan512d, _MM_SWIZ_REG_CDAB);
    t1 = _mm512_permutex_pd(t0, 0x4E); //_mm512_swizzle_pd(t0, _MM_SWIZ_REG_BADC);
    t0 = _mm512_mask_blend_pd(0xAA, t1, t0);

    t1 = _mm512_mask_blend_pd(0x0F, zero512d, t0);
    t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));

    scan512d = _mm512_add_pd(scan512d, _mm512_mask_blend_pd(0x11, t0, t1));
    
    t0 = _mm512_permutex_pd(scan512d, 0x4E); //_mm512_swizzle_pd(scan512d, _MM_SWIZ_REG_BADC);
    
    t1 = _mm512_mask_blend_pd(0x0F, zero512d, t0);
    t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));
    
    scan512d = _mm512_add_pd(scan512d, _mm512_mask_blend_pd(0x33, t0, t1));
    
    t1 = _mm512_mask_blend_pd(0x0F, zero512d, scan512d);
    t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));
    scan512d = _mm512_add_pd(scan512d, t1);

    return scan512d;
}

#endif // UTILS_AVX512_H
