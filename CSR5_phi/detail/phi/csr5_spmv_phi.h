#ifndef CSR5_SPMV_PHI_H
#define CSR5_SPMV_PHI_H

#include "common_phi.h"
#include "utils_phi.h"

template<typename iT, typename vT>
__attribute__ ((target(mic)))
void partition_fast_track(const vT           *d_value_partition,
                                 const vT           *d_x,
                                 const iT           *d_column_index_partition,
                                 vT                 *d_calibrator,
                                 vT                 *d_y,
                                 const iT            row_start,
                                 const iT            par_id,
                                 const int           tid,
                                 const iT            start_row_start,
                                 const vT            alpha,
                                 const int           sigma,
                                 const int           stride_vT,
                                 const bool          direct)
{

    __m512d sum512d = _mm512_setzero_pd();
    __m512d value512d, x512d;
    __m512i column_index512i;
    
    #pragma unroll(ANONYMOUSLIB_CSR5_SIGMA)
    for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
    {
        value512d = _mm512_load_pd(&d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
        column_index512i = (i % 2) ?
                    _mm512_permute4f128_epi32(column_index512i, _MM_PERM_BADC) :
                    _mm512_load_epi32(&d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
        x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);
        sum512d = _mm512_fmadd_pd(value512d, x512d, sum512d);
    }

    vT sum = _mm512_reduce_add_pd(sum512d);

    if (row_start == start_row_start && !direct)
        d_calibrator[tid * stride_vT] += sum;
    else
    {
        if(direct)
            d_y[row_start] = sum;
        else
            d_y[row_start] += sum;
    }
}

template<typename iT, typename uiT, typename vT>
__attribute__ ((target(mic)))
void spmv_csr5_compute_kernel(const iT           *d_column_index,
                              const vT           *d_value,
                              const iT           *d_row_pointer,
                              const vT           *d_x,
                              const uiT          *d_partition_pointer,
                              const uiT          *d_partition_descriptor,
                              const iT           *d_partition_descriptor_offset_pointer,
                              const iT           *d_partition_descriptor_offset,
                              vT                 *d_calibrator,
                              vT                 *d_y,
                              const iT            p,
                              const int           num_packet,
                              const int           bit_y_offset,
                              const int           bit_scansum_offset,
                              const vT            alpha,
                              const int           c_sigma)
{
    const int num_thread = omp_get_max_threads();
    const int chunk = ceil((double)(p-1) / (double)num_thread);

    const __m512d c_zero512d        = _mm512_setzero_pd();
    const __m512i c_one512i         = _mm512_set1_epi32(1);

    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    const int num_thread_active = ceil((p-1.0)/chunk);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        iT start_row_start = tid < num_thread_active ? d_partition_pointer[tid * chunk] & 0x7FFFFFFF : 0;

        __m512d value512d;
        __m512d x512d;
        __m512i column_index512i;

        __m512d sum512d = c_zero512d;
        __m512d tmp_sum512d = c_zero512d;
        __m512d first_sum512d = c_zero512d;
        __m512d last_sum512d = c_zero512d;

        __m512i scansum_offset512i;
        __m512i y_offset512i;
        __m512i y_idx512i;
        __m512i start512i;
        __m512i stop512i;
        __m512i descriptor512i;

        __mmask16 local_bit16;
        __mmask16 direct16;

        #pragma omp for schedule(static, chunk)
        #pragma noprefetch
        for (int par_id = 0; par_id < p - 1; par_id++)
        {
            const int prefetch_distance = 1;
            const iT *d_column_index_partition_prefetch = &d_column_index[(par_id + prefetch_distance) * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];
            const vT *d_value_partition = &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];

            // prefetch
            if (par_id < tid * chunk + chunk - prefetch_distance) // && row_start_prefetch != row_stop_prefetch)
            {

                #pragma unroll(ANONYMOUSLIB_CSR5_SIGMA)
                for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
                {
                    int idx0 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA];
                    int idx1 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 1];
                    int idx2 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 2];
                    int idx3 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 3];
                    int idx4 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 4];
                    int idx5 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 5];
                    int idx6 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 6];
                    int idx7 = d_column_index_partition_prefetch[i * ANONYMOUSLIB_CSR5_OMEGA + 7];

                    _mm_prefetch((const char *)&d_x[idx0], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx1], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx2], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx3], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx4], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx5], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx6], _MM_HINT_T1);
                    _mm_prefetch((const char *)&d_x[idx7], _MM_HINT_T1);

                }
            }

            const int *d_column_index_partition = &d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma];

            uiT row_start     = d_partition_pointer[par_id];
            const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

            if (row_start == row_stop) // fast track through reduction
            {
                // check whether the the partition contains the first element of row "row_start"
                // => we are the first writing data to d_y[row_start]
                bool fast_direct = (d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet] >>
                                                    (31 - (bit_y_offset + bit_scansum_offset)) & 0x1);
                partition_fast_track<iT, vT>
                        (d_value_partition, d_x, d_column_index_partition,
                         d_calibrator, d_y, row_start, par_id,
                         tid, start_row_start, alpha, c_sigma, stride_vT, fast_direct);
            }
            else // normal track for all the other partitions
            {
                const bool empty_rows = (row_start >> 31) & 0x1;
                row_start &= 0x7FFFFFFF;

                vT *d_y_local = &d_y[row_start+1];
                const int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

                __mmask8 storemask8;

                first_sum512d = c_zero512d;
                stop512i = _mm512_castpd_si512(first_sum512d);
#if ANONYMOUSLIB_CSR5_SIGMA > 20
                const uiT *d_partition_descriptor_partition = &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet];
                descriptor512i = _mm512_mask_i32gather_epi32(stop512i, 0xFF, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), 
                                                             d_partition_descriptor_partition, 4);

#else
                if(par_id % 2)
                {
                    descriptor512i = _mm512_load_epi32(&d_partition_descriptor[(par_id-1) * ANONYMOUSLIB_CSR5_OMEGA * num_packet]);
                    descriptor512i = _mm512_permute4f128_epi32(descriptor512i, _MM_PERM_BADC);
                }
                else
                    descriptor512i = _mm512_load_epi32(&d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet]);
#endif

                y_offset512i = _mm512_srli_epi32(descriptor512i, 32 - bit_y_offset);
                scansum_offset512i = _mm512_slli_epi32(descriptor512i, bit_y_offset);
                scansum_offset512i = _mm512_srli_epi32(scansum_offset512i, 32 - bit_scansum_offset);

                descriptor512i = _mm512_slli_epi32(descriptor512i, bit_y_offset + bit_scansum_offset);

                local_bit16 = _mm512_cmp_epi32_mask(_mm512_srli_epi32(descriptor512i, 31), c_one512i, _MM_CMPINT_EQ);
                
                // remember if the first element of this partition is the first element of a new row
                bool first_direct = false;
                if(local_bit16 & 0x1)
                    first_direct = true;
                    
                // remember if the first element of the first partition of the current thread is the first element of a new row
                bool first_all_direct = false;
                if(par_id == tid * chunk)
                    first_all_direct = first_direct;
                    
                local_bit16 |= 0x1;

                start512i = _mm512_mask_blend_epi32(local_bit16, c_one512i, _mm512_setzero_epi32());
                direct16 = _mm512_kand(local_bit16, 0xFE);

                value512d = _mm512_load_pd(d_value_partition);

                column_index512i = _mm512_load_epi32(d_column_index_partition);
                x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);

                sum512d = _mm512_mul_pd(value512d, x512d);

                // step 1. thread-level seg sum
#if ANONYMOUSLIB_CSR5_SIGMA > 20
                int ly = 0;
#endif
                #pragma unroll(ANONYMOUSLIB_CSR5_SIGMA-1)
                for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)
                {
                    column_index512i = (i % 2) ?
                                _mm512_permute4f128_epi32(column_index512i, _MM_PERM_BADC) :
                                _mm512_load_epi32(&d_column_index_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);

#if ANONYMOUSLIB_CSR5_SIGMA > 20
                    int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

                    if (!(ly || norm_i) || (ly && !(norm_i % 32)))
                    {
                        ly++;
                        descriptor512i = _mm512_mask_i32gather_epi32(stop512i, 0xFF, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), 
                                                                     &d_partition_descriptor_partition[ly * ANONYMOUSLIB_CSR5_OMEGA], 4);
                    }
                    norm_i = !ly ? i : norm_i;
                    norm_i = 31 - norm_i % 32;

                    local_bit16 = _mm512_cmp_epi32_mask(_mm512_and_epi32(_mm512_srli_epi32(descriptor512i, norm_i), c_one512i), c_one512i, _MM_CMPINT_EQ);
#else
                    local_bit16 = _mm512_cmp_epi32_mask(_mm512_and_epi32(_mm512_srli_epi32(descriptor512i, 31-i), c_one512i), c_one512i, _MM_CMPINT_EQ);
#endif

                    if (local_bit16 & 0xFF)
                    {

                        //// mask scatter
                        storemask8 = _mm512_kand(direct16, local_bit16) & 0xFF;
                        if (storemask8)
                        {
                            y_idx512i = empty_rows ? 
                                    _mm512_mask_i32gather_epi32(y_offset512i, storemask8, y_offset512i, &d_partition_descriptor_offset[offset_pointer], 4) : 
                                    y_offset512i;
                            _mm512_mask_i32loscatter_pd(d_y_local, storemask8, y_idx512i, sum512d, 8);
                            y_offset512i = _mm512_mask_add_epi32(y_offset512i, storemask8, y_offset512i, c_one512i);
                        }

                        storemask8 = _mm512_kandn(direct16, local_bit16) & 0xFF;
                        first_sum512d = _mm512_mask_blend_pd(storemask8, first_sum512d, sum512d);

                        storemask8 = local_bit16 & 0xFF;
                        sum512d = _mm512_mask_blend_pd(storemask8, sum512d, c_zero512d);

                        direct16 = _mm512_kor(local_bit16, direct16);
                        stop512i = _mm512_mask_add_epi32(stop512i, direct16, stop512i, c_one512i);
                    }

                    value512d = _mm512_load_pd(&d_value_partition[i * ANONYMOUSLIB_CSR5_OMEGA]);
                    x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);
                    sum512d = _mm512_fmadd_pd(value512d, x512d, sum512d);

                }

                storemask8 = direct16 & 0xFF;
                first_sum512d = _mm512_mask_blend_pd(storemask8, sum512d, first_sum512d);

                last_sum512d = sum512d;

                storemask8 = _mm512_cmp_epi32_mask(start512i, c_one512i, _MM_CMPINT_EQ) & 0xFF;
                sum512d = _mm512_mask_blend_pd(storemask8, c_zero512d, first_sum512d);

                sum512d = _mm512_castsi512_pd(_mm512_permutevar_epi32(_mm512_set_epi32(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2), _mm512_castpd_si512(sum512d)));
                sum512d = _mm512_mask_blend_pd(0x80, sum512d, c_zero512d);

                tmp_sum512d = sum512d;
                sum512d = hscan_phi(sum512d, c_zero512d);

                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
                scansum_offset512i = _mm512_permutevar_epi32(_mm512_set_epi32(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0), scansum_offset512i);
                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, scansum_offset512i);
                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, _mm512_set_epi32(1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0));

                sum512d = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_permutevar_epi32(scansum_offset512i, _mm512_castpd_si512(sum512d))), sum512d);
                sum512d = _mm512_add_pd(sum512d, tmp_sum512d);

                storemask8 = _mm512_cmp_epi32_mask(start512i, stop512i, _MM_CMPINT_LE) & 0xFF;
                last_sum512d = _mm512_add_pd(last_sum512d, _mm512_mask_blend_pd(storemask8, c_zero512d, sum512d));

                // mask scatter
                storemask8 = direct16 & 0xFF;
                if (storemask8)
                {
                    y_idx512i = empty_rows ? 
                                _mm512_mask_i32gather_epi32(y_offset512i, direct16, y_offset512i, &d_partition_descriptor_offset[offset_pointer], 4) : 
                                y_offset512i;
                    _mm512_mask_i32loscatter_pd(d_y_local, storemask8, y_idx512i, last_sum512d, 8);
                }

                sum512d = _mm512_mask_blend_pd(storemask8, last_sum512d, first_sum512d);
                sum512d = _mm512_mask_blend_pd(0x1, c_zero512d, sum512d);
                vT sum = _mm512_mask_reduce_add_pd(0x1, sum512d);

                if (row_start == start_row_start && !first_all_direct)
                    d_calibrator[tid * stride_vT] += sum;
                else
                {
                    if(first_direct)
                        d_y[row_start] = sum;
                    else
                        d_y[row_start] += sum;
                }

            }
        }
    }
}

template<typename iT, typename uiT, typename vT>
__attribute__ ((target(mic)))
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                vT        *d_calibrator,
                                vT        *d_y,
                                const iT   p)
{
    const int num_thread = omp_get_max_threads();
    const int chunk = ceil((double)(p-1) / (double)num_thread);
    const int stride_vT = ANONYMOUSLIB_X86_CACHELINE / sizeof(vT);
    // calculate the number of maximal active threads (for a static loop scheduling with size chunk)
    int num_thread_active = ceil((p-1.0)/chunk);
    int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

    for (int i = 0; i < num_cali; i++)
    {
        d_y[(d_partition_pointer[i * chunk] << 1) >> 1] += d_calibrator[i * stride_vT];
    }
}

template<typename iT, typename uiT, typename vT>
__attribute__ ((target(mic)))
void spmv_csr5_tail_partition_kernel(const iT           *d_row_pointer,
                                     const iT           *d_column_index,
                                     const vT           *d_value,
                                     const vT           *d_x,
                                     vT                 *d_y,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const iT            m,
                                     const int           sigma,
                                     const vT            alpha)
{
    const iT index_first_element_tail = (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma;
    
    for (iT row_id = tail_partition_start; row_id < m; row_id++)
    {
        const iT idx_start = row_id == tail_partition_start ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
        const iT idx_stop  = d_row_pointer[row_id + 1];

        vT sum = 0;
        for (iT idx = idx_start; idx < idx_stop; idx++)
            sum += d_value[idx] * d_x[d_column_index[idx]];// * alpha;

        if(row_id == tail_partition_start && d_row_pointer[row_id] != index_first_element_tail)
        {
            d_y[row_id] = d_y[row_id] + sum;
        }
        else
        {
            d_y[row_id] = sum;
        }
    }
}


template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
__attribute__ ((target(mic)))
void csr5_spmv(const int                 sigma,
              const ANONYMOUSLIB_IT         p,
              const ANONYMOUSLIB_IT         m,
              const int                 bit_y_offset,
              const int                 bit_scansum_offset,
              const int                 num_packet,
              const ANONYMOUSLIB_IT        *row_pointer,
              const ANONYMOUSLIB_IT        *column_index,
              const ANONYMOUSLIB_VT        *value,
              const ANONYMOUSLIB_UIT       *partition_pointer,
              const ANONYMOUSLIB_UIT       *partition_descriptor,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset_pointer,
              const ANONYMOUSLIB_IT        *partition_descriptor_offset,
              ANONYMOUSLIB_VT              *calibrator,
              const ANONYMOUSLIB_IT         tail_partition_start,
              const ANONYMOUSLIB_VT         alpha,
              const ANONYMOUSLIB_VT        *x,
              ANONYMOUSLIB_VT              *y)
{
#ifdef __MIC__
    spmv_csr5_compute_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (column_index, value, row_pointer, x,
             partition_pointer, partition_descriptor,
             partition_descriptor_offset_pointer, partition_descriptor_offset,
             calibrator, y, p,
             num_packet, bit_y_offset, bit_scansum_offset, alpha, sigma);

    spmv_csr5_calibrate_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (partition_pointer, calibrator, y, p);

    spmv_csr5_tail_partition_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            (row_pointer, column_index, value, x, y,
             tail_partition_start, p, m, sigma, alpha);

#endif
}

#endif // CSR5_SPMV_PHI_H
