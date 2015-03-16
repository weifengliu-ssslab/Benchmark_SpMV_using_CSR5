#ifndef FORMAT_AVX2_H
#define FORMAT_AVX2_H

#include "common_avx2.h"
#include "utils_avx2.h"

template<typename iT, typename uiT>
void generate_partition_pointer_s1_kernel(const iT     *d_row_pointer,
                                          uiT          *d_partition_pointer,
                                          const int     sigma,
                                          const iT      p,
                                          const iT      m,
                                          const iT      nnz)
{
    #pragma omp parallel for
    for (iT global_id = 0; global_id <= p; global_id++)
    {
        // compute partition boundaries by partition of size sigma * omega
        iT boundary = global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA;

        // clamp partition boundaries to [0, nnz]
        boundary = boundary > nnz ? nnz : boundary;

        // binary search
        d_partition_pointer[global_id] = binary_search_right_boundary_kernel<iT>(d_row_pointer, boundary, m + 1) - 1;
    }
}

template<typename iT, typename uiT>
void generate_partition_pointer_s2_kernel(const iT   *d_row_pointer,
                                          uiT        *d_partition_pointer,
                                          const iT    p)
{
    #pragma omp parallel for
    for (iT group_id = 0; group_id < p; group_id++)
    {
        int dirty = 0;

        uiT start = d_partition_pointer[group_id];
        uiT stop  = d_partition_pointer[group_id+1];

        start = (start << 1) >> 1;
        stop  = (stop << 1) >> 1;

        if(start == stop)
            continue;

        for (iT row_idx = start; row_idx <= stop; row_idx++)
        {
            if (d_row_pointer[row_idx] == d_row_pointer[row_idx+1])
            {
                dirty = 1;
                break;
            }
        }

        if (dirty)
        {
            start |= sizeof(uiT) == 4 ? 0x80000000 : 0x8000000000000000;
            d_partition_pointer[group_id] = start;
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_pointer(const int           sigma,
                               const ANONYMOUSLIB_IT   p,
                               const ANONYMOUSLIB_IT   m,
                               const ANONYMOUSLIB_IT   nnz,
                               ANONYMOUSLIB_UIT       *partition_pointer,
                               const ANONYMOUSLIB_IT  *row_pointer)
{
    // step 1. binary search row pointer
    generate_partition_pointer_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (row_pointer, partition_pointer, sigma, p, m, nnz);

    // step 2. check empty rows
    generate_partition_pointer_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (row_pointer, partition_pointer, p);

//    // print for debug
//    cout << "partition_pointer = ";
//    print_1darray<ANONYMOUSLIB_UIT>(partition_pointer, p+1);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
void generate_partition_descriptor_s1_kernel(const iT    *d_row_pointer,
                                             const uiT   *d_partition_pointer,
                                             uiT         *d_partition_descriptor,
                                             const iT     m,
                                             const iT     p,
                                             const int    sigma,
                                             const int    bit_all_offset,
                                             const int    num_packet)
{
    #pragma omp parallel for
    for (int par_id = 0; par_id < p-1; par_id++)
    {
        const iT row_start = d_partition_pointer[par_id]     & 0x7FFFFFFF;
        const iT row_stop  = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        for (int rid = row_start; rid <= row_stop; rid++)
        {
            int ptr = d_row_pointer[rid];
            int pid = ptr / (ANONYMOUSLIB_CSR5_OMEGA * sigma);

            if (pid == par_id)
            {
                int lx = (ptr / sigma) % ANONYMOUSLIB_CSR5_OMEGA;

                const int glid = ptr % sigma + bit_all_offset;
                const int ly = glid / 32;
                const int llid = glid % 32;

                const uiT val = 0x1 << (31 - llid);

                const int location = pid * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lx;
                d_partition_descriptor[location] |= val;
            }
        }
    }
}

template<typename iT, typename uiT>
void generate_partition_descriptor_s2_kernel(const uiT    *d_partition_pointer,
                                             uiT          *d_partition_descriptor,
                                             iT           *d_partition_descriptor_offset_pointer,
                                             const int     sigma,
                                             const int     num_packet,
                                             const int     bit_y_offset,
                                             const int     bit_scansum_offset,
                                             const iT      p)
{
    int num_thread = omp_get_max_threads();
    int *s_segn_scan_all = (int *)_mm_malloc(2 * ANONYMOUSLIB_CSR5_OMEGA * sizeof(int) * num_thread, ANONYMOUSLIB_X86_CACHELINE);
    int *s_present_all   = (int *)_mm_malloc(2 * ANONYMOUSLIB_CSR5_OMEGA * sizeof(int) * num_thread, ANONYMOUSLIB_X86_CACHELINE);
    for (int i = 0; i < num_thread; i++)
        s_present_all[i * 2 * ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA] = 1;

    const int bit_all_offset = bit_y_offset + bit_scansum_offset;

    #pragma omp parallel for
    for (int par_id = 0; par_id < p-1; par_id++)
    {
        int tid = omp_get_thread_num();
        int *s_segn_scan = &s_segn_scan_all[tid * 2 * ANONYMOUSLIB_CSR5_OMEGA];
        int *s_present = &s_present_all[tid * 2 * ANONYMOUSLIB_CSR5_OMEGA];

        memset(s_segn_scan, 0, (ANONYMOUSLIB_CSR5_OMEGA + 1) * sizeof(int));
        memset(s_present, 0, ANONYMOUSLIB_CSR5_OMEGA * sizeof(int));

        bool with_empty_rows = (d_partition_pointer[par_id] >> 31) & 0x1;
        iT row_start       = d_partition_pointer[par_id]     & 0x7FFFFFFF;
        const iT row_stop  = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        if (row_start == row_stop)
            continue;

        #pragma simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++)
        {
            int start = 0, stop = 0, segn = 0;
            bool present = 0;
            uiT bitflag = 0;

            present |= !lane_id;

            // extract the first bit-flag packet
            int ly = 0;
            uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];
            bitflag = (first_packet << bit_all_offset) | ((uiT)present << 31);
            start = !((bitflag >> 31) & 0x1);
            present |= (bitflag >> 31) & 0x1;

            for (int i = 1; i < sigma; i++)
            {
                if ((!ly && i == 32 - bit_all_offset) || (ly && (i - (32 - bit_all_offset)) % 32 == 0))
                {
                    ly++;
                    bitflag = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
                }
                const int norm_i = !ly ? i : i - (32 - bit_all_offset);
                stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;
                present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
            }

            // compute y_offset for all partitions
            segn = stop - start + present;
            segn = segn > 0 ? segn : 0;

            s_segn_scan[lane_id] = segn;

            // compute scansum_offset
            s_present[lane_id] = present;
        }

        scan_single<int>(s_segn_scan, ANONYMOUSLIB_CSR5_OMEGA + 1);

        if (with_empty_rows)
        {
            d_partition_descriptor_offset_pointer[par_id] = s_segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
            d_partition_descriptor_offset_pointer[p] += s_segn_scan[ANONYMOUSLIB_CSR5_OMEGA];
        }

        #pragma simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++)
        {
            int y_offset = s_segn_scan[lane_id];

            int scansum_offset = 0;
            int next1 = lane_id + 1;
            if (s_present[lane_id])
            {
                while (!s_present[next1] && next1 < ANONYMOUSLIB_CSR5_OMEGA)
                {
                    scansum_offset++;
                    next1++;
                }
            }

            uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];

            y_offset = lane_id ? y_offset - 1 : 0;

            first_packet |= y_offset << (32 - bit_y_offset);
            first_packet |= scansum_offset << (32 - bit_all_offset);

            d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id] = first_packet;
        }
    }

    _mm_free(s_segn_scan_all);
    _mm_free(s_present_all);
}


template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor(const int           sigma,
                                  const ANONYMOUSLIB_IT   p,
                                  const ANONYMOUSLIB_IT   m,
                                  const int           bit_y_offset,
                                  const int           bit_scansum_offset,
                                  const int           num_packet,
                                  const ANONYMOUSLIB_IT  *row_pointer,
                                  const ANONYMOUSLIB_UIT *partition_pointer,
                                  ANONYMOUSLIB_UIT       *partition_descriptor,
                                  ANONYMOUSLIB_IT        *partition_descriptor_offset_pointer,
                                  ANONYMOUSLIB_IT        *_num_offsets)
{
    int bit_all_offset = bit_y_offset + bit_scansum_offset;

    generate_partition_descriptor_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (row_pointer, partition_pointer, partition_descriptor, m, p, sigma, bit_all_offset, num_packet);

    generate_partition_descriptor_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (partition_pointer, partition_descriptor, partition_descriptor_offset_pointer,
             sigma, num_packet, bit_y_offset, bit_scansum_offset, p);

    if (partition_descriptor_offset_pointer[p])
        scan_single<ANONYMOUSLIB_IT>(partition_descriptor_offset_pointer, p+1);

    *_num_offsets = partition_descriptor_offset_pointer[p];

    // print for debug
//    cout << "partition_descriptor(1) = " << endl;
//    print_tile<ANONYMOUSLIB_UIT>(partition_descriptor, num_packet, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "partition_descriptor(2) = " << endl;
//    print_tile<ANONYMOUSLIB_UIT>(&partition_descriptor[num_packet * ANONYMOUSLIB_CSR5_OMEGA], num_packet, ANONYMOUSLIB_CSR5_OMEGA);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
void generate_partition_descriptor_offset_kernel(const iT           *d_row_pointer,
                                                 const uiT          *d_partition_pointer,
                                                 const uiT          *d_partition_descriptor,
                                                 const iT           *d_partition_descriptor_offset_pointer,
                                                 iT                 *d_partition_descriptor_offset,
                                                 const iT            p,
                                                 const int           num_packet,
                                                 const int           bit_y_offset,
                                                 const int           bit_scansum_offset,
                                                 const int           c_sigma)
{
    const int bit_all_offset = bit_y_offset + bit_scansum_offset;
    const int bit_bitflag = 32 - bit_all_offset;

    #pragma omp parallel for
    for (int par_id = 0; par_id < p-1; par_id++)
    {
        bool with_empty_rows = (d_partition_pointer[par_id] >> 31) & 0x1;
        if (!with_empty_rows)
            continue;

        iT row_start       = d_partition_pointer[par_id]     & 0x7FFFFFFF;
        const iT row_stop  = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

        int offset_pointer = d_partition_descriptor_offset_pointer[par_id];
        #pragma simd
        for (int lane_id = 0; lane_id < ANONYMOUSLIB_CSR5_OMEGA; lane_id++)
        {
            bool local_bit;

            // extract the first bit-flag packet
            int ly = 0;
            uiT descriptor = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];
            int y_offset = descriptor >> (32 - bit_y_offset);

            descriptor = descriptor << bit_all_offset;
            descriptor = lane_id ? descriptor : descriptor | 0x80000000;

            local_bit = (descriptor >> 31) & 0x1;

            if (local_bit && lane_id)
            {
                const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma;
                const iT y_index = binary_search_right_boundary_kernel<iT>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
                //printf("threadid = %i, i = %i, y_idx = %i, y_offset = %i\n", lane_id, 0, y_index, y_offset);
                d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

                y_offset++;
            }

            for (int i = 1; i < c_sigma; i++)
            {
                if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag))))
                {
                    ly++;
                    descriptor = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
                }
                const int norm_i = 31 & (!ly ? i : i - bit_bitflag);

                local_bit = (descriptor >> (31 - norm_i)) & 0x1;

                if (local_bit)
                {
                    const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma + i;
                    const iT y_index = binary_search_right_boundary_kernel<iT>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
                    //printf("threadid = %i, i = %i, y_idx = %i, y_offset = %i\n", lane_id, i, y_index, y_offset);
                    d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

                    y_offset++;
                }
            }
        }
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor_offset(const int           sigma,
                                         const ANONYMOUSLIB_IT   p,
                                         const int           bit_y_offset,
                                         const int           bit_scansum_offset,
                                         const int           num_packet,
                                         const ANONYMOUSLIB_IT  *row_pointer,
                                         const ANONYMOUSLIB_UIT *partition_pointer,
                                         ANONYMOUSLIB_UIT       *partition_descriptor,
                                         ANONYMOUSLIB_IT        *partition_descriptor_offset_pointer,
                                         ANONYMOUSLIB_IT        *partition_descriptor_offset)
{
    generate_partition_descriptor_offset_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            (row_pointer, partition_pointer,
             partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset,
             p, num_packet, bit_y_offset, bit_scansum_offset, sigma);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename T, typename uiT>
void aosoa_transpose_kernel_smem(T         *d_data,
                                 const uiT *d_partition_pointer,
                                 const int  nnz,
                                 const int  sigma,
                                 const bool R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR
{
    int num_p = ceil((double)nnz / (double)(ANONYMOUSLIB_CSR5_OMEGA * sigma)) - 1;

    int num_thread = omp_get_max_threads();
    T *s_data_all = (T *)_mm_malloc(sigma * ANONYMOUSLIB_CSR5_OMEGA * sizeof(T) * num_thread, ANONYMOUSLIB_X86_CACHELINE);

    #pragma omp parallel for
    for (int par_id = 0; par_id < num_p; par_id++)
    {
        int tid = omp_get_thread_num();
        T *s_data = &s_data_all[sigma * ANONYMOUSLIB_CSR5_OMEGA * tid];

        // if this is fast track partition, do not transpose it
        if (d_partition_pointer[par_id] == d_partition_pointer[par_id + 1])
            continue;

        // load global data to shared mem
        int idx_y, idx_x;
        #pragma simd
        for (int idx = 0; idx < ANONYMOUSLIB_CSR5_OMEGA * sigma; idx++)
        {
            if (R2C)
            {
                idx_y = idx % sigma;
                idx_x = idx / sigma;
            }
            else
            {
                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
            }

            s_data[idx_y * ANONYMOUSLIB_CSR5_OMEGA + idx_x] = d_data[par_id * ANONYMOUSLIB_CSR5_OMEGA * sigma + idx];
        }

        // store transposed shared mem data to global
        #pragma simd
        for (int idx = 0; idx < ANONYMOUSLIB_CSR5_OMEGA * sigma; idx++)
        {
            if (R2C)
            {
                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
            }
            else
            {
                idx_y = idx % sigma;
                idx_x = idx / sigma;
            }

            d_data[par_id * ANONYMOUSLIB_CSR5_OMEGA * sigma + idx] = s_data[idx_y * ANONYMOUSLIB_CSR5_OMEGA + idx_x];
        }
    }

    _mm_free(s_data_all);
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int aosoa_transpose(const int           sigma,
                    const int           nnz,
                    const ANONYMOUSLIB_UIT *partition_pointer,
                    ANONYMOUSLIB_IT        *column_index,
                    ANONYMOUSLIB_VT        *value,
                    bool                R2C)
{
    aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(column_index, partition_pointer, nnz, sigma, R2C);
    aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT>(value, partition_pointer, nnz, sigma, R2C);

//    // print for debug
//    cout << "column_index(1) = " << endl;
//    print_tile<ANONYMOUSLIB_IT>(column_index, sigma, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "column_index(2) = " << endl;
//    print_tile<ANONYMOUSLIB_IT>(&column_index[sigma * ANONYMOUSLIB_CSR5_OMEGA], sigma, ANONYMOUSLIB_CSR5_OMEGA);

//    // print for debug
//    cout << "value(1) = " << endl;
//    print_tile<ANONYMOUSLIB_VT>(value, sigma, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "value(2) = " << endl;
//    print_tile<ANONYMOUSLIB_VT>(&value[sigma * ANONYMOUSLIB_CSR5_OMEGA], sigma, ANONYMOUSLIB_CSR5_OMEGA);

    return ANONYMOUSLIB_SUCCESS;
}

#endif // FORMAT_AVX2_H
