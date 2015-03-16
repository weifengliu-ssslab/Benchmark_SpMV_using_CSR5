#ifndef FORMAT_H
#define FORMAT_H

#include "common_cuda.h"
#include "utils_cuda.h"

int format_warmup()
{
    int *d_scan;
    checkCudaErrors(cudaMalloc((void **)&d_scan, ANONYMOUSLIB_CSR5_OMEGA * sizeof(int)));

    int num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    int num_blocks  = 4000;

    for (int i = 0; i < 50; i++)
        warmup_kernel<<< num_blocks, num_threads >>>(d_scan);

    checkCudaErrors(cudaFree(d_scan));
}

template<typename iT, typename uiT>
__global__
void generate_partition_pointer_s1_kernel(const iT     *d_row_pointer,
                                          uiT          *d_partition_pointer,
                                          const int     sigma,
                                          const iT      p,
                                          const iT      m,
                                          const iT      nnz)
{
    // global thread id
    iT global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);

    // compute partition boundaries by partition of size sigma * omega
    iT boundary = global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA;

    // clamp partition boundaries to [0, nnz]
    boundary = boundary > nnz ? nnz : boundary;

    // binary search
    if (global_id <= p)
        d_partition_pointer[global_id] = binary_search_right_boundary_kernel<iT>(d_row_pointer, boundary, m + 1) - 1;
}

template<typename iT, typename uiT>
__global__
void generate_partition_pointer_s2_kernel(const iT   *d_row_pointer,
                                          uiT        *d_partition_pointer)
{
    const iT group_id = blockIdx.x;
    const int local_id = threadIdx.x;
    const iT local_size = blockDim.x;

    volatile __shared__ int s_dirty[1];

    if (!local_id)
        s_dirty[0] = 0;
    __syncthreads();

    uiT start = d_partition_pointer[group_id];
    uiT stop  = d_partition_pointer[group_id+1];

    start = (start << 1) >> 1;
    stop  = (stop << 1) >> 1;

    if(start == stop)
        return;

    uiT num_row_in_partition = stop + 1 - start;
    int loop = ceil((float)num_row_in_partition / (float)local_size);
    iT row_idx, row_off_l, row_off_r;

    for (int i = 0; i < loop; i++)
    {
        row_idx = i * local_size + start + local_id;

        if (row_idx < stop)
        {
            row_off_l = d_row_pointer[row_idx];
            row_off_r = d_row_pointer[row_idx+1];

            if (row_off_l == row_off_r)
                s_dirty[0] = 1;
        }
        __syncthreads();

        if (s_dirty[0])
            break;
    }

    if (s_dirty[0] && !local_id)
    {
        start |= sizeof(uiT) == 4 ? 0x80000000 : 0x8000000000000000;
        d_partition_pointer[group_id] = start;
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
    int num_threads = 128;
    int num_blocks  = ceil((double)(p + 1) / (double)num_threads);

    // step 1. binary search row pointer
    generate_partition_pointer_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            <<< num_blocks, num_threads >>>(row_pointer, partition_pointer, sigma, p, m, nnz);

    // step 2. check empty rows
    num_threads = 64;
    num_blocks  = p;
    generate_partition_pointer_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            <<< num_blocks, num_threads >>>(row_pointer, partition_pointer);

//    // print for debug
//    ANONYMOUSLIB_UIT *check_partition_pointer = (ANONYMOUSLIB_UIT *)malloc((p+1)*sizeof(ANONYMOUSLIB_UIT));
//    checkCudaErrors(cudaMemcpy(check_partition_pointer, partition_pointer, (p+1) * sizeof(ANONYMOUSLIB_UIT), cudaMemcpyDeviceToHost));

//    cout << "partition_pointer = ";
//    print_1darray<ANONYMOUSLIB_UIT>(check_partition_pointer, p+1);
//    free(check_partition_pointer);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
__global__
void generate_partition_descriptor_s1_kernel(const iT    *d_row_pointer,
                                             uiT         *d_partition_descriptor,
                                             const iT     m,
                                             const int    sigma,
                                             const int    bit_all_offset,
                                             const int    num_packet)
{
    const iT global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);

    if (global_id < m)
    {
        const iT row_offset = d_row_pointer[global_id];

        const iT  gx    = row_offset / sigma;

        const iT  lx    = gx % ANONYMOUSLIB_CSR5_OMEGA;
        const iT  pid   = gx / ANONYMOUSLIB_CSR5_OMEGA;

        const int glid  = row_offset % sigma + bit_all_offset;
        const int llid  = glid % 32;
        const int ly    = glid / 32;

        const uiT val = 0x1 << (31 - llid);

        const int location = pid * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lx;

        atomicOr(&d_partition_descriptor[location], val);
    }
}

template<typename iT, typename uiT>
__global__
void generate_partition_descriptor_s2_kernel(const uiT    *d_partition_pointer,
                                             uiT          *d_partition_descriptor,
                                             iT           *d_partition_descriptor_offset_pointer,
                                             const int     sigma,
                                             const int     num_packet,
                                             const int     bit_y_offset,
                                             const int     bit_scansum_offset,
                                             const int     p)
{
    const int lane_id = threadIdx.x % ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = threadIdx.x / ANONYMOUSLIB_CSR5_OMEGA;
    const int par_id = (blockIdx.x * blockDim.x + threadIdx.x) / ANONYMOUSLIB_CSR5_OMEGA;

    volatile __shared__ uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];

    if (threadIdx.x < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)
        s_row_start_stop[threadIdx.x] = d_partition_pointer[par_id + threadIdx.x];
    __syncthreads();

    uiT row_start       = s_row_start_stop[bunch_id];
    bool with_empty_rows = (row_start >> 31) & 0x1;
    row_start          &= 0x7FFFFFFF; //( << 1) >> 1
    const iT row_stop   = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;

    // if this is fast track partition, do not generate its partition_descriptor
    if (row_start == row_stop)
        return;

    int y_offset = 0;
    int scansum_offset = 0;

    int start = 0, stop = 0, segn = 0;
    bool present = 0;
    uiT bitflag = 0;

#if __CUDA_ARCH__ < 300
    volatile __shared__ int s_segn_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];
#endif
    volatile __shared__ int s_present[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];

    const int bit_all_offset = bit_y_offset + bit_scansum_offset;

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

#if __CUDA_ARCH__ >= 300
    y_offset = scan_32_shfl<int>(segn, lane_id); // inclusive scan
    if (lane_id == ANONYMOUSLIB_CSR5_OMEGA - 1 && with_empty_rows)
    {
        d_partition_descriptor_offset_pointer[par_id] = y_offset; // the total number of segments in this partition
        d_partition_descriptor_offset_pointer[p] = 1; // the total number of segments in total
    }
    y_offset -= segn; // convert to exclusive scan
#else
    s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = segn;
    scan_32_plus1<int>(&s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)], lane_id); // exclusive scan
    if (!lane_id && with_empty_rows)
    {
        d_partition_descriptor_offset_pointer[par_id] = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + ANONYMOUSLIB_CSR5_OMEGA]; // the total number of segments in this partition
        d_partition_descriptor_offset_pointer[p] = 1; // the total number of segments in total
    }
    y_offset = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id];
#endif

    // compute scansum_offset
    s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = present;
    int next1 = lane_id + 1;
    if (present)
    {
        while (!s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + next1] && next1 < ANONYMOUSLIB_CSR5_OMEGA)
        {
            scansum_offset++;
            next1++;
        }
    }

    y_offset = lane_id ? y_offset - 1 : 0;

    first_packet |= y_offset << (32 - bit_y_offset);
    first_packet |= scansum_offset << (32 - bit_all_offset);

    d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id] = first_packet;
}

template<typename iT>
__global__
void generate_partition_descriptor_s3_kernel(iT           *d_partition_descriptor_offset_pointer,
                                             const int     p)
{
    const int local_id = threadIdx.x;
    const int local_size = blockDim.x;

    iT sum = 0;
    volatile __shared__ iT s_partition_descriptor_offset_pointer[256 + 1];

    int loop = ceil((float)p / (float)local_size);

    for (int i = 0; i < loop; i++)
    {
        s_partition_descriptor_offset_pointer[local_id] = (local_id + i * local_size < p) ? d_partition_descriptor_offset_pointer[local_id + i * local_size] : 0;
        __syncthreads();

        scan_256_plus1<iT>(s_partition_descriptor_offset_pointer);
        __syncthreads();

        s_partition_descriptor_offset_pointer[local_id] += sum;
        if (!local_id)
            s_partition_descriptor_offset_pointer[256] += sum;
        __syncthreads();

        sum = s_partition_descriptor_offset_pointer[256];

        if (local_id + i * local_size < p + 1)
            d_partition_descriptor_offset_pointer[local_id + i * local_size] = s_partition_descriptor_offset_pointer[local_id];
    }
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
    int num_threads = 128;
    int num_blocks = ceil((double)m / (double)num_threads);

    int bit_all_offset = bit_y_offset + bit_scansum_offset;

    generate_partition_descriptor_s1_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            <<< num_blocks, num_threads >>>(row_pointer, partition_descriptor,
                                            m, sigma, bit_all_offset, num_packet);

    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks  = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    generate_partition_descriptor_s2_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            <<< num_blocks, num_threads >>>(partition_pointer, partition_descriptor, partition_descriptor_offset_pointer,
                                            sigma, num_packet, bit_y_offset, bit_scansum_offset, p);

    ANONYMOUSLIB_IT num_offsets;
    checkCudaErrors(cudaMemcpy(&num_offsets, &partition_descriptor_offset_pointer[p], sizeof(ANONYMOUSLIB_IT),   cudaMemcpyDeviceToHost));

    if (num_offsets)
    {
        // prefix-sum partition_descriptor_offset_pointer
        num_threads = 256;
        num_blocks  = 1;
        generate_partition_descriptor_s3_kernel<ANONYMOUSLIB_IT>
                <<< num_blocks, num_threads >>>(partition_descriptor_offset_pointer, p);


        checkCudaErrors(cudaMemcpy(&num_offsets, &partition_descriptor_offset_pointer[p], sizeof(ANONYMOUSLIB_IT),   cudaMemcpyDeviceToHost));
    }

    *_num_offsets = num_offsets;

//    // print for debug
//    ANONYMOUSLIB_UIT *check_partition_descriptor = (ANONYMOUSLIB_UIT *)malloc(p*ANONYMOUSLIB_CSR5_OMEGA*num_packet*sizeof(ANONYMOUSLIB_UIT));
//    checkCudaErrors(cudaMemcpy(check_partition_descriptor, partition_descriptor, p*ANONYMOUSLIB_CSR5_OMEGA*num_packet * sizeof(ANONYMOUSLIB_UIT), cudaMemcpyDeviceToHost));

//    cout << "check_partition_descriptor(1) = " << endl;
//    print_tile<ANONYMOUSLIB_UIT>(check_partition_descriptor, num_packet, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "check_partition_descriptor(2) = " << endl;
//    print_tile<ANONYMOUSLIB_UIT>(&check_partition_descriptor[num_packet * ANONYMOUSLIB_CSR5_OMEGA], num_packet, ANONYMOUSLIB_CSR5_OMEGA);

//    free(check_partition_descriptor);

    return ANONYMOUSLIB_SUCCESS;
}

template<typename iT, typename uiT>
__inline__ __device__
void partition_normal_track_empty_Ologn(const iT             *d_row_pointer,
                                        const uiT            *d_partition_descriptor,
                                        const iT             *d_partition_descriptor_offset_pointer,
                                        iT                   *d_partition_descriptor_offset,
                                        const iT              par_id,
                                        const int             lane_id,
                                        const int             bit_y_offset,
                                        const int             bit_scansum_offset,
                                        iT                    row_start,
                                        const iT              row_stop,
                                        const int             c_sigma)
{
    bool local_bit;

    int offset_pointer = d_partition_descriptor_offset_pointer[par_id];

    uiT descriptor = d_partition_descriptor[lane_id];

    int y_offset = descriptor >> (32 - bit_y_offset);
    const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;

    // step 1. thread-level seg sum
    // extract the first bit-flag packet
    int ly = 0;
    descriptor = descriptor << (bit_y_offset + bit_scansum_offset);
    descriptor = lane_id ? descriptor : descriptor | 0x80000000;

    local_bit = (descriptor >> 31) & 0x1;

    if (local_bit && lane_id)
    {
        const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma;
        const iT y_index = binary_search_right_boundary_kernel<iT>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
        d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

        y_offset++;
    }

    for (int i = 1; i < c_sigma; i++)
    {
        if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag))))
        {
            ly++;
            descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
        }
        const int norm_i = 31 & (!ly ? i : i - bit_bitflag);

        local_bit = (descriptor >> (31 - norm_i)) & 0x1;

        if (local_bit)
        {
            const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma + i;
            const iT y_index = binary_search_right_boundary_kernel<iT>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
            d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;

            y_offset++;
        }
    }
}

template<typename iT, typename uiT>
__inline__ __device__
void generate_partition_descriptor_offset_partition(const iT           *d_row_pointer,
                                                    const uiT          *d_partition_pointer,
                                                    const uiT          *d_partition_descriptor,
                                                    const iT           *d_partition_descriptor_offset_pointer,
                                                    iT                 *d_partition_descriptor_offset,
                                                    const iT            par_id,
                                                    const int           lane_id,
                                                    const int           bunch_id,
                                                    const int           bit_y_offset,
                                                    const int           bit_scansum_offset,
                                                    const int           c_sigma)
{
    uiT row_start, row_stop;

#if __CUDA_ARCH__ >= 350
    if (lane_id < 2)
        row_start = __ldg(&d_partition_pointer[par_id + lane_id]);
    row_stop = __shfl(row_start, 1);
    row_start = __shfl(row_start, 0);
    row_stop &= 0x7FFFFFFF;
#else
    volatile __shared__ uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];
    if (threadIdx.x < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)
        s_row_start_stop[threadIdx.x] = d_partition_pointer[par_id + threadIdx.x];
    __syncthreads();

    row_start = s_row_start_stop[bunch_id];
    row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;
#endif

    if (row_start >> 31) // with empty rows
    {
        row_start &= 0x7FFFFFFF;     //( << 1) >> 1

        partition_normal_track_empty_Ologn<iT, uiT>
                (d_row_pointer,
                 d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, row_start, row_stop, c_sigma);
    }
    else // without empty rows
    {
        return;
    }
}

template<typename iT, typename uiT>
__global__
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
    // warp lane id
    const int lane_id = 31 & threadIdx.x; //threadIdx.x % ANONYMOUSLIB_CSR5_OMEGA;
    // warp global id == par_id
    const iT  par_id = (blockIdx.x * blockDim.x + threadIdx.x) / ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = threadIdx.x / ANONYMOUSLIB_CSR5_OMEGA;

    if (par_id >= p - 1)
        return;

    generate_partition_descriptor_offset_partition<iT, uiT>
                (d_row_pointer, d_partition_pointer,
                 &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],
                 d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, c_sigma);
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
    int num_threads = ANONYMOUSLIB_THREAD_GROUP;
    int num_blocks = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    generate_partition_descriptor_offset_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
            <<< num_blocks, num_threads >>>(row_pointer, partition_pointer,
                                            partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset,
                                            p, num_packet, bit_y_offset, bit_scansum_offset, sigma);


    return ANONYMOUSLIB_SUCCESS;
}

template<typename T, typename uiT, int CSR5_SIGMA>
__global__
void aosoa_transpose_kernel_smem(T         *d_data,
                                 const uiT *d_partition_pointer,
                                 const bool R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR
{
    __shared__ uiT s_par[2];

    const int local_id = threadIdx.x;

    if (local_id < 2)
        s_par[local_id] = d_partition_pointer[blockIdx.x + local_id];
    __syncthreads();

    // if this is fast track partition, do not transpose it
    if (s_par[0] == s_par[1])
        return;

    __shared__ T s_data[CSR5_SIGMA * (ANONYMOUSLIB_CSR5_OMEGA + 1)];

    // load global data to shared mem
    int idx_y, idx_x;
    #pragma unroll
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * CSR5_SIGMA; idx += blockDim.x)
    {
        if (R2C)
        {
            idx_y = idx % CSR5_SIGMA;
            idx_x = idx / CSR5_SIGMA;
        }
        else
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }

        s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x] = d_data[blockIdx.x * ANONYMOUSLIB_CSR5_OMEGA * CSR5_SIGMA + idx];
    }
    __syncthreads();

    // store transposed shared mem data to global
    #pragma unroll
    for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * CSR5_SIGMA; idx += blockDim.x)
    {
        if (R2C)
        {
            idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;
            idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;
        }
        else
        {
            idx_y = idx % CSR5_SIGMA;
            idx_x = idx / CSR5_SIGMA;
        }

        d_data[blockIdx.x * ANONYMOUSLIB_CSR5_OMEGA * CSR5_SIGMA + idx] = s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x];

        //if (blockIdx.x == 0 && sizeof(T) == 8)
        //    printf("Round %d\t thread %d, szT = %d, val = %f\n", idx, threadIdx.x, sizeof(T), s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x]);
    }
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int aosoa_transpose(const int           sigma,
                    const int           nnz,
                    const ANONYMOUSLIB_UIT *partition_pointer,
                    ANONYMOUSLIB_IT        *column_index,
                    ANONYMOUSLIB_VT        *value,
                    bool                R2C)
{
    int num_threads = 128;
    int num_blocks = ceil((double)nnz / (double)(ANONYMOUSLIB_CSR5_OMEGA * sigma)) - 1;

    switch (sigma)
    {
    case 4:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 4><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 4><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 5:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 5><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 5><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 6:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 6><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 6><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 7:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 7><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 7><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 8:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 8><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 8><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 9:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 9><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 9><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 10:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 10><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 10><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;

    case 11:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 11><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 11><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 12:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 12><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 12><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 13:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 13><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 13><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 14:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 14><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 14><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 15:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 15><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 15><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 16:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 16><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 16><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 17:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 17><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 17><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 18:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 18><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 18><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 19:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 19><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 19><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 20:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 20><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 20><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;

    case 21:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 21><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 21><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 22:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 22><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 22><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 23:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 23><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 23><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 24:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 24><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 24><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 25:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 25><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 25><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 26:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 26><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 26><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 27:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 27><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 27><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 28:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 28><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 28><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 29:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 29><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 29><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 30:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 30><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 30><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;

    case 31:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 31><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 31><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    case 32:
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, 32><<< num_blocks, num_threads >>>(column_index, partition_pointer, R2C);
        aosoa_transpose_kernel_smem<ANONYMOUSLIB_VT, ANONYMOUSLIB_UIT, 32><<< num_blocks, num_threads >>>(value, partition_pointer, R2C);
        break;
    }

//    // print for debug
//    ANONYMOUSLIB_IT *check_column_index = (ANONYMOUSLIB_IT *)malloc(nnz*sizeof(ANONYMOUSLIB_IT));
//    checkCudaErrors(cudaMemcpy(check_column_index, column_index, nnz * sizeof(ANONYMOUSLIB_IT), cudaMemcpyDeviceToHost));

//    cout << "check_column_index(1) = " << endl;
//    print_tile<ANONYMOUSLIB_IT>(check_column_index, sigma, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "check_column_index(2) = " << endl;
//    print_tile<ANONYMOUSLIB_IT>(&check_column_index[sigma * ANONYMOUSLIB_CSR5_OMEGA], sigma, ANONYMOUSLIB_CSR5_OMEGA);

//    free(check_column_index);

//    // print for debug
//    ANONYMOUSLIB_VT *check_value = (ANONYMOUSLIB_VT *)malloc(nnz*sizeof(ANONYMOUSLIB_VT));
//    checkCudaErrors(cudaMemcpy(check_value, value, nnz * sizeof(ANONYMOUSLIB_VT), cudaMemcpyDeviceToHost));

//    cout << "check_value(1) = " << endl;
//    print_tile<ANONYMOUSLIB_VT>(check_value, sigma, ANONYMOUSLIB_CSR5_OMEGA);
//    cout << "check_value(2) = " << endl;
//    print_tile<ANONYMOUSLIB_VT>(&check_value[sigma * ANONYMOUSLIB_CSR5_OMEGA], sigma, ANONYMOUSLIB_CSR5_OMEGA);

//    free(check_value);

    return ANONYMOUSLIB_SUCCESS;
}

#endif // FORMAT_H
