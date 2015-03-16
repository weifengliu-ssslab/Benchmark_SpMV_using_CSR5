#ifndef CSR5_SPMV_H
#define CSR5_SPMV_H

#include "common_cuda.h"
#include "utils_cuda.h"

template<typename iT, typename vT>
__inline__ __device__
vT candidate(const vT           *d_value_partition,
             const vT           *d_x,
             cudaTextureObject_t d_x_tex,
             const iT           *d_column_index_partition,
             const iT            candidate_index,
             const vT            alpha)
{
    vT x = 0;
#if __CUDA_ARCH__ >= 350
    x = __ldg(&d_x[d_column_index_partition[candidate_index]]);
#else
    fetch_x<iT>(d_x_tex, d_column_index_partition[candidate_index], &x);
#endif
    return d_value_partition[candidate_index] * x;// * alpha;
}

template<typename vT>
__forceinline__ __device__
vT segmented_sum_shfl(vT        tmp_sum,
                      const int scansum_offset,
                      const int lane_id)
{
    vT sum = __shfl_down(tmp_sum, 1);
    sum = lane_id == ANONYMOUSLIB_CSR5_OMEGA - 1 ? 0 : sum;
    vT scan_sum = scan_32_shfl(sum); //scan_32_shfl<vT>(sum, lane_id); // inclusive scan
    tmp_sum = __shfl_down(scan_sum, scansum_offset);
    tmp_sum = tmp_sum - scan_sum + sum;

    return tmp_sum;
}

template<typename vT>
__forceinline__ __device__
vT segmented_sum(vT             tmp_sum,
                 volatile vT   *s_sum,
                 const int      scansum_offset,
                 const int      lane_id)
{
    if (lane_id)
        s_sum[lane_id - 1] = tmp_sum;
    s_sum[lane_id] = lane_id == ANONYMOUSLIB_CSR5_OMEGA - 1 ? 0 : s_sum[lane_id];
    vT sum = tmp_sum = s_sum[lane_id];
    scan_32<vT>(s_sum, lane_id); // exclusive scan
    s_sum[lane_id] += tmp_sum; // inclusive scan = exclusive scan + original value
    tmp_sum = s_sum[lane_id + scansum_offset];
    tmp_sum = tmp_sum - s_sum[lane_id] + sum;

    return tmp_sum;
}

template<typename iT, typename vT, int c_sigma>
__inline__ __device__
void partition_fast_track(const vT           *d_value_partition,
                          const vT           *d_x,
                          cudaTextureObject_t d_x_tex,
                          const iT           *d_column_index_partition,
                          vT                 *d_calibrator,
#if __CUDA_ARCH__ < 300
                          volatile vT        *s_sum,
#endif
                          const int           lane_id,
                          const iT            par_id,
                          const vT            alpha)
{
    vT sum = 0;

    #pragma unroll
    for (int i = 0; i < c_sigma; i++)
        sum += candidate<iT, vT>(d_value_partition, d_x, d_x_tex, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha);

#if __CUDA_ARCH__ >= 300 // use shfl intrinsic
    sum = sum_32_shfl<vT>(sum);
    if (!lane_id)
        d_calibrator[par_id] = sum;
#else // use smem
    s_sum[lane_id] = sum;
    sum_32<vT>(s_sum, lane_id);
    if (!lane_id)
        d_calibrator[par_id] = s_sum[0];
#endif
}

template<typename iT, typename uiT, typename vT, int c_sigma>
__inline__ __device__
void partition_normal_track(const iT           *d_column_index_partition,
                            const vT           *d_value_partition,
                            const vT           *d_x,
                            cudaTextureObject_t d_x_tex,
                            const uiT          *d_partition_descriptor,
                            const iT           *d_partition_descriptor_offset_pointer,
                            const iT           *d_partition_descriptor_offset,
                            vT                 *d_calibrator,
                            vT                 *d_y,
#if __CUDA_ARCH__ < 300
                            volatile vT        *s_sum,
                            volatile int       *s_scan,
#endif
                            const iT            par_id,
                            const int           lane_id,
                            const int           bit_y_offset,
                            const int           bit_scansum_offset,
                            iT                  row_start,
                            const bool          empty_rows,
                            const vT            alpha)
{
    int start = 0;
    int stop = 0;

    bool local_bit;
    vT sum = 0;

    int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

    uiT descriptor = d_partition_descriptor[lane_id];

    int y_offset = descriptor >> (32 - bit_y_offset);
    const int scansum_offset = (descriptor << bit_y_offset) >> (32 - bit_scansum_offset);
    const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;

    bool direct = false;

    vT first_sum, last_sum;

    // step 1. thread-level seg sum

    int ly = 0;

    // extract the first bit-flag packet
    descriptor = descriptor << (bit_y_offset + bit_scansum_offset);
    descriptor = lane_id ? descriptor : descriptor | 0x80000000;

    local_bit = (descriptor >> 31) & 0x1;
    start = !local_bit;
    direct = local_bit & (bool)lane_id;

    sum = candidate<iT, vT>(d_value_partition, d_x, d_x_tex, d_column_index_partition, lane_id, alpha);

    #pragma unroll
    for (int i = 1; i < c_sigma; i++)
    {
        int norm_i = i - bit_bitflag;

        if (!(ly || norm_i) || (ly && !(31 & norm_i)))
        {
            ly++;
            descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];
        }
        norm_i = !ly ? 31 & i : 31 & norm_i;
        norm_i = 31 - norm_i;

        local_bit = (descriptor >> norm_i) & 0x1;

        if (local_bit)
        {
            if (direct)
                d_y[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = sum;
            else
                first_sum = sum;
        }

        y_offset += local_bit & direct;

        direct |= local_bit;
        sum = local_bit ? 0 : sum;
        stop += local_bit;

        sum += candidate<iT, vT>(d_value_partition, d_x, d_x_tex, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha);
    }

    first_sum = direct ? first_sum : sum;
    last_sum = sum;

    // step 2. segmented sum
    sum = start ? first_sum : 0;

#if __CUDA_ARCH__ >= 300
    sum = segmented_sum_shfl<vT>(sum, scansum_offset, lane_id);
#else
    sum = segmented_sum<vT>(sum, s_sum, scansum_offset, lane_id);
#endif

    // step 3-1. add s_sum to position stop
    last_sum += (start <= stop) ? sum : 0;

    // step 3-2. write sums to result array
    if (direct)
        d_y[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = last_sum;

    // the first/last value of the first thread goes to calibration
    if (!lane_id)
        d_calibrator[par_id] = direct ? first_sum : last_sum;
}

template<typename iT, typename uiT, typename vT, int c_sigma>
__inline__ __device__
void spmv_partition(const iT           *d_column_index_partition,
                    const vT           *d_value_partition,
                    const iT           *d_row_pointer,
                    const vT           *d_x,
                    cudaTextureObject_t d_x_tex,
                    const uiT          *d_partition_pointer,
                    const uiT          *d_partition_descriptor,
                    const iT           *d_partition_descriptor_offset_pointer,
                    const iT           *d_partition_descriptor_offset,
                    vT                 *d_calibrator,
                    vT                 *d_y,
                    const iT            par_id,
                    const int           lane_id,
                    const int           bunch_id,
                    const int           bit_y_offset,
                    const int           bit_scansum_offset,
                    const vT            alpha)
{
#if __CUDA_ARCH__ < 300
    volatile __shared__ vT  s_sum[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];
    volatile __shared__ int s_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA)];
#endif

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

    if (row_start == row_stop) // fast track through reduction
    {
        partition_fast_track<iT, vT, c_sigma>
                (d_value_partition, d_x, d_x_tex, d_column_index_partition,
                 d_calibrator,
#if __CUDA_ARCH__ < 300
                 &s_sum[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
#endif
                 lane_id, par_id, alpha);
    }
    else
    {
        const bool empty_rows = (row_start >> 31) & 0x1;
        row_start &= 0x7FFFFFFF;

        d_y = &d_y[row_start+1];

        partition_normal_track<iT, uiT, vT, c_sigma>
                (d_column_index_partition, d_value_partition, d_x, d_x_tex,
                 d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 d_calibrator, d_y,
    #if __CUDA_ARCH__ < 300
                 &s_sum[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],
                 &s_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)],
    #endif
                 par_id, lane_id,
                 bit_y_offset, bit_scansum_offset, row_start, empty_rows, alpha);
    }
}

template<typename iT, typename uiT, typename vT, int c_sigma>
__global__
void spmv_csr5_compute_kernel(const iT           *d_column_index,
                              const vT           *d_value,
                              const iT           *d_row_pointer,
                              const vT           *d_x,
                              cudaTextureObject_t d_x_tex,
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
                              const vT            alpha)
{
    // warp lane id
    const int lane_id = 31 & threadIdx.x; //threadIdx.x % ANONYMOUSLIB_CSR5_OMEGA;
    // warp global id == par_id
    const iT  par_id = (blockIdx.x * blockDim.x + threadIdx.x) / ANONYMOUSLIB_CSR5_OMEGA;
    const int bunch_id = threadIdx.x / ANONYMOUSLIB_CSR5_OMEGA;

    if (par_id >= p - 1)
        return;

    spmv_partition<iT, uiT, vT, c_sigma>
                (&d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma],
                 &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma],
                 d_row_pointer, d_x, d_x_tex, d_partition_pointer,
                 &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],
                 d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,
                 d_calibrator, d_y,
                 par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, alpha);
}

template<typename iT, typename uiT, typename vT>
__global__
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                const vT  *d_calibrator,
                                vT        *d_y,
                                const iT   p)
{
    const int lane_id  = threadIdx.x % ANONYMOUSLIB_THREAD_BUNCH;
    const int bunch_id = threadIdx.x / ANONYMOUSLIB_THREAD_BUNCH;
    const int local_id = threadIdx.x;
    const iT global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);

    vT sum;

    volatile __shared__ iT s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP+1];
    volatile __shared__ vT  s_calibrator[ANONYMOUSLIB_THREAD_GROUP];
    volatile __shared__ vT  s_sum[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_THREAD_BUNCH];

    s_partition_pointer[local_id] = global_id < p-1 ? d_partition_pointer[global_id] & 0x7FFFFFFF : -1;
    s_calibrator[local_id] = sum = global_id < p-1 ? d_calibrator[global_id] : 0;
    __syncthreads();

    // do a fast track if all s_partition_pointer are the same
    if (s_partition_pointer[0] == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])
    {
        sum = sum_32_shfl<vT>(sum);
        if (!lane_id)
            s_sum[bunch_id] = sum;
        __syncthreads();

        if (!bunch_id)
        {
            sum = lane_id < (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_THREAD_BUNCH) ? s_sum[lane_id] : 0;
            sum = sum_32_shfl<vT>(sum);
        }

        if (!local_id)
            atomicAdd(&d_y[s_partition_pointer[0]], sum);

        return;
    }

    int local_par_id = local_id;
    iT row_start_current, row_start_target, row_start_previous;
    sum = 0;

    // use (p - 1), due to the tail partition is dealt with CSR-vector method
    if (global_id < p - 1)
    {
        row_start_previous = local_id ? s_partition_pointer[local_id-1] : -1;
        row_start_current = s_partition_pointer[local_id];

        if (row_start_previous != row_start_current)
        {
            row_start_target = row_start_current;

            while (row_start_target == row_start_current && local_par_id < blockDim.x)
            {
                sum +=  s_calibrator[local_par_id];
                local_par_id++;
                row_start_current = s_partition_pointer[local_par_id];
            }

            if (row_start_target == s_partition_pointer[0] || row_start_target == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])
                atomicAdd(&d_y[row_start_target], sum);
            else
                d_y[row_start_target] += sum;
        }
    }
}

template<typename iT, typename uiT, typename vT>
__global__
void spmv_csr5_tail_partition_kernel(const iT           *d_row_pointer,
                                     const iT           *d_column_index,
                                     const vT           *d_value,
                                     const vT           *d_x,
                                     cudaTextureObject_t d_x_tex,
                                     vT                 *d_y,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const int           sigma,
                                     const vT            alpha)
{
    const int local_id = threadIdx.x;

    const iT row_id    = tail_partition_start + blockIdx.x;
    const iT row_start = !blockIdx.x ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];
    const iT row_stop  = d_row_pointer[row_id + 1];

    vT sum = 0;

    for (iT idx = local_id + row_start; idx < row_stop; idx += ANONYMOUSLIB_CSR5_OMEGA)
        sum += candidate<iT, vT>(d_value, d_x, d_x_tex, d_column_index, idx, alpha);

#if __CUDA_ARCH__ >= 300 // use shfl intrinsic
    sum = sum_32_shfl<vT>(sum);
#else
    volatile __shared__ vT s_sum[ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA / 2];
    s_sum[local_id] = sum;
    sum_32<vT>(s_sum, local_id);
    sum = s_sum[local_id];
#endif

    if (!local_id)
        d_y[row_id] = !blockIdx.x ? d_y[row_id] + sum : sum;
}


template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int csr5_spmv(const int                 sigma,
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
              cudaTextureObject_t       x_tex,
              ANONYMOUSLIB_VT              *y)
{
    int err = ANONYMOUSLIB_SUCCESS;

    int num_threads = ANONYMOUSLIB_THREAD_GROUP;
    int num_blocks = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    switch (sigma)
    {
    case 4:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 4><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 5:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 5><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 6:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 6><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 7:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 7><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 8:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 8><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 9:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 9><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 10:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 10><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;

    case 11:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 11><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 12:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 12><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 13:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 13><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 14:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 14><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 15:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 15><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 16:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 16><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 17:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 17><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 18:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 18><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 19:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 19><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 20:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 20><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;

    case 21:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 21><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 22:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 22><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 23:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 23><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 24:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 24><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 25:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 25><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 26:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 26><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 27:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 27><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 28:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 28><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 29:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 29><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 30:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 30><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;

    case 31:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 31><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    case 32:
        spmv_csr5_compute_kernel<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT, 32><<< num_blocks, num_threads >>>(column_index, value, row_pointer, x, x_tex, partition_pointer, partition_descriptor, partition_descriptor_offset_pointer, partition_descriptor_offset, calibrator, y, p, num_packet, bit_y_offset, bit_scansum_offset, alpha);
        break;
    }

    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks = ceil((double)(p-1)/(double)num_threads);

    spmv_csr5_calibrate_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            <<< num_blocks, num_threads >>>
            (partition_pointer, calibrator, y, p);

    num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    num_blocks = m - tail_partition_start;

    spmv_csr5_tail_partition_kernel
            <ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
            <<< num_blocks, num_threads >>>
            (row_pointer, column_index, value, x, x_tex, y,
             tail_partition_start, p, sigma, alpha);

    return err;
}

#endif // CSR5_SPMV_H
