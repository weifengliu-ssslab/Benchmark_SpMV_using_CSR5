#ifndef UTILS_CUDA_H
#define UTILS_CUDA_H

#include "common_cuda.h"

struct anonymouslib_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

template<typename iT>
__inline__ __device__
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

#if __CUDA_ARCH__ >= 500
        key_median = __ldg(&d_row_pointer[median]);
#else
        key_median = d_row_pointer[median];
#endif

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}


template<typename T>
__inline__ __device__
void sum_32(volatile  T *s_sum,
            const int    local_id)
{
    s_sum[local_id] += s_sum[local_id + 16];
    s_sum[local_id] += s_sum[local_id + 8];
    s_sum[local_id] += s_sum[local_id + 4];
    s_sum[local_id] += s_sum[local_id + 2];
    s_sum[local_id] += s_sum[local_id + 1];
}

#if __CUDA_ARCH__ <= 300

__device__ __forceinline__
double __shfl_down(double var, unsigned int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ __forceinline__
double __shfl_up(double var, unsigned int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_up(a.x, srcLane, width);
    a.y = __shfl_up(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ __forceinline__
double __shfl_xor(double var, int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_xor(a.x, srcLane, width);
    a.y = __shfl_xor(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

#endif

template<typename vT>
__forceinline__ __device__
vT sum_32_shfl(vT sum)
{
//    #pragma unroll
//    for (int offset = ANONYMOUSLIB_CSR5_OMEGA / 2; offset > 0; offset >>= 1)
//        sum += __shfl_down(sum, offset);

    #pragma unroll
    for(int mask = ANONYMOUSLIB_CSR5_OMEGA / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor(sum, mask);

    return sum;
}

// exclusive scan using a single thread
template<typename T>
__inline__ __device__
void scan_single( volatile  T *s_scan,
              const int      local_id,
              const int      l)
{
    T old_val, new_val;
    if (!local_id)
    {
        old_val = s_scan[0];
        s_scan[0] = 0;
        for (int i = 1; i < l; i++)
        {
            new_val = s_scan[i];
            s_scan[i] = old_val + s_scan[i-1];
            old_val = new_val;
        }
    }
}

// exclusive scan
template<typename T>
__inline__ __device__
void scan_32(volatile  T *s_scan,
             const int    local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

// inclusive scan
template<typename T>
__forceinline__ __device__
T scan_32_shfl(T         x,
               const int local_id)
{
//    #pragma unroll
//    for( int offset = 1 ; offset < ANONYMOUSLIB_CSR5_OMEGA ; offset <<= 1 )
//    {
//        T y = __shfl_up(x, offset);
//        x = local_id >= offset ? x + y : x;
//    }

    T y = __shfl_up(x, 1);
    x = local_id >= 1 ? x + y : x;
    y = __shfl_up(x, 2);
    x = local_id >= 2 ? x + y : x;
    y = __shfl_up(x, 4);
    x = local_id >= 4 ? x + y : x;
    y = __shfl_up(x, 8);
    x = local_id >= 8 ? x + y : x;
    y = __shfl_up(x, 16);
    x = local_id >= 16 ? x + y : x;

    return x;
}

enum
{
    /// The number of warp scan steps
    STEPS = 5,

    // The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    SHFL_C = ((-1 << STEPS) & 31) << 8
};

// inclusive scan for double data type
__forceinline__ __device__
double scan_32_shfl(double    x)
{
    #pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .s32 lo;"
            "  .reg .s32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.f64 %0, %0, %1;"
            "}"
            : "=d"(x) : "d"(x), "r"(1 << STEP), "r"(SHFL_C));
    }

    return x;
}

// exclusive scan
template<typename T>
__inline__ __device__
void scan_32_plus1(volatile  T *s_scan,
                   const int    local_id)
{
    int ai, bi;
    const int baseai = 2 * local_id + 1;
    const int basebi = baseai + 1;
    T temp;

    if (local_id < 16)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0)  { s_scan[32] = s_scan[31] + s_scan[15]; s_scan[31] = s_scan[15]; s_scan[15] = 0; }
    if (local_id < 2)   { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)   { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)   { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

template<typename T>
__inline__ __device__
T scan_plus1_shfl(volatile  T *s_scan,
                  const int     local_id,
                  T r_in,
                  const int seg_num)
{
    // 3-stage method. scan-scan-propogate

    // shfl version
    const int lane_id = local_id % ANONYMOUSLIB_THREAD_BUNCH;
    const int seg_id = local_id / ANONYMOUSLIB_THREAD_BUNCH;

    // stage 1. thread bunch scan
    T r_scan = 0;

    //if (seg_id < seg_num)
    //{
        r_scan = scan_32_shfl<T>(r_in, lane_id);

        if (lane_id == ANONYMOUSLIB_THREAD_BUNCH - 1)
            s_scan[seg_id] = r_scan;

        r_scan = __shfl_up(r_scan, 1);
        r_scan = lane_id ? r_scan : 0;
    //}

    __syncthreads();

    // stage 2. one thread bunch scan
    r_in = (local_id < seg_num) ? s_scan[local_id] : 0;
    if (!seg_id)
        r_in = scan_32_shfl<T>(r_in, lane_id);

    if (local_id < seg_num)
        s_scan[local_id + 1] = r_in;

    // single thread in-place scan
    //scan_single<T>(s_scan, local_id, seg_num+1);

    __syncthreads();

    // stage 3. propogate (element-wise add) to all
    if (seg_id) // && seg_id < seg_num)
        r_scan += s_scan[seg_id];

    return r_scan;
}

template<typename T>
__inline__ __device__
void scan_256_plus1(volatile T *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    T temp;

    if (threadIdx.x < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex,
             const iT             i,
             float               *x)
{
    *x = tex1Dfetch<float>(d_x_tex, i);
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex,
             const iT             i,
             double              *x)
{
    int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
    *x = __hiloint2double(x_int2.y, x_int2.x);
}

__forceinline__ __device__
static double atomicAdd(double *addr, double val)
{
    double old = *addr, assumed;
    do
    {
        assumed = old;
        old = __longlong_as_double(
                    atomicCAS((unsigned long long int*)addr,
                              __double_as_longlong(assumed),
                              __double_as_longlong(val+assumed)));

    }while(assumed != old);

    return old;
}

__global__
void warmup_kernel(int *d_scan)
{
    volatile __shared__ int s_scan[ANONYMOUSLIB_CSR5_OMEGA];
    s_scan[threadIdx.x] = 1;
    scan_32<int>(s_scan, threadIdx.x);
    if(!blockIdx.x)
        d_scan[threadIdx.x] = s_scan[threadIdx.x];
}


#endif // UTILS_CUDA_H
