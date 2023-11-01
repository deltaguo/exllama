#include "q4_matmul.cuh"
#include "column_remap.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "../cuda_compat.cuh"
#include "../cuda_buffers.cuh"
#if defined(USE_ROCM)
#define WMMA_KERNEL
#include <rocprim/rocprim.hpp>
#include <hip/hip_ext.h>
#include "rocwmma/rocwmma.hpp"
#include "../hip_compat.cuh"
typedef __fp16 half8 __attribute__((ext_vector_type(8)));
typedef __fp16 half4 __attribute__((ext_vector_type(4)));
typedef float mfma_float4 __attribute__((ext_vector_type(4)));

#define FLOAT2(pointer) (reinterpret_cast<float2 *>((void *)&(pointer))[0])
#define MFMA_FLOAT4(pointer) (reinterpret_cast<mfma_float4 *>((void *)&(pointer))[0])
#define HALF8(pointer) (reinterpret_cast<half8 *>((void *)&(pointer))[0])
#define HALF4(pointer) (reinterpret_cast<half4 *>((void *)&(pointer))[0])
#define HALF2(pointer) (reinterpret_cast<half2 *>((void *)&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float *>((void *)&(pointer))[0])
const int THREADS_X = 64; // Block size and thread count along columns in w and out
#else
const int THREADS_X = 32; // Block size and thread count along columns in w and out
#endif
const int THREADS_Y = 1; // Block size and thread count along rows in x and out

#if defined(USE_SMEM)
const int GROUP_STEP = 32; // Assumed group size when block_size_z % groupsize != 0
#endif

#if defined(USE_ROCM)
template <typename Y, typename X>
__host__ __device__ constexpr Y bit_cast(const X &x)
{
    union AsType
    {
        X x;
        Y y;
    };
    return AsType{x}.y;
}

// transpose fp16 2x2
__device__ void transpose_fp16_2x2_register(const half2 &x0, const half2 &x1, half2 &y0, half2 &y1)
{
    constexpr int32_t m0 = 0x05040100;
    constexpr int32_t m1 = 0x07060302;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 04 01 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 44 77 88
    // index is reversed because of little endianness (least significant bits first)
    y0 = bit_cast<half2>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0));
    y1 = bit_cast<half2>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m1));
}

__device__ __forceinline__ half dot_product_8_wmma(
    const half acc,
    MatrixView_half &h_,
    const int h_row,
    const int h_column, // divisible by 8
    MatrixView_q4_column &v_,
    const int v_row, // divisible by 8
    const int v_column,
    const half v_scale,
    const uint32_t v_zero, // + 1 (!!)
    const int count)
{
    // 向量化访存
    using int16_tx2 = __attribute__((__vector_size__(2 * sizeof(int16_t)))) int16_t;
    using halfx4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#define INT_16X2(pointer) ((bit_cast<int16_tx2 *>((void *)&(pointer)))[0])
#define INT_32(pointer) ((bit_cast<int32_t *>((void *)&(pointer)))[0])
#define HALF2(pointer) ((bit_cast<half2 *>((void *)&(pointer)))[0])
#define HALFX4(pointer) ((bit_cast<halfx4 *>((void *)&(pointer)))[0])
#define FLOATX4(pointer) ((bit_cast<floatx4 *>((void *)&(pointer)))[0])

    const int K_tile = 32;
    const half *__restrict__ h_ptr = h_.item_ptr(h_row, h_column);
    uint32_t *__restrict__ v_ptr = (uint32_t *)v_.item_uint32_ptr(v_row, v_column);
    short int v_zero_short = (short int)v_zero;

    int16_t v_zero_2_add16[2];
    v_zero_2_add16[0] = 16 - v_zero_short;
    v_zero_2_add16[1] = 16 - v_zero_short;

    half2 val_1040_half2;
    val_1040_half2 = __half2half2(__float2half(1040));
    half val_1024_half;
    val_1024_half = __float2half(1040);

    // matrix core 寄存器，全部申请为数组，使用时使用向量化访存来做
    half fragA[K_tile / 4][4];
    half fragB[K_tile / 4][4];
    float fragAcc[4] = {(0.0f)};

    // 读取B，使用数组原因：一方面，unroll展开后数组的读取效率高。另一方面，可以计算和访存可以重叠
    short int v_read_B_q[K_tile / 8][2];

    // 用于读取A和最后Acc的数据移动
    __shared__ half sh_A[K_tile < 64 ? 64 : K_tile];

// 读取B
#pragma unroll
    for (int i = 0; i < K_tile / 8; i++)
    {
        HALF2(v_read_B_q[i][0]) = HALF2(*v_ptr);
        v_ptr += v_.width;
    }

// 转化数据并放入fragB中
#pragma unroll
    for (int k_index = 0; k_index < K_tile / 4; k_index++)
    {
        fragB[k_index][0] = __short2half_rn((short int)((v_read_B_q[k_index / 2][k_index % 2] >> (0)) & 0x0f) - v_zero_short);
        fragB[k_index][1] = __short2half_rn((short int)((v_read_B_q[k_index / 2][k_index % 2] >> (4)) & 0x0f) - v_zero_short);
        fragB[k_index][2] = __short2half_rn((short int)((v_read_B_q[k_index / 2][k_index % 2] >> (8)) & 0x0f) - v_zero_short);
        fragB[k_index][3] = __short2half_rn((short int)((v_read_B_q[k_index / 2][k_index % 2] >> (12)) & 0x0f) - v_zero_short);
    }

    int j = 0;

    for (; j < count - K_tile / 8; j = j + K_tile / 8)
    {

// 预取下一个B，让B的读取和A的读取，计算重叠
#pragma unroll
        for (int i = 0; i < K_tile / 8; i++)
        {
            HALF2(v_read_B_q[i][0]) = HALF2(*v_ptr);
            v_ptr += v_.width;
        }

        /**
         * 这里是为了不浪费线程。假设K_tile=16。
         * 那么block需要读取16个数据，但是有64个线程，有48个线程空闲。不妨我们读64个数据，反正这些数据后面也要读取。
         * 这样每隔4次循环就读取一次
         **/
        if ((j * 8) % 128 == 0)
        {
            HALF2(sh_A[2 * threadIdx.x]) = HALF2(*(h_ptr + (j * 8) + 2 * threadIdx.x));
        }

        __syncthreads();

// 读取到fragA
#pragma unroll
        for (int k_index = 0; k_index < K_tile / 4; k_index++)
        {
            HALFX4(fragA[k_index][0]) = HALFX4(*(sh_A + (j * 8) % 128 / K_tile * K_tile + k_index * 4));
        }

#pragma unroll
        for (int k_index = 0; k_index < K_tile / 8; k_index++)
        {
            int16_t tmp0[2];
            INT_32(tmp0[0]) = (((INT_32(v_read_B_q[k_index][0]) >> 0) & 0x000f000f) + INT_32(v_zero_2_add16[0])) | (0x64006400);
            HALF2(tmp0[0]) = __hsub2(HALF2(tmp0[0]), val_1040_half2);

            int16_t tmp1[2];
            INT_32(tmp1[0]) = (((INT_32(v_read_B_q[k_index][0]) >> 4) & 0x000f000f) + INT_32(v_zero_2_add16[0])) | (0x64006400);
            HALF2(tmp1[0]) = __hsub2(HALF2(tmp1[0]), val_1040_half2);

            int16_t tmp2[2];
            INT_32(tmp2[0]) = (((INT_32(v_read_B_q[k_index][0]) >> 8) & 0x000f000f) + INT_32(v_zero_2_add16[0])) | (0x64006400);
            HALF2(tmp2[0]) = __hsub2(HALF2(tmp2[0]), val_1040_half2);

            int16_t tmp3[2];
            INT_32(tmp3[0]) = (((INT_32(v_read_B_q[k_index][0]) >> 12) & 0x000f000f) + INT_32(v_zero_2_add16[0])) | (0x64006400);
            HALF2(tmp3[0]) = __hsub2(HALF2(tmp3[0]), val_1040_half2);

            FLOATX4(fragAcc[0]) = __builtin_amdgcn_mfma_f32_4x4x4f16(HALFX4(fragA[2 * k_index][0]), HALFX4(fragB[2 * k_index][0]), FLOATX4(fragAcc[0]), 0, 0, 0);
            FLOATX4(fragAcc[0]) = __builtin_amdgcn_mfma_f32_4x4x4f16(HALFX4(fragA[2 * k_index + 1][0]), HALFX4(fragB[2 * k_index + 1][0]), FLOATX4(fragAcc[0]), 0, 0, 0);

            transpose_fp16_2x2_register(HALF2(tmp0[0]), HALF2(tmp1[0]), HALF2(fragB[2 * k_index][0]), HALF2(fragB[2 * k_index + 1][0]));
            transpose_fp16_2x2_register(HALF2(tmp2[0]), HALF2(tmp3[0]), HALF2(fragB[2 * k_index][2]), HALF2(fragB[2 * k_index + 1][2]));
        }
    }

    if ((j * 8) % 128 == 0 && threadIdx.x < K_tile)
    { // 如果 group_size 是 128 的倍数，这里不需要数据读取
        sh_A[threadIdx.x] = *(h_ptr + (j * 8) + threadIdx.x);
    }

    __syncthreads();

#pragma unroll
    for (int k_index = 0; k_index < K_tile / 4; k_index++)
    {
        HALFX4(fragA[k_index][0]) = HALFX4(*(sh_A + (j * 8) % 128 / K_tile * K_tile + k_index * 4));
    }

#pragma unroll
    for (int k_index = 0; k_index < K_tile / 4; k_index++)
    {
        FLOATX4(fragAcc[0]) = __builtin_amdgcn_mfma_f32_4x4x4f16(HALFX4(fragA[k_index][0]), HALFX4(fragB[k_index][0]), FLOATX4(fragAcc[0]), 0, 0, 0);
    }

    // *(sh_A + threadIdx.x % 4 + threadIdx.x / 4 * 4) = __float2half(fragAcc[0]);

    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.z == 0) {
    //     printf("fragAcc[n_index][0]: %f \n", fragAcc[0]);
    // }
    half result = __hadd(__hmul(__float2half(fragAcc[0]), v_scale), acc);

    return result;
}
#endif

typedef void (*fp_q4_matmul_kernel)(
    const half *,
    const uint32_t *,
    half *,
    const half *,
    const uint32_t *,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint32_t *,
    bool);

template <bool use_half2, bool use_groupsize, bool use_x_map>
__global__ void q4_matmul_kernel(
    const half *__restrict__ x,
    const uint32_t *__restrict__ w,
    half *__restrict__ out,
    const half *__restrict__ w_scales,
    const uint32_t *__restrict__ w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const int block_size_z,
    const uint32_t *__restrict__ x_map,
    bool no_zero)
{
#ifndef WMMA_KERNEL
#if defined(USE_SMEM)

    extern __shared__ half2 x_cache[];
    half *x_cache_h = (half *)x_cache;

#endif

    // Start of block

    int x_column = block_size_z * blockIdx.z;
    int x_column_end = min(dim, block_size_z * (blockIdx.z + 1));

    int w_column = THREADS_X * blockIdx.x + threadIdx.x; // assume width of weight matrix divisible by THREADS_X
    int x_row = THREADS_Y * blockIdx.y + threadIdx.y;

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    if (!no_zero && blockIdx.z == 0 && (threadIdx.x & 1) == 0)
    {
        *((uint32_t *)out_.item_ptr(x_row, w_column)) = 0;
    }
    __syncthreads();

    // Loop over part of x row (and w column)

    half2 acc = {};
    half acc_h = {};

    if constexpr (use_groupsize)
    {
        // For quant matrices where groupsize divides BLOCK_SIZE_Z we always start on a group boundary, so this
        // could be slightly faster

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++, k += groupsize)
        {
#if defined(USE_SMEM)

            for (int i = threadIdx.x; i < groupsize; i += THREADS_X)
            {
                if constexpr (use_x_map)
                    x_cache_h[i] = *x_.item_ptr(x_row, x_map[k + i]);
                else
                    x_cache_h[i] = *x_.item_ptr(x_row, k + i);
            }
            __syncthreads();

            if constexpr (use_half2)
            {
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc = dot_product_8(acc, x_cache, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
            else
            {
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc_h = dot_product_8_h(acc_h, x_cache_h, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
            __syncthreads();

#else

            if constexpr (use_half2)
            {
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map)
                    acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else
                    acc = dot_product_8(acc, (const half2 *)x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
            else
            {
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map)
                    acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else
                    acc_h = dot_product_8_h(acc_h, x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }

#endif
        }
    }
    else
    {
        // Otherwise assume groupsize is a multiple of GROUP_STEP, do GROUP_STEP columns per iteration and trust the cache

#if defined(USE_SMEM)

        for (int k = x_column; k < x_column + iterations * 8; k += GROUP_STEP)
        {
            for (int i = threadIdx.x; i < GROUP_STEP; i += THREADS_X)
            {
                if constexpr (use_x_map)
                    x_cache_h[i] = *x_.item_ptr(x_row, x_map[k + i]);
                else
                    x_cache_h[i] = *x_.item_ptr(x_row, k + i);
            }
            __syncthreads();

            if constexpr (use_half2)
            {
                int group = k / groupsize;
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc = dot_product_8(acc, x_cache, w_, k, w_column, w_scale, w_zero, GROUP_STEP / 8);
            }
            else
            {
                int group = k / groupsize;
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc_h = dot_product_8_h(acc_h, x_cache_h, w_, k, w_column, w_scale, w_zero, GROUP_STEP / 8);
            }
            __syncthreads();
        }

#else

        for (int k = x_column; k < x_column + iterations * 8; k += 8)
        {
            if constexpr (use_half2)
            {
                int group = k / groupsize;
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map)
                    acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else
                    acc = dot_product_8(acc, (const half2 *)x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, 1);
            }
            else
            {
                int group = k / groupsize;
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map)
                    acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else
                    acc_h = dot_product_8_h(acc_h, x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, 1);
            }
        }

#endif
    }

    // Add to block result

    if constexpr (use_half2)
    {
        half result = __hadd(acc.x, acc.y);
        atomicAdd(out_.item_ptr(x_row, w_column), result);
    }
    else
    {
        atomicAdd(out_.item_ptr(x_row, w_column), acc_h);
    }
#else
    int x_column = block_size_z * blockIdx.z;
    int x_column_end = min(dim, block_size_z * (blockIdx.z + 1));
    int w_column = THREADS_X * blockIdx.x + threadIdx.x; // assume width of weight matrix divisible by THREADS_X (32)
    int x_row = THREADS_Y * blockIdx.y + threadIdx.y;    // 0

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    if (!no_zero && blockIdx.z == 0 && (threadIdx.x & 1) == 0)
    {
        *((uint32_t *)out_.item_ptr(x_row, w_column)) = 0;
    }
    __syncthreads();

    // Loop over part of x row (and w column)

    half2 acc = {};
    half acc_h = __float2half(0);

    if constexpr (use_groupsize)
    {

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++, k += groupsize)
        {
            {
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                acc_h = dot_product_8_wmma(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, min(block_size_z / 8, groupsize / 8));
            }
        }
    }

    half result;
    result = acc_h;
    __shared__ half sh_result[64];
    sh_result[threadIdx.x] = result;
    __syncthreads();
    if (threadIdx.x % 2 == 0)
    {
        atomicAdd((half2 *)out_.item_ptr(x_row, w_column), HALF2(sh_result[threadIdx.x]));
    }

#endif
}

fp_q4_matmul_kernel q4_matmul_kernel_pick(ExLlamaTuning *tuningParams, int block_size_z, int groupsize, uint32_t *x_map)
{
// <bool use_half2, bool use_groupsize, bool use_x_map>
#if defined(USE_ROCM)
    return q4_matmul_kernel<false, true, false>;
#endif
    if (tuningParams->matmul_no_half2)
    {
        if (block_size_z % groupsize == 0)
        {
            if (x_map)
                return q4_matmul_kernel<false, true, true>;
            else
                return q4_matmul_kernel<false, true, false>;
        }
        else
        {
            if (x_map)
                return q4_matmul_kernel<false, false, true>;
            else
                return q4_matmul_kernel<false, false, false>;
        }
    }
    else
    {
        if (block_size_z % groupsize == 0)
        {
            if (x_map)
                return q4_matmul_kernel<true, true, true>;
            else
                return q4_matmul_kernel<true, true, false>;
        }
        else
        {
            if (x_map)
                return q4_matmul_kernel<true, false, true>;
            else
                return q4_matmul_kernel<true, false, false>;
        }
    }
};

// Compute y = x @ w

void q4_matmul_cuda(
    ExLlamaTuning *tuningParams,
    const half *x,
    const int x_height,
    const Q4Matrix *w,
    half *out,
    bool no_zero,
    cudaStream_t alt_stream)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);

    uint32_t *x_map = w->cuda_x_map;
    const half *x_mapped = x;
    if (x_map && !tuningParams->matmul_fused_remap && !alt_stream)
    {
        CudaBuffers *buffers = get_buffers(w->device);
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
        x_map = NULL;
    }

    int block_size_z;
    if (w->width == 4096)
        block_size_z = 384; // 7B
    else if (w->width == 11008)
        block_size_z = 256;
    else if (w->width == 5120)
        block_size_z = 384; // 13B
    else if (w->width == 13824)
        block_size_z = 256;
    else if (w->width == 6656)
        block_size_z = 256; // 33B
    else if (w->width == 17920)
        block_size_z = 128;
    else
        block_size_z = 256;

    // if (!no_zero) cudaMemsetAsync(out, 0, x_height * w->width * sizeof(half));

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (dim + block_size_z - 1) / block_size_z);

    fp_q4_matmul_kernel kernel = q4_matmul_kernel_pick(tuningParams, block_size_z, w->groupsize, x_map);

#if defined(USE_SMEM)

    int shared_mem = (block_size_z % w->groupsize == 0 ? w->groupsize : GROUP_STEP) * sizeof(half);

#else

    int shared_mem = 0;

#endif

    kernel<<<blocks, threads, shared_mem, alt_stream>>>(x_mapped, w->cuda_qweight, out, w->cuda_scales, w->cuda_qzeros, height, dim, width, w->groupsize, block_size_z, x_map, no_zero);
}

void q4_matmul_recons_cuda(
    ExLlamaTuning *tuningParams,
    const half *x,
    const int x_height,
    Q4Matrix *w,
    half *out,
    const cublasHandle_t handle,
    bool no_zero)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);
    CudaBuffers *buffers = get_buffers(w->device);

    const half *x_mapped = x;
    if (w->cuda_x_map)
    {
        TORCH_CHECK(buffers->temp_state_size >= x_height * dim, "temp_state buffer is too small");
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
    }

    w->reconstruct(buffers->temp_dq);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700

    const float alpha = 1.0f;
    const float beta = no_zero ? 1.0f : 0.0f;
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, CUDA_R_16F, width,
                  x_mapped, CUDA_R_16F, dim, &beta, out, CUDA_R_16F, width);

#else

    const half alpha = __float2half(1.0f);
    const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, width, x_mapped, dim, &beta, out, width);

#endif
}
