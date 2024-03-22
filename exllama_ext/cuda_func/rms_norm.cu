#include "rms_norm.cuh"

#ifndef USE_ROCM
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"


const int THREADS_X = 32;
const int THREADS_Y = 8;
const int BLOCKSIZE_X = 16;

// scratch = sum(x * x, dim = -1)

typedef void (*fp_rms_norm_row_product_kernel)
(
    half*,
    float*,
    const int,
    const int
);

template<bool use_half2>
__global__ void rms_norm_row_product_kernel
(
    half* __restrict__ x,
    float* __restrict__ scratch,
    const int rows,
    const int dim
)
{
    int column = (THREADS_X * blockIdx.x + threadIdx.x) * BLOCKSIZE_X;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;

    if (row >= rows) return;
    if (column >= dim) return;

//     if (column == 0)
//     {
//         scratch[row] = 0.0f;
//         __syncthreads();
//     }

    float acc = 0.0f;
    int idx = row * dim + column;

    // Accumulate

    if constexpr (use_half2)
    {
        half2* x_ptr = (half2*) &x[idx];

#pragma unroll
        for (int k = 0; k < BLOCKSIZE_X / 2; k++)
        {
            half2 x2 = *x_ptr++;
            float m0 = __half2float(x2.x);
            float m1 = __half2float(x2.y);
            acc = fma(m0, m0, acc);
            acc = fma(m1, m1, acc);
        }
    }
    else
    {
        half* x_ptr = x + idx;
#pragma unroll
        for (int k = 0; k < BLOCKSIZE_X; k++)
        {
            float m0 = __half2float(*x_ptr++);
            acc = fma(m0, m0, acc);
        }
    }

//     // Use Warp Shuffle to accumulate within the warp
//
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         acc += __shfl_down_sync(0xffffffff, acc, offset);
//     if (threadIdx.x % warpSize == 0)
//         atomicAdd(&scratch[row], acc);

    atomicAdd(&scratch[row], acc);
}

// x = x * w / sqrt(scratch / dim + epsilon)

typedef void (*fp_rms_norm_kernel)
(
    half*,
    const half*,
    half*,
    float*,
    const float,
    const float,
    const int,
    const int
);

template<bool use_half2>
__global__ void rms_norm_kernel
(
    half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out,
    float* __restrict__ scratch,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int dim
)
{
    int column = (THREADS_X * blockIdx.x + threadIdx.x) * BLOCKSIZE_X;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= rows) return;
    if (column >= dim) return;

    float rmf = rsqrtf(scratch[row] * r_dim + epsilon);
    half rm = __float2half_rn(rmf);
    half2 rm2 = __half2half2(rm);

    if constexpr (use_half2)
    {
        half2* x2_ptr = (half2*) &x[row * dim + column];
        half2* out2_ptr = (half2*) &out[row * dim + column];
        const half2* w2_ptr = (const half2*) &w[column];

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE_X / 2; k++)
        {
            half2 m2 = *x2_ptr++;
            half2 w2 = *w2_ptr++;
            m2 = __hmul2(m2, rm2);
            m2 = __hmul2(m2, w2);
            *out2_ptr++ = m2;
        }
    }
    else
    {
        half* x_ptr = &x[row * dim + column];
        half* out_ptr = &out[row * dim + column];
        const half* w_ptr = &w[column];

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE_X; k++)
        {
            half m = *x_ptr++;
            half w = *w_ptr++;
            m = __hmul(m, rm);
            m = __hmul(m, w);
            *out_ptr++ = m;
        }
    }

//     __syncthreads();
//     if (column >= dim - BLOCKSIZE_X) scratch[row] = 0.0f;
}

fp_rms_norm_row_product_kernel rms_norm_row_product_kernel_pick(ExLlamaTuning* tuningParams)
{
    // <bool use_half2>
    if (tuningParams->matmul_no_half2) {
        return rms_norm_row_product_kernel<false>;
    } else {
        return rms_norm_row_product_kernel<true>;
    }
};

fp_rms_norm_kernel rms_norm_kernel_pick(ExLlamaTuning* tuningParams)
{
    // <bool use_half2>
    if (tuningParams->matmul_no_half2) {
        return rms_norm_kernel<false>;
    } else {
        return rms_norm_kernel<true>;
    }
};
#else
#define NUM_WARPS (16)
#define WARP_SIZE (64)
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define BLOCK_SIZE WARP_SIZE
#define NUM_THREADS NUM_WARPS * WARP_SIZE

typedef void (*fp_rms_norm_kernel)
(
    const half*,
    const half*,
    half*,
    const float,
    const float,
    const int,
    const int
);

template <int blocks_per_warp>
__global__ void rms_norm_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int dim
)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x;
    const half2* x_row = (const half2*) (x + row * dim);
    half2* y_row = (half2*) (y + row * dim);
    const half2* w2 = (const half2*) w;

    // Compute sum of squares for each block

    float sum = 0.0f;
    float itemf[blocks_per_warp][2];
    __shared__ float sum_[1024];

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim / 2) break;

        half2 x2 = x_row[column];
        float f0 = __half2float(__low2half(x2));
        float f1 = __half2float(__high2half(x2));
        f0 = fmaxf(-65504.0f, fminf(f0, 65504.0f));
        f1 = fmaxf(-65504.0f, fminf(f1, 65504.0f));
        itemf[i][0] = f0;
        itemf[i][1] = f1;
        sum = fma(f0, f0, sum);
        sum = fma(f1, f1, sum);
    }
    // sum_[threadIdx.x]= sum;
    // Shuffle to sum across lanes

    __shared__ float sums[NUM_WARPS];

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor(sum, offset);
    // sum = 0;
    // for(int i = 0; i<1024;++i){
    //     sum += sum_[i];
    // }

    // Get norm

    float rmf = rsqrtf(sum * r_dim + epsilon);

    // Normalize x, scaling by w

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim / 2) return;
        half2 w2_ = w2[column];

        float x_itemf0 = itemf[i][0];
        float x_itemf1 = itemf[i][1];
        float w_itemf0 = __half2float(__low2half(w2_));
        float w_itemf1 = __half2float(__high2half(w2_));
        float n0 = x_itemf0 * w_itemf0 * rmf;
        float n1 = x_itemf1 * w_itemf1 * rmf;
        y_row[column] = __halves2half2(__float2half_rn(n0), __float2half_rn(n1));
    }
}

fp_rms_norm_kernel pick_rms_norm_kernel(const int blocks_per_warp)
{
    if (blocks_per_warp == 1) return rms_norm_kernel<1>;
    if (blocks_per_warp == 2) return rms_norm_kernel<2>;
    if (blocks_per_warp == 3) return rms_norm_kernel<3>;
    if (blocks_per_warp == 4) return rms_norm_kernel<4>;
    if (blocks_per_warp == 5) return rms_norm_kernel<5>;
    if (blocks_per_warp == 6) return rms_norm_kernel<6>;
    if (blocks_per_warp == 7) return rms_norm_kernel<7>;
    if (blocks_per_warp == 8) return rms_norm_kernel<8>;
	return NULL;
}
#endif

// x = x * w / sqrt(row_mean(x * x) + epsilon)
//
// works in-place if x == out

void rms_norm_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,
    const half* w,
    half* out,
    const float epsilon,
    const int rows,
    const int dim,
    const int device_index
)
{
#ifndef USE_ROCM
    CudaBuffers* buffers = get_buffers(device_index);
    float* temp = buffers->get_zeros_float(rows);

    float r_dim = 1.0f / (float) dim;

    dim3 threads2(THREADS_X, THREADS_Y, 1);
    dim3 blocks2
    (
        ((dim + THREADS_X - 1) / THREADS_X + THREADS_X - 1) / BLOCKSIZE_X,
        (rows + THREADS_Y - 1) / THREADS_Y,
        1
    );

    //cudaMemsetAsync(temp, 0, rows * sizeof(float));

    fp_rms_norm_row_product_kernel kernel1 = rms_norm_row_product_kernel_pick(tuningParams);
    kernel1<<<blocks2, threads2>>>(x, temp, rows, dim);

    fp_rms_norm_kernel kernel2 = rms_norm_kernel_pick(tuningParams);
    kernel2<<<blocks2, threads2>>>(x, w, out, temp, epsilon, r_dim, rows, dim);

    //cudaMemsetAsync(temp, 0, rows * sizeof(float));
#else
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS;
    blockDim.y = 1;
    gridDim.x = rows;
    gridDim.y = 1;

    float r_dim = 1.0f / (float) dim;

    int blocks_per_warp = DIVIDE(dim, NUM_THREADS * 2);
    fp_rms_norm_kernel kernel = pick_rms_norm_kernel(blocks_per_warp);
    kernel<<<gridDim, blockDim>>>(x, w, out, epsilon, r_dim, rows, dim);
#endif
}