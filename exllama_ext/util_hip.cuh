// !!! This is a file automatically generated by hipify!!!
#ifndef _util_cuh
#define _util_cuh

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

#if defined(USE_ROCM)
#define cudaUnspecified hipErrorUnknown
#else
#define cudaUnspecified hipErrorApiFailureBase
#endif

// React to failure on return code != hipSuccess

#define _cuda_check(fn) \
do { \
    {_cuda_err = fn;} \
    if (_cuda_err != hipSuccess) goto _cuda_fail; \
} while(false)

// React to failure on return code == 0

#define _alloc_check(fn) \
do { \
    if (!(fn)) { _cuda_err = cudaUnspecified; goto _cuda_fail; } \
    else _cuda_err = hipSuccess; \
} while(false)

// Clone CPU <-> CUDA

template <typename T>
T* cuda_clone(const void* ptr, int num)
{
    T* cuda_ptr;
    hipError_t r;

    r = hipMalloc(&cuda_ptr, num * sizeof(T));
    if (r != hipSuccess) return NULL;
    r = hipMemcpy(cuda_ptr, ptr, num * sizeof(T), hipMemcpyHostToDevice);
    if (r != hipSuccess) return NULL;
    hipDeviceSynchronize();
    return cuda_ptr;
}

template <typename T>
T* cpu_clone(const void* ptr, int num)
{
    T* cpu_ptr;
    hipError_t r;

    cpu_ptr = (T*) malloc(num * sizeof(T));
    if (cpu_ptr == NULL) return NULL;
    r = hipMemcpy(cpu_ptr, ptr, num * sizeof(T), hipMemcpyDeviceToHost);
    if (r != hipSuccess) return NULL;
    hipDeviceSynchronize();
    return cpu_ptr;
}

// Pack two half values into a half2, host version

__host__ inline __half2 pack_half2(__half h1, __half h2)
{
    unsigned short s1 = *reinterpret_cast<unsigned short*>(&h1);
    unsigned short s2 = *reinterpret_cast<unsigned short*>(&h2);
    ushort2 us2 = make_ushort2(s1, s2);
    return *reinterpret_cast<__half2*>(&us2);
}

#endif