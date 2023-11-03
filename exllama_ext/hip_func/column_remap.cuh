// !!! This is a file automatically generated by hipify!!!
#ifndef _column_remap_cuh
#define _column_remap_cuh

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

void column_remap_cuda
(
    const half* x,
    half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map
);

#endif