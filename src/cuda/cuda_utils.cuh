#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <curand_kernel.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

__global__ void rand_init(curandState *rand_state);

int calculate_optimal_threads_per_block();

#endif // CUDA_UTILS_CUH