#include "cuda_utils.cuh"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__
void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(1984, 0, 0, rand_state);
}

int calculate_optimal_threads_per_block() {
    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Assuming you're using device 0

    // Maximum threads per block allowed by the hardware
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    // Threads per warp is always 32
    int warpSize = deviceProp.warpSize;

    // Choose optimal block size as a multiple of warp size, not exceeding maxThreadsPerBlock
    int optimalThreadsPerBlock = maxThreadsPerBlock;

    // Adjust threads per block if necessary
    if (optimalThreadsPerBlock % warpSize != 0) {
        optimalThreadsPerBlock = (optimalThreadsPerBlock / warpSize) * warpSize;
    }

    return optimalThreadsPerBlock;
}