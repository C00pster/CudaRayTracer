#include "primitives/world.cuh"
#include "camera/camera.cuh"
#include "cuda/cuda_utils.cuh"
#include "primitives/scene.cuh"
#include <cstdint>

#define X 1920
#define Y 1080
#define RED 0
#define GREEN 1
#define BLUE 2

#define RND (curand_uniform(&local_rand_state))

int main() {
    int threads_per_block = calculate_optimal_threads_per_block();
    int threads_per_block_x = sqrt(threads_per_block);
    int threads_per_block_y = threads_per_block / threads_per_block_x;
    int samples_per_pixel = 1000;
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    std::cout << "Creating world\n";

    int num_pixels = X * Y;
    size_t framebuffer_size = num_pixels * sizeof(Color);

    Color *framebuffer;
    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    World **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(World *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    globe(threads_per_block_x, threads_per_block_y, samples_per_pixel, framebuffer, 
          d_world, d_camera, X, Y, d_rand_state);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();
}