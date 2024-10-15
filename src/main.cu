#include "utils/constants.h"
#include "math/math.cuh"
#include "primitives/sphere.h"
#include "primitives/world.h"
#include "camera/camera.h"
#include "materials/material.h"
#include "cuda/cuda_utils.h"
#include "rendering/rendering.h"
#include "materials/texture.cuh"
#include "primitives/scene.h"
#include <cstdint>

#define X 160
#define Y 90
#define RED 0
#define GREEN 1
#define BLUE 2

#define RND (curand_uniform(&local_rand_state))

int main() {
    int threads_per_block = calculate_optimal_threads_per_block();
    int threads_per_block_x = sqrt(threads_per_block);
    int threads_per_block_y = threads_per_block / threads_per_block_x;
    int samples_per_pixel = 1;
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

    create_scene(d_world, d_camera, d_rand_state, X, Y, 3);
    std::cout << "World created\n";

    clock_t start, stop;
    start = clock();

    dim3 blocks((X + threads_per_block_x - 1) / threads_per_block_x,
            (Y + threads_per_block_y - 1) / threads_per_block_y);
    dim3 threads(threads_per_block_x, threads_per_block_y);
    render_init<<<blocks, threads>>>(X, Y, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(framebuffer, X, Y, samples_per_pixel, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Time taken: " << timer_seconds << " seconds\n";

    write_framebuffer_to_file(framebuffer, X, Y);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();
}