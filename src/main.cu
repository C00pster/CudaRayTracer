#include "utils/constants.h"
#include "math/math.cuh"
#include "primitives/sphere.h"
#include "primitives/world.h"
#include "camera/camera.h"
#include "materials/material.h"
#include "cuda/cuda_utils.h"
#include "rendering/rendering.h"
#include "materials/texture.cuh"
#include <cstdint>

#define X 1920
#define Y 1080
#define RED 0
#define GREEN 1
#define BLUE 2

#define RND (curand_uniform(&local_rand_state))

__global__
void create_world(Primitive **d_list, Primitive **d_sorted_list, World **d_world, int64_t* morton_codes, int64_t* sorted_morton_codes,
                  Camera **d_camera, int width, int height, curandState *rand_state, int num_primitives) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        Texture *checker = new CheckerTexture(
            new ConstantTexture(Vec4(0.2f, 0.3f, 0.1f)),
            new ConstantTexture(Vec4(0.9f, 0.9f, 0.9f))
        );
        d_list[0] = new Sphere(Vec4(0, -1000, 0), 1000, new Lambertian(checker));
        d_list[1] = new Sphere(Vec4(0, 1, 0), 1.0f, new Dielectric(1.5f));
        d_list[2] = new Sphere(Vec4(-4, 1, 0), 1.0f, 
                        new Lambertian(
                            new ConstantTexture(
                                Vec4(0.4f, 0.2f, 0.1f))));
        d_list[3] = new Sphere(Vec4(4, 1, 0), 1.0f, new Metal(Vec4(0.7f, 0.6f, 0.5f), 0.0f));
        *rand_state = local_rand_state;
        *d_world = new World(d_list, d_sorted_list, num_primitives, morton_codes, sorted_morton_codes);

        Vec4 lookfrom(13, 2, 3);
        Vec4 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera = new Camera(lookfrom, lookat, Vec4(0, 1, 0), 30.0f, float(X)/float(Y), aperture, dist_to_focus);
    }
}

__global__ void free_world(Primitive **d_list, Camera **d_camera) {
    for (int i = 0; i < 22*22+1+3; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_camera;
}

int main() {
    int threads_per_block = calculate_optimal_threads_per_block();
    int threads_per_block_x = sqrt(threads_per_block);
    int threads_per_block_y = threads_per_block / threads_per_block_x;
    int samples_per_pixel = 1000;
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    std::cout << "Rendering image using CUDA\n";

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

    Primitive **d_list;
    Primitive **d_sorted_list;
    int num_primitives = 4;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_primitives * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&d_sorted_list, (num_primitives * 2 + 1) * sizeof(Primitive *)));
    World **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(World *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    int64_t *morton_codes;
    int64_t *sorted_codes;
    checkCudaErrors(cudaMalloc((void **)&morton_codes, num_primitives * sizeof(int64_t)));
    checkCudaErrors(cudaMalloc((void **)&sorted_codes, num_primitives * sizeof(int64_t)));
    create_world<<<1, 1>>>(d_list, d_sorted_list, d_world, morton_codes, sorted_codes, d_camera, X, Y, d_rand_state2, num_primitives);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
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
    free_world<<<1, 1>>>(d_list, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();
}