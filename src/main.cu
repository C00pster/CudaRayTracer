#include "utils/constants.h"
#include "math/math.cuh"
#include "primitives/sphere.h"
#include "primitives/primitive_list.h"
#include "camera/camera.h"
#include "materials/material.h"
#include "cuda/cuda_utils.h"
#include "rendering/rendering.h"
#include "materials/texture.cuh"

#define X 1920
#define Y 1080
#define RED 0
#define GREEN 1
#define BLUE 2

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Primitive **d_list, Primitive** d_world, Camera **d_camera,
                             int width, int height, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        Texture *checker = new CheckerTexture(
            new ConstantTexture(Vec4(0.2f, 0.3f, 0.1f)),
            new ConstantTexture(Vec4(0.9f, 0.9f, 0.9f))
        );
        d_list[0] = new Sphere(Vec4(0, -1000, 0), 1000, new Lambertian(checker));

        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec4 center(a + RND*0.9f, 0.2f, b + RND*0.9f);
                // if ((center - Vec4(4, 0.2f, 0)).length() > 0.9f) {
                    if (choose_mat < 0.8f) {
                        Color albedo = Color(RND, RND, RND) * Color(RND, RND, RND);
                        Vec4 center2 = center + Vec4(0.0f, RND * 0.5f, 0.0f);
                        d_list[i++] = new Sphere(center, center2, 0.2f, 
                            new Lambertian(new ConstantTexture(Vec4(RND * RND, RND * RND, RND * RND))));
                    } else if (choose_mat < 0.95f) {
                        d_list[i++] = new Sphere(
                            center, 
                            0.2f, 
                            new Metal(Vec4(0.5f * (1 + RND), 0.5f * (1 + RND), 0.5f * (1 + RND)), 0.5f * RND)
                        );
                    } else {
                        d_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
                    }
                // }
            }
        }
        d_list[i++] = new Sphere(Vec4(0, 1, 0), 1.0f, new Dielectric(1.5f));
        d_list[i++] = new Sphere(Vec4(-4, 1, 0), 1.0f, 
                        new Lambertian(
                            new ConstantTexture(
                                Vec4(0.4f, 0.2f, 0.1f))));
        d_list[i++] = new Sphere(Vec4(4, 1, 0), 1.0f, new Metal(Vec4(0.7f, 0.6f, 0.5f), 0.0f));
        *rand_state = local_rand_state;
        *d_world = new PrimitiveList(d_list, 22*22+1+3);

        Vec4 lookfrom(13, 2, 3);
        Vec4 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1f;
        *d_camera = new Camera(lookfrom, lookat, Vec4(0, 1, 0), 30.0f, float(X)/float(Y), aperture, dist_to_focus);
    }
}

__global__ void free_world(Primitive **d_list, Primitive **d_world, Camera **d_camera) {
    for (int i = 0; i < 22*22+1+3; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int threads_per_block = calculate_optimal_threads_per_block();
    int threads_per_block_x = sqrt(threads_per_block);
    int threads_per_block_y = threads_per_block / threads_per_block_x;
    int samples_per_pixel = 500;
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    std::cout << "Rendering image using CUDA\n";
    printf("Using threads per block (x,y): (%d,%d)\n", threads_per_block_x, threads_per_block_y);

    int num_pixels = X * Y;
    size_t framebuffer_size = num_pixels * sizeof(Color);

    Color *framebuffer;
    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));
    std::cout << "Allocated framebuffer\n";

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));
    std::cout << "Allocated rand state\n";

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Initialized rand state 2\n";

    Primitive **d_list;
    int num_primitives = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_primitives * sizeof(Primitive *)));
    Primitive **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Primitive *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, X, Y, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Initialized primitives, camera, and world\n";

    clock_t start, stop;
    start = clock();

    // Launch the kernel
    // dim3 is a CUDA type that represents a 3D grid
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
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(framebuffer));

    cudaDeviceReset();
}