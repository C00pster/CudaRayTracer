#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <string>
#include <iostream>
#include "math/vec4.h"
#include "cuda/cuda_utils.h"

__global__
void checkered_spheres_device(Primitive **d_list, Primitive **d_sorted_list, World **d_world, int64_t* morton_codes, int64_t* sorted_morton_codes,
                  Camera **d_camera, int width, int height, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Texture* checker = new CheckerTexture(
        0.32f, 
        Color(0.2f, 0.3f, 0.1f),
        Color(0.9f, 0.9f, 0.9f)
    );
    int counter = 0;
    d_list[counter++] = new Sphere(Vec4(0, -10, 0), 10, new Lambertian(checker));
    d_list[counter++] = new Sphere(Vec4(0, 10, 0), 10, new Lambertian(checker));

    *d_camera = new Camera(Vec4(13, 2, 3), Vec4(0, 0, 0), Vec4(0, 1, 0), 30.0f, float(width)/float(height), 0.1f, 10.0f);

    *d_world = new World(d_list, d_sorted_list, 2, morton_codes, sorted_morton_codes);
}

__global__
void bouncing_spheres_device(Primitive **d_list, Primitive **d_sorted_list, World **d_world, int64_t* morton_codes, int64_t* sorted_morton_codes,
                  Camera **d_camera, int width, int height, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int num_primitives = 4;
    
    Texture *checker = new CheckerTexture(
        0.4f,
        new ConstantTexture(Vec4(0.2f, 0.3f, 0.1f)),
        new ConstantTexture(Vec4(0.9f, 0.9f, 0.9f))
    );

    int counter = 0;
    d_list[counter++] = new Sphere(Vec4(0, -1000, 0), 1000, new Lambertian(checker));
    d_list[counter++] = new Sphere(Vec4(0, 1, 0), 1.0f, new Dielectric(1.5f));
    d_list[counter++] = new Sphere(Vec4(-4, 1, 0), 1.0f, 
                    new Lambertian(
                        new ConstantTexture(
                            Vec4(0.4f, 0.2f, 0.1f))));
    d_list[counter++] = new Sphere(Vec4(4, 1, 0), 1.0f, new Metal(Vec4(0.7f, 0.6f, 0.5f), 0.0f));

    *d_camera = new Camera(Vec4(13, 2, 3), Vec4(0, 0, 0), Vec4(0, 1, 0), 30.0f, float(width)/float(height), 0.1f, 10.0f);

    *d_world = new World(d_list, d_sorted_list, num_primitives, morton_codes, sorted_morton_codes);
}

__host__
void create_scene(World** d_world, Camera** d_camera, curandState *rand_state, int X, int Y, int scene) {
    int num_primitives;
    Primitive **d_list;
    Primitive **d_sorted_list;
    int64_t *morton_codes;
    int64_t *sorted_morton_codes;

    switch(scene) {
        case 1:
            num_primitives = 4;
            break;
        case 2:
            num_primitives = 2;
            break;
        default:
            std::cerr << "Invalid scene number" << std::endl;
            exit(1);
    }
    checkCudaErrors(cudaMalloc((void **)&d_list, num_primitives * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&d_sorted_list, (num_primitives * 2 - 1) * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&morton_codes, num_primitives * sizeof(int64_t)));
    checkCudaErrors(cudaMalloc((void **)&sorted_morton_codes, num_primitives * sizeof(int64_t)));

    switch(scene) {
        case 1:      
            bouncing_spheres_device<<<1, 1>>>(d_list, d_sorted_list, d_world, morton_codes, sorted_morton_codes, d_camera, X, Y, rand_state);
            break;
        case 2:
            checkered_spheres_device<<<1, 1>>>(d_list, d_sorted_list, d_world, morton_codes, sorted_morton_codes, d_camera, X, Y, rand_state);
            break;
        default:
            std::cerr << "Invalid scene number" << std::endl;
            exit(1);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // SCENE_H