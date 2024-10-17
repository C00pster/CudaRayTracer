#ifndef RENDERING_CUH
#define RENDERING_CUH

#include "math/ray.cuh"
#include "primitives/primitive.cuh"
#include "primitives/world.cuh"
#include "camera/camera.cuh"
#include "materials/material.cuh"
#include <curand_kernel.h>
#include <cfloat>

__device__ 
Vec4 color(const Ray& r, World** world, curandState *local_rand_state);

__global__ 
void render_init(int width, int height, curandState *rand_state);

__global__ 
void render(
    Color *framebuffer, 
    int width, 
    int height, 
    int ns, 
    Camera **cam, 
    World** world, 
    curandState *rand_state
);

#endif // RENDERING_CUH