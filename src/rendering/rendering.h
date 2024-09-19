#ifndef RENDERING_H
#define RENDERING_H

#include "math/ray.h"
#include "primitives/primitive.h"
#include <curand_kernel.h>

__device__ Vec4 color(const Ray& r, Primitive **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Vec4 cur_attenuation = Vec4(1.0f, 1.0f, 1.0f);

    #pragma unroll 2
    for (int i = 0; i < 50; i++) {
        HitRecord rec;

        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec4 attenuation;
            if (!rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                return Vec4(0, 0, 0);
            } else {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
        } else {
            Vec4 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec4 color = (1.0f - t) * Vec4(1.0f, 1.0f, 1.0f) + t * Vec4(0.5f, 0.7f, 1.0f);
            return cur_attenuation * color;
        }
    }

    return Vec4(0, 0, 0); // Exceeded recursion
}

__global__ void render_init(int width, int height, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;

    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(Color *framebuffer, int width, int height, int ns, Camera **cam, 
                       Primitive **world, curandState *rand_state) {
    // Thread coordinates
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the thread is outside the image
    if ((i >= width) || (j >= height)) return;

    int pixel_index = j * width + i; // Row offset + Column offset
    curandState local_rand_state = rand_state[pixel_index];
    Vec4 col(0, 0, 0);


    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col.apply_sqrt();
    framebuffer[pixel_index] = col;
}

#endif // RENDERING_H