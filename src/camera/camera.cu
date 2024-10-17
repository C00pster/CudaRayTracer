#include "camera.cuh"

__device__
Vec4 random_in_unit_disk(curandState *local_rand_state) {
    Vec4 p;
    do {
        p = 2.0f*Vec4(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec4(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}