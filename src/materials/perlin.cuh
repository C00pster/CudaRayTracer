#ifndef PERLIN_CUH
#define PERLIN_CUH

__device__ float* perlin_generate(curandState* local_rand_state) {
    float* p = new float[256];
    for (int i = 0; i < 256; ++i) {
        p[i] = curand_uniform(local_rand_state);
    }
    return p;
}

__device__ void permute(int *p, int n, curandState* local_rand_state) {
    for (int i = n - 1; i > 0; --i) {
        int target = curand_uniform(local_rand_state) * (i + 1);
        int temp = p[i];
        p[i] = p[target];
        p[target] = temp;
    }
}

__device__ int* perlin_generate_perm(curandState* local_rand_state) {
    int* p = new int[256];
    for (int i = 0; i < 256; ++i) {
        p[i] = i;
    }
    permute(p, 256, local_rand_state);
    return p;
}

class Perlin {
public:
    __device__ Perlin(curandState* local_rand_state) {
        ranfloat = perlin_generate(local_rand_state);
        perm_x = perlin_generate_perm(local_rand_state);
        perm_y = perlin_generate_perm(local_rand_state);
        perm_z = perlin_generate_perm(local_rand_state);
    }

    __device__ float noise(int scale, const Vec4& p) const {
        int x = uint8_t(p.x() * scale) & 255;
        int y = uint8_t(p.y() * scale) & 255;
        int z = uint8_t(p.z() * scale) & 255;
        return ranfloat[perm_x[x] ^ perm_y[y] ^ perm_z[z]];
    }

    curandState* local_rand_state;
    float* ranfloat;
    int* perm_x;
    int* perm_y;
    int* perm_z;
};

#endif // PERLIN_CUH