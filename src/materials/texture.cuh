#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "perlin.cuh"

class Texture {
public:
    __device__ virtual Vec4 value(float u, float v, const Color &p) const = 0;
};

class ConstantTexture : public Texture {
public:
    __device__ ConstantTexture() {}
    __device__ ConstantTexture(const Color &c) : color(c) {}
    __device__ virtual Vec4 value(float u, float v, const Vec4 &p) const {
        return color;
    }

private:
    Vec4 color;
};

class CheckerTexture : public Texture {
public:
    __device__ CheckerTexture() {}
    __device__ CheckerTexture(Texture *e, Texture *o) :  even(e), odd(o) {}
    __device__ virtual Vec4 value(float u, float v, const Vec4 &p) const {
        float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0) {
            return odd->value(u, v, p);
        } else {
            return even->value(u, v, p);
        }
    }

private:
    float inv_scale;
    Texture *even;
    Texture *odd;
};

class NoiseTexture : public Texture {
public:
    __device__ NoiseTexture(int scale, curandState* local_rand_state) : scale(scale), noise(Perlin(local_rand_state)) {}
    __device__ virtual Color value(float u, float v, const Vec4& p) const {
        return Color(1, 1, 1) * noise.noise(scale, p);
    }

    Perlin noise;
    int scale;
};

class ImageTexture : public Texture {
public:
    __device__ ImageTexture() {}
    __device__ ImageTexture(unsigned char *pixels, int width, int height) : data(pixels), nx(width), ny(height) {}
    __device__ virtual Color value(float u, float v, const Vec4& p) const {
        int i = u * nx;
        int j = (1 - v) * ny - 0.001;
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i >= nx) i = nx - 1;
        if (j >= ny) j = ny - 1;
        float r = int(data[3 * i + 3 * nx * j]) / 255.0;
        float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
        float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
        return Color(r, g, b);
    }

    unsigned char *data;
    int nx, ny;
};

#endif // TEXTURE_CUH