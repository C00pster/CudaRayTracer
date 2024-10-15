#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "math/vec4.h"

class Texture {
    public:
        ~Texture() = default;

        __device__ 
        virtual Vec4 value(float u, float v, const Color &p) const = 0;
};

class ConstantTexture : public Texture {
    public:
        __device__
        ConstantTexture() {}

        __device__
        ConstantTexture(const Color &c) : color(c) {}

        __device__
        virtual Vec4 value(float u, float v, const Vec4 &p) const {
            return color;
        }

    private:
        Vec4 color;
};

class CheckerTexture : public Texture {
    public:
        __device__
        CheckerTexture() {}

        __device__
        CheckerTexture(float scale, Texture *e, Texture *o) : inv_scale(1.0f / scale), even(e), odd(o) {}

        __device__
        CheckerTexture(float scale, const Color &c1, const Color &c2) : inv_scale(1.0f / scale), even(new ConstantTexture(c1)), odd(new ConstantTexture(c2)) {}

        __device__
        virtual Vec4 value(float u, float v, const Vec4 &p) const {
            int x = int(std::floor(inv_scale * p.x()));
            int y = int(std::floor(inv_scale * p.y()));
            int z = int(std::floor(inv_scale * p.z()));

            bool isEven = (x + y + z) % 2 == 0;

            return isEven ? even->value(u, v, p) : odd->value(u, v, p);
        }

    private:
        float inv_scale;
        Texture *even;
        Texture *odd;
};

#endif // TEXTURE_CUH