#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "math/vec4.h"
#include "rtw_image.h"
#include "math/interval.h"

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

class ImageTexture : public Texture {
    public:
        DeviceImage img;

        __device__
        ImageTexture(const DeviceImage& image) {
            img = image;
        }

        __device__
        Color value(float u, float v, const Point3& p) const override {
            if (img.data == nullptr) {
                return Color(0, 1, 1);
            }

            u = fmodf(u, 1.0f);
            v = fmodf(v, 1.0f);
            if (u < 0.0f) u += 1.0f;
            if (v < 0.0f) v += 1.0f;

            int i = static_cast<int>(u * img.width);
            int j = static_cast<int>((1.0f - v) * (img.height - 0.001f));

            i = i < 0 ? 0 : (i >= img.width ? img.width - 1 : i);
            j = j < 0 ? 0 : (j >= img.height ? img.height - 1 : j);

            int idx = (j * img.width + i) * img.channels;

            int max_idx = img.width * img.height * img.channels;
            if (idx < 0 || idx >= max_idx) {
                return Color(0, 1, 1);
            }

            float r = static_cast<float>(img.data[idx]) / 255.0f;
            float g = static_cast<float>(img.data[idx + 1]) / 255.0f;
            float b = static_cast<float>(img.data[idx + 2]) / 255.0f;

            return Color(r, g, b);
        }
};

#endif // TEXTURE_CUH