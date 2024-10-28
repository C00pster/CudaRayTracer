#ifndef RAY_CUH
#define RAY_CUH

#include "vec4.cuh"

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const Point3 &o, const Vec4 &d, float time = 0.0f) : orig(o), dir(d), tm(time) {}
    __device__ Point3 origin() const { return orig; }
    __device__ Vec4 direction() const { return dir; }
    __device__ float time() const { return tm; }
    __device__ Point3 at(float t) const { return orig + t * dir; }

private:
    Point3 orig;
    Vec4 dir;
    float tm;
};

#endif // RAY_CUH