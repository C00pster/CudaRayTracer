#ifndef AABB_CUH
#define AABB_CUH

#include "math/ray.cuh"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }

__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class AABB {
public:
    Point3 minimum;
    Point3 maximum;

    __device__ AABB() {}
    __device__ AABB(const Point3& a, const Point3& b) : minimum(a), maximum(b) {}

    __device__ Point3 min() const { return minimum; }
    __device__ Point3 max() const { return maximum; }

    __device__ bool hit(const Ray& r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min()[a] - r.origin()[a]) * invD;
            float t1 = (max()[a] - r.origin()[a]) * invD;
            float tt = t0;

            if (invD < 0.0f) {
                t0 = t1;
                t1 = tt;
            }

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            if (t_max <= t_min) {
                return false;
            }
        }
        return true;
    }
};

__device__ AABB surrounding_box(const AABB box0, AABB box1) {
    Point3 small(ffmin(box0.min().x(), box1.min().x()),
                    ffmin(box0.min().y(), box1.min().y()),
                    ffmin(box0.min().z(), box1.min().z()));
    Point3 big(ffmax(box0.max().x(), box1.max().x()),
                ffmax(box0.max().y(), box1.max().y()),
                ffmax(box0.max().z(), box1.max().z()));
    return AABB(small, big);
}

#endif // AABB_CUH