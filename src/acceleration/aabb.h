#ifndef AABB_H
#define AABB_H

#include "math/ray.h"

class AABB {
    public:
        Point3 minimum;
        Point3 maximum;

        __device__ AABB() {}
        __device__ AABB(const Point3& a, const Point3& b) : minimum(a), maximum(b) {}
        __device__ AABB(const AABB& box1, const AABB& box2) {
            minimum = Point3(fminf(box1.minimum.x(), box2.minimum.x()),
                             fminf(box1.minimum.y(), box2.minimum.y()),
                             fminf(box1.minimum.z(), box2.minimum.z()));
            maximum = Point3(fmaxf(box1.maximum.x(), box2.maximum.x()),
                             fmaxf(box1.maximum.y(), box2.maximum.y()),
                             fmaxf(box1.maximum.z(), box2.maximum.z()));
        }

        __device__ void axis_interval(int n, float& min, float& max) const {
            min = minimum[n];
            max = maximum[n];
        }

        __device__ bool hit(const Ray& r, float t_min, float t_max) const {
            const Point3& origin = r.origin();
            const Vec4& direction = r.direction();

            for (int a = 0; a < 3; a++) {
                float min, max;
                axis_interval(a, min, max);
                const double adinv = 1.0 / direction[a];

                float t0 = (min - origin[a]) * adinv;
                float t1 = (max - origin[a]) * adinv;

                if (t0 < t1) {
                    if (t0 > t_min) t_min = t0;
                    if (t1 < t_max) t_max = t1;
                } else {
                    if (t1 > t_min) t_min = t1;
                    if (t0 < t_max) t_max = t0;
                }

                if (t_max <= t_min) return false;
            }
            return true;
        }
};

__device__ AABB surrounding_box(const AABB& box0, const AABB& box1) {
    return AABB(Point3(fminf(box0.minimum.x(), box1.minimum.x()),
                       fminf(box0.minimum.y(), box1.minimum.y()),
                       fminf(box0.minimum.z(), box1.minimum.z())),
                Point3(fmaxf(box0.maximum.x(), box1.maximum.x()),
                       fmaxf(box0.maximum.y(), box1.maximum.y()),
                       fmaxf(box0.maximum.z(), box1.maximum.z())));
}

#endif // AABB_H