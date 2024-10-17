#include "aabb.cuh"

__device__ AABB surrounding_box(const AABB& box0, const AABB& box1) {
    return AABB(Point3(fminf(box0.minimum.x(), box1.minimum.x()),
                       fminf(box0.minimum.y(), box1.minimum.y()),
                       fminf(box0.minimum.z(), box1.minimum.z())),
                Point3(fmaxf(box0.maximum.x(), box1.maximum.x()),
                       fmaxf(box0.maximum.y(), box1.maximum.y()),
                       fmaxf(box0.maximum.z(), box1.maximum.z())));
}