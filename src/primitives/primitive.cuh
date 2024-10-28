#ifndef PRIMITIVE_CUH
#define PRIMITIVE_CUH

#include "math/ray.cuh"
#include "acceleration/aabb.cuh"

class Material;

struct HitRecord {
    float t;
    float u;
    float v;
    Point3 p;
    Vec4 normal;
    Material *mat_ptr;
    bool front_face;

    __device__ inline void set_face_normal(const Ray &r, const Vec4 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Primitive {
    public:
        __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
        __device__ virtual bool bounding_box(float t0, float t1, AABB& bbox) const = 0;
};

#endif // PRIMITIVE_CUH