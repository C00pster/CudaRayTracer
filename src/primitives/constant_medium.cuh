#ifndef CONSTAT_MEDIUM_CUH
#define CONSTANT_MEDIUM_CUH

#include "primitive.cuh"
#include "materials/material.cuh"
#include "materials/texture.cuh"

class ConstantMedium : public Primitive {
public:
    __device__ ConstantMedium(Primitive* b, float f, Texture* a, curandState* local_rand_state) : boundary(b), neg_inv_density(-1.0f / f), rand_state(local_rand_state) {
        phase_function = new Isotropic(a);
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord rec1, rec2;

        if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1)) return false;

        if(!boundary->hit(r, rec1.t + 0.0001f, FLT_MAX, rec2)) return false;

        if (rec1.t < t_min) rec1.t = t_min;
        if (rec2.t > t_max) rec2.t = t_max;

        if (rec1.t >= rec2.t) return false;
        if (rec1.t < 0) rec1.t = 0;

        const float ray_length = r.direction().length();
        const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        const float hit_distance = neg_inv_density * log(curand_uniform(rand_state));

        if (hit_distance > distance_inside_boundary) return false;
        
        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);
        rec.normal = Vec4(1, 0, 0);
        rec.front_face = true;
        rec.mat_ptr = phase_function;

        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        return boundary->bounding_box(t0, t1, output_box);
    }

    Primitive* boundary;
    Material* phase_function;
    curandState* rand_state;
    float neg_inv_density;
};

#endif // CONSTANT_MEDIUM_CUH