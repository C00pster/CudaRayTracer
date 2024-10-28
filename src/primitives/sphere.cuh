#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "primitives/primitive.cuh"
#include "materials/material.cuh"
#include "math/utils.cuh"

class Sphere : public Primitive {
public:
    __device__ Sphere() {}
    __device__ Sphere(const Point3& c, float r, Material *m) : center(c), radius(fmaxf(r, 0)), mat_ptr(m) {};

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec4 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }
        return false;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& bounding_box) const {
        bounding_box = AABB(center - Vec4(radius, radius, radius, 0), center + Vec4(radius, radius, radius, 0));
        return true;
    }

    __device__ void get_sphere_uv(const Point3& p, float& u, float& v) const {
        float theta = asin(p.y());
        float phi = atan2(p.z(), p.x());
        u = 1-(phi + M_PI) / (2 * M_PI);
        v = (theta + M_PI / 2) / M_PI;
    }

    Point3 center;
    float radius;
    Material *mat_ptr;
};

#endif // SPHERE_CUH