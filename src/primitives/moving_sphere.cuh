#ifndef MOVING_SPHERE_CUH
#define MOVING_SPHERE_CUH

#include "acceleration/aabb.cuh"
#include "primitive.cuh"

class MovingSphere : public Primitive {
public:
    __device__ MovingSphere() {}
    __device__ MovingSphere(Point3 cen0, Point3 cen1, float t0, float t1, float r, Material *m)
        : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};

    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Point3 oc = r.origin() - center(r.time());
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center(r.time())) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center(r.time())) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }
        return false;
    }

    __device__ bool bounding_box(float t0, float t1, AABB& output_box) const {
        AABB box0(center0 - Vec4(radius, radius, radius), center0 + Vec4(radius, radius, radius));
        AABB box1(center1 - Vec4(radius, radius, radius), center1 + Vec4(radius, radius, radius));
        output_box = surrounding_box(box0, box1);
        return true;
    }

    __device__ Point3 center(float time) const {
        return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
    }

    Point3 center0, center1;
    float time0, time1;
    float radius;
    Material* mat_ptr;
};

#endif // MOVING_SPHERE_CUH