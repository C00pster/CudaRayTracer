#ifndef SPHERE_H
#define SPHERE_H

#include "primitives/primitive.h"
#include "math/utils.h"

class Sphere : public Primitive {
    public:
        __host__ __device__ 
        Sphere() {}

        // Stationary Sphere
        __host__ __device__ 
        Sphere(const Point3& static_center, float r, Material *m) : center(static_center, Vec4()), radius(fmaxf(r, 0)), mat_ptr(m) {
            Vec4 rvec = Vec4(radius, radius, radius);
            bbox = AABB(static_center - rvec, static_center + rvec);
        }

        // Moving Sphere
        __host__ __device__ 
        Sphere(const Point3& center1, const Point3& center2, float r, Material *m) : center(center1, center2 - center1), radius(r), mat_ptr(m) {
            Vec4 rvec = Vec4(radius, radius, radius);
            AABB box1(center.at(0) - rvec, center.at(0) + rvec);
            AABB box2(center.at(1) - rvec, center.at(1) + rvec);
            bbox = AABB(box1, box2);
        }

        __device__ 
        virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
            Point3 current_center = center.at(r.time());
            Vec4 oc = current_center - r.origin();
            float a = r.direction().squared_length();
            float b = dot(r.direction(), oc);
            float c = oc.squared_length() - radius*radius;
            float discriminant = b*b - a*c;

            if (discriminant < 0.0f) return false;

            float root = (b - sqrt(discriminant)) / a;

            if (!surrounds(root, t_min, t_max)) {
                root = (b + sqrt(discriminant)) / a;
                if (!surrounds(root, t_min, t_max)) return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            Vec4 outward_normal = (rec.p - current_center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;

            return true;
        }

        __device__ 
        virtual bool bounding_box(AABB& bounding_box) const override { 
            bounding_box = bbox; 
            return true; 
        }

        __device__
        virtual Point3 get_centroid() const override {
            return center.at(0.5f);
        }

        Ray center;
        float radius;
        Material *mat_ptr;
        AABB bbox;

        private:
            __device__
            static void get_sphere_uv(const Point3& p, float& u, float& v) {
                float theta = acos(-p.y());
                float phi = atan2(-p.z(), p.x()) + M_PI;

                u = phi / (2*M_PI);
                v = theta / M_PI;
            }
};

#endif // SPHERE_H