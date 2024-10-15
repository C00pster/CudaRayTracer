#ifndef MATERIAL_H
#define MATERIAL_H

struct HitRecord;

#include "math/ray.h"
#include "primitives/primitive.h"
#include "texture.cuh"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ bool refract(const Vec4 &v, const Vec4 &n, float ni_over_nt, Vec4 &refracted) {
    Vec4 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else {
        return false;
    }
} 

#define RANDVEC4 Vec4(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vec4 random_in_unit_sphere(curandState *local_rand_state) {
    Vec4 p;
    do {
        p = 2.0f * RANDVEC4 - Vec4(1.0f, 1.0f, 1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ bool reflect(const Vec4 &v, const Vec4 &n, Vec4 &reflected) {
    reflected = v - 2 * dot(v, n) * n;
    return true;
}

class Material {
    public:
        __device__ 
        virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
    public:
        Texture* albedo;

        __device__ 
        Lambertian(Texture* a) : albedo(a) {}

        __device__ 
        virtual bool scatter(const Ray &r_in, const HitRecord &rec, Vec4 &attenuation, Ray &scattered, curandState *local_rand_state) const {
            Vec4 scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);

            if (scatter_direction.near_zero()) scatter_direction = rec.normal;

            scattered = Ray(rec.p, scatter_direction, r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }
};

class Metal : public Material {
    public:
        Vec4 albedo;
        float fuzz;

        __device__ 
        Metal(const Vec4 &a, float f) : albedo(a) {
            if (f < 1) fuzz = f;
            else fuzz = 1;
        }

        __device__ 
        virtual bool scatter(const Ray &r_in, const HitRecord &rec, Vec4 &attenuation, Ray &scattered, curandState *local_rand_state) const {
            Vec4 reflected;
            reflect(unit_vector(r_in.direction()), rec.normal, reflected);
            reflected += (fuzz * random_in_unit_sphere(local_rand_state));
            scattered = Ray(rec.p, reflected, r_in.time());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
};

class Dielectric : public Material {
    public:
        float ref_idx;

        __host__ __device__ 
        Dielectric(float ri) : ref_idx(ri) {}

        __device__ 
        virtual bool scatter(const Ray &r_in, const HitRecord &rec, Vec4 &attenuation, Ray &scattered, curandState *local_rand_state) const {
            Vec4 outward_normal;
            Vec4 reflected;
            
            reflect(unit_vector(r_in.direction()), rec.normal, reflected);
            float ni_over_nt;
            attenuation = Color(1.0f, 1.0f, 1.0f); // No attenuation for dielectric
            Vec4 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                cosine = sqrtf(1.0f - ref_idx * ref_idx * (1.0f - cosine * cosine));
            } else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }

            if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
                reflect_prob = schlick(cosine, ref_idx);
            else reflect_prob = 1.0;

            if (curand_uniform(local_rand_state) < reflect_prob) {
                scattered = Ray(rec.p, reflected, r_in.time());
            } else {
                scattered = Ray(rec.p, refracted, r_in.time());
            }

            return true;
        }
};

#endif // MATERIAL_H