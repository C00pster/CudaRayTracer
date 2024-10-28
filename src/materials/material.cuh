#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct HitRecord;

#include "primitives/primitive.cuh"
#include "math/ray.cuh"
#include "math/color.cuh"
#include "texture.cuh"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

class Material {
    public:
        __device__ virtual Color emitted(float u, float v, const Vec4 &p) const {
            return Color(0.0f, 0.0f, 0.0f);
        }
        __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered, curandState *local_rand_state) const = 0;
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
            Vec4 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
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
            
            reflect(unit_vector(r_in.direction()), rec.normal);
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

class DiffuseLight : public Material {
    public:
        __device__ DiffuseLight(Texture* e) : emit(e) {}

        __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec, Vec4 &attenuation, Ray &scattered, curandState *local_rand_state) const {
            return false; // No scattering for light material
        }

        __device__ Vec4 emitted(float u, float v, const Vec4 &p) const {
            return emit->value(u, v, p);
        }

        Texture* emit;
};

class Isotropic : public Material {
public:
    __device__ Isotropic(Texture* a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec4& attenuation, Ray& scattered, curandState* local_rand_state) const {
        scattered = Ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    Texture* albedo;
};

class Transparent : public Material {
public:
    __device__ Transparent(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec4& attenuation, Ray& scattered, curandState* local_rand_state) const {
        Vec4 outward_normal;
        Vec4 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = Color(1.0f, 1.0f, 1.0f);
        Vec4 refracted;

        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrtf(1.0f - ref_idx * ref_idx * (1.0f - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0;

        if (curand_uniform(local_rand_state) < reflect_prob) {
            scattered = Ray(rec.p, reflected, r_in.time());
        }
        else {
            scattered = Ray(rec.p, refracted, r_in.time());
        }

        return true;
    }

    float ref_idx;
};

#endif // MATERIAL_CUH