#include "material.cuh"

__device__
float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__
bool refract(const Vec4 &v, const Vec4 &n, float ni_over_nt, Vec4 &refracted) {
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

__device__
Vec4 random_in_unit_sphere(curandState *local_rand_state) {
    Vec4 p;
    do {
        p = 2.0f * RANDVEC4 - Vec4(1.0f, 1.0f, 1.0f);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__
bool reflect(const Vec4 &v, const Vec4 &n, Vec4 &reflected) {
    reflected = v - 2 * dot(v, n) * n;
    return true;
}