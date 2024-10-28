#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "math/ray.cuh"

__device__ Vec4 random_in_unit_disk(curandState *local_rand_state) {
    Vec4 p;
    do {
        p = 2.0f * Vec4(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - Vec4(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class Camera {
public:
    __device__ Camera(Vec4 lookfrom, Vec4 lookat, Vec4 vup, float vfov, float aspect, float aperture, float focus_dist, float t0 = 0.0f, float t1 = 0.0f) {
        time0 = t0;
        time1 = t1;
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }

    __device__ Ray get_ray(float s, float t, curandState *local_rand_state) {
        Vec4 rd = lens_radius*random_in_unit_disk(local_rand_state);
        Vec4 offset = u * rd.x() + v * rd.y();
        float ray_time = time0 + curand_uniform(local_rand_state) * (time1 - time0);

        return Ray(
            origin + offset, 
            lower_left_corner + s*horizontal + t*vertical - origin - offset, 
            ray_time); 
    }

    Vec4 origin;
    Vec4 lower_left_corner;
    Vec4 horizontal;
    Vec4 vertical;
    Vec4 u, v, w;
    float lens_radius;
    float time0, time1;
};

#endif // CAMERA_CUH