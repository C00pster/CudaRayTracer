#ifndef VEC4_H
#define VEC4_H

#include "utils/constants.h"

// __host__ - Flag to indicate that the function is called from the host
// __device__ - Flag to indicate that the function is called from the device
class Vec4 {
    public:
        // float4 is a CUDA vector type
        float4 e;

        //Constructors
        __host__ __device__ Vec4() { e = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
        __host__ __device__ Vec4(float x, float y, float z, float w) { e = make_float4(x, y, z, w); }
        __host__ __device__ Vec4(float x, float y, float z) { e = make_float4(x, y, z, 1.0f); }
        __host__ __device__ Vec4(const float4 &v) : e(v) {}

        //Accessors
        __host__ __device__ inline float x() const { return e.x; }
        __host__ __device__ inline float y() const { return e.y; }
        __host__ __device__ inline float z() const { return e.z; }
        __host__ __device__ inline float w() const { return e.w; }
        __host__ __device__ inline float r() const { return e.x; }
        __host__ __device__ inline float g() const { return e.y; }
        __host__ __device__ inline float b() const { return e.z; }

        //Operators
        __host__ __device__ inline const Vec4& operator+() const { return *this; }
        __host__ __device__ inline Vec4 operator-() const { return Vec4(-e.x, -e.y, -e.z, e.w); } //w stays unchanged

        __host__ __device__ inline Vec4& operator+=(const Vec4 &v2);
        __host__ __device__ inline Vec4& operator-=(const Vec4 &v2);
        __host__ __device__ inline Vec4& operator*=(const Vec4 &v2);
        __host__ __device__ inline Vec4& operator/=(const Vec4 &v2);
        __host__ __device__ inline Vec4& operator*=(const float t);
        __host__ __device__ inline Vec4& operator/=(const float t);

        // Magnitude
        __host__ __device__ inline float length() const { return sqrtf(e.x*e.x + e.y*e.y + e.z*e.z); }
        __host__ __device__ inline float squared_length() const { return e.x*e.x + e.y*e.y + e.z*e.z; }

        // Normalize
        __host__ __device__ inline void make_unit_vector();
        __device__ Vec4& apply_sqrt() {
            e.x = sqrt(e.x);
            e.y = sqrt(e.y);
            e.z = sqrt(e.z);
            return *this;
        }
};

using Point3 = Vec4; // Alias for 3D point

// Stream handling
inline std::istream& operator>>(std::istream &is, Vec4 &t) {
    is >> t.e.x >> t.e.y >> t.e.z;
    return is;
}

inline std::ostream& operator<<(std::ostream &out, const Vec4 &v) {
    return out << v.e.x << ' ' << v.e.y << ' ' << v.e.z;
}

__host__ __device__ inline void Vec4::make_unit_vector() {
    float k = rsqrtf(e.x*e.x + e.y*e.y + e.z*e.z);
    e.x *= k; e.y *= k; e.z *= k;
}

// Overload operators
__host__ __device__ inline Vec4 operator+(const Vec4 &v1, const Vec4 &v2) {
    return Vec4(
        make_float4(v1.e.x + v2.e.x,
                    v1.e.y + v2.e.y, 
                    v1.e.z + v2.e.z, 
                    v1.e.w)
    );
}
__host__ __device__ inline Vec4 operator-(const Vec4 &v1, const Vec4 &v2) {
    return Vec4(
        make_float4(v1.e.x - v2.e.x,
                    v1.e.y - v2.e.y, 
                    v1.e.z - v2.e.z, 
                    v1.e.w)  // w remains unchanged
    );
}
__host__ __device__ inline Vec4 operator*(const Vec4 &v1, const Vec4 &v2) {
    return Vec4(
        make_float4(v1.e.x * v2.e.x,
                    v1.e.y * v2.e.y, 
                    v1.e.z * v2.e.z, 
                    v1.e.w)  // w remains unchanged
    );
}
__host__ __device__ inline Vec4 operator/(const Vec4 &v1, const Vec4 &v2) {
    return Vec4(
        make_float4(v1.e.x / v2.e.x,
                    v1.e.y / v2.e.y, 
                    v1.e.z / v2.e.z, 
                    v1.e.w)  // w remains unchanged
    );
}
__host__ __device__ inline Vec4 operator*(float t, const Vec4 &v) {
    return Vec4(
        make_float4(t * v.e.x,
                    t * v.e.y, 
                    t * v.e.z, 
                    v.e.w)  // w remains unchanged
    );
}
__host__ __device__ inline Vec4 operator/(Vec4 v, float t) {
    return Vec4(
        make_float4(v.e.x / t,
                    v.e.y / t, 
                    v.e.z / t, 
                    v.e.w)  // w remains unchanged
    );
}
__host__ __device__ inline Vec4 operator*(const Vec4 &v, float t) {
    return Vec4(
        make_float4(t * v.e.x,
                    t * v.e.y, 
                    t * v.e.z, 
                    v.e.w)  // w remains unchanged
    );
}

__host__ __device__ inline Vec4& Vec4::operator+=(const Vec4 &v) {
    e.x += v.e.x;
    e.y += v.e.y;
    e.z += v.e.z;
    return *this;
}
__host__ __device__ inline Vec4& Vec4::operator-=(const Vec4 &v) {
    e.x -= v.e.x;
    e.y -= v.e.y;
    e.z -= v.e.z;
    return *this;
}
__host__ __device__ inline Vec4& Vec4::operator*=(const Vec4 &v) {
    e.x *= v.e.x;
    e.y *= v.e.y;
    e.z *= v.e.z;
    return *this;
}
__host__ __device__ inline Vec4& Vec4::operator/=(const Vec4 &v) {
    e.x /= v.e.x;
    e.y /= v.e.y;
    e.z /= v.e.z;
    return *this;
}

// Scalar multiplication and division
__host__ __device__ inline Vec4& Vec4::operator*=(const float t) {
    e.x *= t;
    e.y *= t;
    e.z *= t;
    return *this;
}
__host__ __device__ inline Vec4& Vec4::operator/=(const float t) {
    float k = 1.0/t;
    e.x *= k;
    e.y *= k;
    e.z *= k;
    return *this;
}

// Linear algebra operations
__host__ __device__ inline float dot(const Vec4 &v1, const Vec4 &v2) {
    return v1.e.x * v2.e.x + v1.e.y * v2.e.y + v1.e.z * v2.e.z;
}
__host__ __device__ inline Vec4 cross(const Vec4 &v1, const Vec4 &v2) {
    return Vec4(make_float4((v1.e.y * v2.e.z - v1.e.z * v2.e.y),
                (-(v1.e.x * v2.e.z - v1.e.z * v2.e.x)),
                (v1.e.x * v2.e.y - v1.e.y * v2.e.x),
                v1.e.w));
}

__host__ __device__ inline Vec4 unit_vector(Vec4 v) {
    float length = sqrtf(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    return Vec4(v.x() / length, v.y() / length, v.z() / length, v.w());
}

#endif // VEC4_H