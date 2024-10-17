#ifndef UTILS_CUH
#define UTILS_CUH

__device__
inline bool surrounds(const float val, const float min, const float max) {
    return min <= val && val <= max;
}

__device__
inline float clamp(const float value, const float min, const float max) {
    return value < min ? min : (value > max ? max : value);
}

__device__
inline bool overlaps(const float min1, const float max1, const float min2, const float max2) {
    return min1 <= max2 && min2 <= max1;
}

#endif // UTILS_CUH