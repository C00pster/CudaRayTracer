#ifndef COLOR_CUH
#define COLOR_CUH

#include "math/vec4.cuh"
#include <iostream>
#include <fstream>

using Color = Vec4;

inline float linear_to_gamma(float value) {
    if (value > 0) {
        return sqrt(value);
    }
    return 0;
}

void write_framebuffer_to_file(
    Color *framebuffer, 
    int width, 
    int height, 
    const char *filename = "output.ppm"
);

#endif // COLOR_CUH