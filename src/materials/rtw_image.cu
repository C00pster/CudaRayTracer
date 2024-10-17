#include "rtw_image.cuh"

#ifndef _MSC_VER
    #pragma warning (push, 0)
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#define STB_IMAGE_IMPLEMENTATION
// #define STBI_FAILURE_USERMSG
#include "external/stb_image.h"

#pragma GCC diagnostic pop

__global__
void init_device_image(DeviceImage* d_img, unsigned char* data, int width, int height, int channels) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    d_img->data = data;
    d_img->width = width;
    d_img->height = height;
    d_img->channels = channels;
}

#ifndef _MSC_VER
    #pragma warning (pop)
#endif