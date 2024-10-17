#ifndef RTW_STB_IMAGE_CUH
#define RTW_STB_IMAGE_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "external/stb_image.h"

struct DeviceImage {
    unsigned char*  data;
    int             width;
    int             height;
    int             channels;
};

__global__
void init_device_image(DeviceImage* d_img, unsigned char* data, int width, int height, int channels);

class RTWImage {
    public:
        __host__
        RTWImage(const char* image_filename) {
            data = stbi_load(image_filename, &width, &height, &channels, 0);
            if (!data) {
                std::cerr << "Error: Could not load image: " << image_filename << std::endl;
                width = height = channels = 0;
                device_data = nullptr;
                return;
            }

            size_t image_size = width * height * channels * sizeof(unsigned char);

            cudaError_t err = cudaMalloc(&device_data, image_size);
            if (err != cudaSuccess) {
                std::cerr << "Error: Could not allocate device memory for image: " << image_filename << std::endl;
                stbi_image_free(data);
                width = height = channels = 0;
                device_data = nullptr;
                return;
            }

            err = cudaMemcpy(device_data, data, image_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Error: Could not copy image data to device for image: " << image_filename << std::endl;
                cudaFree(device_data);
                stbi_image_free(data);
                width = height = channels = 0;
                device_data = nullptr;
                return;
            }
        }

        ~RTWImage() {
            if (data) {
                stbi_image_free(data);
                data = nullptr;
            }
            if (device_data) {
                cudaFree(device_data);
                device_data = nullptr;
            }
        }

        unsigned char* get_device_data() const { return device_data; }
        int get_width() const { return width; }
        int get_height() const { return height; }
        int get_channels() const { return channels; }

    private:
        unsigned char*  data = nullptr;
        unsigned char*  device_data = nullptr;
        int             width = 0;
        int             height = 0;
        int             channels = 0;
};

#endif // RTW_STB_IMAGE_CUH