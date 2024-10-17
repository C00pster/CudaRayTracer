#include "color.cuh"

void write_framebuffer_to_file(
    Color *framebuffer, 
    int width, 
    int height, 
    const char *filename
) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            unsigned char ir = static_cast<unsigned char>(255.99 * framebuffer[pixel_index].r());
            unsigned char ig = static_cast<unsigned char>(255.99 * framebuffer[pixel_index].g());
            unsigned char ib = static_cast<unsigned char>(255.99 * framebuffer[pixel_index].b());
            file << static_cast<int>(ir) << " " << static_cast<int>(ig) << " " << static_cast<int>(ib) << "\n";
        }
    }

    file.close();
}