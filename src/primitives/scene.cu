#include "scene.cuh"

__global__
void globe_device(Primitive **d_list, Primitive **d_sorted_list, DeviceImage* img, World **d_world, int64_t* morton_codes, int64_t* sorted_morton_codes,
                  Camera **d_camera, int width, int height, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int num_primitives = 1;

    ImageTexture* img_texture = new ImageTexture(img);

    d_list[0] = new Sphere(Vec4(0, 0, 0), 2.0f, new Lambertian(img_texture));

    *d_camera = new Camera(Vec4(13, 2, 3), Vec4(0, 0, 0), Vec4(0, 1, 0), 30.0f, float(width)/float(height), 0.1f, 10.0f);

    *d_world = new World(d_list, d_sorted_list, num_primitives, morton_codes, sorted_morton_codes);
}

void bouncing_spheres(int threads_per_block_x, int threads_per_block_y, int samples_per_pixel,
                      Color* framebuffer, World **d_world, Camera **d_camera) {

}

void globe(int threads_per_block_x, int threads_per_block_y, int samples_per_pixel,
           Color* framebuffer, World **d_world, Camera **d_camera, int width, int height,
           curandState *d_rand_state) {
    int num_primitives = 1;
    Primitive **d_list;
    Primitive **d_sorted_list;
    int64_t *morton_codes;
    int64_t *sorted_morton_codes;
    DeviceImage *d_image;

    checkCudaErrors(cudaMalloc((void **)&d_list, num_primitives * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&d_sorted_list, (num_primitives * 2 - 1) * sizeof(Primitive *)));
    checkCudaErrors(cudaMalloc((void **)&morton_codes, num_primitives * sizeof(int64_t)));
    checkCudaErrors(cudaMalloc((void **)&sorted_morton_codes, num_primitives * sizeof(int64_t)));
    checkCudaErrors(cudaMalloc((void **)&d_image, sizeof(DeviceImage)));

    RTWImage image("models/images/earthmap.jpg");
    unsigned char* d_data = image.get_device_data();
    int img_width = image.get_width();
    int img_height = image.get_height();
    int img_channels = image.get_channels();

    init_device_image<<<1, 1>>>(d_image, d_data, img_width, img_height, img_channels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    globe_device<<<1, 1>>>(d_list, d_sorted_list, d_image, d_world, morton_codes, 
                                                 sorted_morton_codes, d_camera, width, height, d_rand_state);

    std::cout << "World created\n";

    clock_t start, stop;
    start = clock();

    dim3 blocks((width + threads_per_block_x - 1) / threads_per_block_x,
            (height + threads_per_block_y - 1) / threads_per_block_y);
    dim3 threads(threads_per_block_x, threads_per_block_y);
    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(framebuffer, width, height, samples_per_pixel, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Time taken: " << timer_seconds << " seconds\n";

    write_framebuffer_to_file(framebuffer, width, height);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_sorted_list));
    checkCudaErrors(cudaFree(morton_codes));
    checkCudaErrors(cudaFree(sorted_morton_codes));
    checkCudaErrors(cudaFree(d_image));
}