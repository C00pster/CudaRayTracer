#include <iostream>
#include <string>
#include <curand_kernel.h>

#include "acceleration/bvh_node.cuh"
#include "camera/camera.cuh"
#include "primitives/primitive_list.cuh"
#include "float.h"
#include "primitives/sphere.cuh"
#include "primitives/rect.cuh"
#include "primitives/moving_sphere.cuh"
#include "materials/texture.cuh"
#include "primitives/box.cuh"
#include "primitives/quad.cuh"
#include "primitives/transform.cuh"
#include "primitives/constant_medium.cuh"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image.h"
#include "external/stb_image_write.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Vec4 color(const Ray& r, const Vec4& background, Primitive **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Vec4 cur_attenuation = Vec4(1.0, 1.0, 1.0);
    Vec4 cur_emitted = Vec4(0.0, 0.0, 0.0);
    for(int i = 0; i < 100; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec4 attenuation;
            Vec4 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_emitted += emitted * cur_attenuation;
                cur_ray = scattered;
            }
            else {
                return cur_emitted + emitted * cur_attenuation;
            }
        }
        else {
            return cur_emitted;
        }
    }
    return cur_emitted; // exceeded recursion
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_cornell_box(Primitive **elist, Primitive **eworld, Camera **camera, int nx, int ny, ImageTexture** texture, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        int i = 0;

        Color red = Color(0.65, 0.05, 0.05);
        Color white = Color(0.73, 0.73, 0.73);
        Color green = Color(0.12, 0.45, 0.15);
        Color light_color = Color(15, 15, 15);

        elist[i++] = new Quad(Point3(555,0,0), Vec4(0,555,0), Vec4(0,0,555), new Lambertian(new ConstantTexture(green)));
        elist[i++] = new Quad(Point3(0,0,0), Vec4(0,555,0), Vec4(0,0,555), new Lambertian(new ConstantTexture(red)));
        elist[i++] = new Quad(Point3(343, 554, 332), Vec4(-130,0,0), Vec4(0,0,-105), new DiffuseLight(new ConstantTexture(light_color)));
        elist[i++] = new Quad(Point3(0,0,0), Vec4(555,0,0), Vec4(0,0,555), new Lambertian(new ConstantTexture(white)));
        elist[i++] = new Quad(Point3(555,555,555), Vec4(-555,0,0), Vec4(0,0,-555), new Lambertian(new ConstantTexture(white)));
        elist[i++] = new Quad(Point3(0,0,555), Vec4(555,0,0), Vec4(0,555,0), new Lambertian(new ConstantTexture(white)));

        Primitive* box1 = box(Point3(0,0,0), Point3(165,330,165), new Lambertian(new ConstantTexture(white)));
        elist[i++] = new Translate(new RotateY(box1, 15), Vec4(265,0,295));

        Primitive* box2 = box(Point3(0,0,0), Point3(165,165,165), new Lambertian(new ConstantTexture(white)));
        elist[i++] = new Translate(new RotateY(box2, -18), Vec4(130,0,65));

        *eworld = new PrimitiveList(elist, i);

        Vec4 lookfrom(278, 278, -800);
        Vec4 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        *camera = new Camera(
            lookfrom,
            lookat,
            Vec4(0,1,0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0,
            1.0
        );
    }
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int maxx, int maxy, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxx) || (j >= maxy)) return;
    int pixel_index = j*maxx + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void texture_init(unsigned char* tex_data, int nx, int ny, ImageTexture** tex) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *tex = new ImageTexture(tex_data, nx, ny);
    }
}

__global__ void render(Vec4* fb, int max_x, int max_y, int ns, Camera **cam, Primitive **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = randState[pixel_index];
    Vec4 col(0,0,0);
    Vec4 background(0, 0, 0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, background, world, &local_rand_state);
    }
    randState[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

int main(int argc, char* argv[]) {
    // if (argc < 5) {
    //     std::cerr << "Usage: " << argv[0] << " [WIDTH] [HEIGHT] [BOUNCES] [OUTPUT FILENAME]" << std::endl;
    // }
    // int nx = std::stoi(std::string(argv[1]));
    // int ny = std::stoi(std::string(argv[2]));
    // int ns = std::stoi(std::string(argv[3]));
    int nx = 1080;
    int ny = 720;
    int ns = 500;
    int tx = 32;
    int ty = 32;

    size_t stackSize = 8192;
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    if (err != cudaSuccess) {
        std::cerr << "Error setting stack size: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Values
    int num_pixels = nx * ny;

    int tex_x, tex_y, tex_n;
    unsigned char *tex_data_host = stbi_load("models/images/earthmap.jpg", &tex_x, &tex_y, &tex_n, 0);

    unsigned char *tex_data;
    checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    ImageTexture **texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(ImageTexture*)));
    texture_init<<<1, 1>>>(tex_data, tex_x, tex_y, texture);

    // Allocating CUDA memory
    Vec4* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, nx * ny * sizeof(Vec4)));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // Allocate 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Building the world
    Primitive **elist;
    int num_entity = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&elist, num_entity * sizeof(Primitive*)));
    Primitive **eworld;
    checkCudaErrors(cudaMalloc((void **)&eworld, sizeof(Primitive*)));
    Camera **camera;
    checkCudaErrors(cudaMalloc((void **)&camera, sizeof(Camera*)));
    create_cornell_box<<<1, 1>>>(elist, eworld, camera, nx, ny, texture, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Rendering..." << std::endl;
    clock_t start, stop;
    start = clock();

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(image, nx, ny, ns, camera, eworld, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    std::cout << "Rendering Complete. Time taken: " << (double)(stop - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    uint8_t* imageHost = new uint8_t[nx * ny * 3 * sizeof(uint8_t)];
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            imageHost[(ny - j - 1) * nx * 3 + i * 3] = 255.99 * image[pixel_index].r();
            imageHost[(ny - j - 1) * nx * 3 + i * 3 + 1] = 255.99 * image[pixel_index].g();
            imageHost[(ny - j - 1) * nx * 3 + i * 3 + 2] = 255.99 * image[pixel_index].b();
        }
    }
    stbi_write_png("output.png", nx, ny, 3, imageHost, nx * 3);

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(eworld));
    checkCudaErrors(cudaFree(elist));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(image));
}