#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "stb/stb_image_write.hpp"
#include "device_algebra/algebra.hpp"


__global__ void trace(unsigned char* buffer, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    double r = double(i) / width;
    double g = double(j) / height;
    double b = 0.25;
    vec3d color = vec3d(r, g, b);

    int index = (j * width + i) * 3;
    buffer[index] = int(color.x() * 255);
    buffer[index + 1] = int(color.y() * 255);
    buffer[index + 2] = int(color.z() * 255);
}


__host__ void render_image(const char* path, int width, int height)
{
    int size = width * height * sizeof(unsigned char) * 3;
    unsigned char* data;
    cudaError_t error = cudaMallocManaged((void**) &data, size);
    if (error) {
        printf("[CUDA] can not allocate gpu memory: %d\n", int(error));
        cudaDeviceReset();
        exit(1);
    }

    dim3 blocks(width / 8, height / 8);
    dim3 threads(8, 8);

    printf("[CUDA] kernel launch...\n");

    trace<<<blocks, threads>>>(data, width, height);
    cudaDeviceSynchronize();

    printf("[CUDA] kernel done, writing data...\n");

    stbi_flip_vertically_on_write(true);
    stbi_write_png(path, width, height, 3, data, 0);
    printf("[INFO] write image done.\n");
    cudaFree(data);
}

__host__ int main(int argc, char* argv[])
{
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/cuda_output.png", 1920, 1080);
    return 0;
}
