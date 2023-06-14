#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "ray.cuh"
#include "pdf.cuh"
#include "demo.cuh"
#include "group.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "intersection.cuh"

#include "stb/stb_image_write.hpp"
#include "device_algebra/algebra.hpp"


#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


__device__ vec3d trace(group* scn, const ray& r, int depth, curandState* state, object* lights)
{
    vec3d accumulated_radiance(0, 0, 0);
    vec3d throughput(1, 1, 1);
    ray current_ray = r;

    for (int i = 0; i < depth; i++)
    {
        intersection crossover;
        if (!scn->intersect(current_ray, crossover)) {
            break;
        }

        vec3d emission = crossover.mat->emit(current_ray, crossover, crossover.position, crossover.uv_coord);
        accumulated_radiance += throughput * emission;

        scatter scattering;
        if (!crossover.mat->shading(current_ray, crossover, scattering, state)) {
            break;
        }

        if (scattering.is_spec) {
            // accumulated_radiance += throughput * scattering.attenuation * trace(scn, scattering.specular, depth - i - 1);
            break;
        }

        ray next_ray;
        double pdf_value;

        if (true) {
            vec3d pdf_generate;
            if (random_double(state) < 0.5) {
                pdf_generate = lights->rand(crossover.position, state);
            }
            else {
                pdf_generate = cos_pdf_generate(crossover.normal, state);
            }
            next_ray = ray(crossover.position, pdf_generate, current_ray.time, current_ray.t_min, current_ray.t_max);
            pdf_value = 0.5 * (cos_pdf_value(crossover.normal, next_ray.direction) + lights->pdf(crossover.position, next_ray.direction));
        }
        else {
            next_ray = ray(crossover.position, cos_pdf_generate(crossover.normal, state), current_ray.time, current_ray.t_min, current_ray.t_max);
            pdf_value = cos_pdf_value(crossover.normal, next_ray.direction);
        }

        if (scattering.pdf_ptr != nullptr) {
            delete scattering.pdf_ptr;
        }

        throughput = throughput * scattering.attenuation * crossover.mat->shading_pdf(current_ray, crossover, next_ray) / pdf_value;
        current_ray = next_ray;
    }

    return accumulated_radiance;
}


__global__ void render(unsigned char* buffer, group** scn, object** lgt, camera** cam, int width, int height, int spp, curandState* state)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= width || i >= height) return;

    int r_idx = i * width + j;
    int index = (i * width + j) * 3;

    // printf("device: %d %d %d\n", j, i, index);
    vec3d color = vec3d(0, 0, 0);
    for (int k = 0; k < spp; k++) {
        double x = double(j + random_double(&state[r_idx])) / (width - 1);
        double y = double(i + random_double(&state[r_idx])) / (height - 1);

        ray r = (*cam)->emit(x, y, &state[r_idx]);
        vec3d sample = trace(*scn, r, 10, &state[r_idx], *lgt);

        if (is_nan(sample) || is_infinity(sample)) {
            sample = vec3d(0, 0, 0);
        }

        color += sample;
    }

    color /= spp;
    color = gamma(color, 0.45);
    color = clamp(color, 0, 1);
    
    buffer[index] = int(color.x() * 255);
    buffer[index + 1] = int(color.y() * 255);
    buffer[index + 2] = int(color.z() * 255);
}


__global__ void create_scene(group** scn, object** lgt, camera** cam)
{
    *scn = cornell_box();
    *cam = new camera(vec3d(278, 278, -800), vec3d(278, 278, 0), vec3d(0, 1, 0), 40, 1, 0, 1, 0, 1);
    *lgt = new flip(new planexz(213, 343, 227, 332, 554, new emissive(vec3d(30, 30, 30))));
}


__global__ void random_init(int width, int height, curandState *state) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= width || y >= height) return;

    int index = y * width + x;
    curand_init(0, index, 0, &state[index]);
}


__host__ void render_image(const char* path, int width, int height, int spp)
{
    int size = width * height * sizeof(unsigned char) * 3;
    unsigned char* data;
    cudaError_t error = cudaMallocManaged((void**) &data, size);
    if (error) {
        printf("[CUDA] can not allocate gpu memory: %d\n", int(error));
        cudaDeviceReset();
        exit(1);
    }

    group** scn;
    object** lgt;
    camera** cam;
    
    cudaMalloc((void**) &scn, sizeof(group*));
    cudaMalloc((void**) &lgt, sizeof(object*));
    cudaMalloc((void**) &cam, sizeof(camera*));

    dim3 blocks(width / 8 + 1, height / 8 + 1);
    dim3 threads(8, 8);

    curandState* state;
    cudaMallocManaged((void**) &state, width * height * sizeof(curandState));

    random_init<<<blocks, threads>>>(width, height, state);
    CHECK(cudaDeviceSynchronize());

    create_scene<<<1, 1>>>(scn, lgt, cam);
    CHECK(cudaDeviceSynchronize());

    printf("[INFO] start render...\n");
    clock_t start = clock();

    printf("[CUDA] kernel launch...\n");

    render<<<blocks, threads>>>(data, scn, lgt, cam, width, height, spp, state);
    CHECK(cudaDeviceSynchronize());

    printf("[CUDA] kernel done, using %f sec.\n", float(clock() - start) / CLOCKS_PER_SEC);

    stbi_flip_vertically_on_write(true);
    stbi_write_png(path, width, height, 3, data, 0);
    printf("[INFO] write image done.\n");
    cudaFree(data);
}

__host__ int main(int argc, char* argv[])
{
    render_image("./cuda_output.png", 1024, 1024, 100);
    return 0;
}
