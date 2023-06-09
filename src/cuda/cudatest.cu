#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>


#define cudaErrorCheck(error)                         \
do {                                                  \
    const cudaError_t error_code = error;             \
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


class person {

public:
    const char* name;
    int age;
    __host__ virtual void copy_to_device() = 0;
    __device__ virtual void act() = 0;
};


class cxk : public person{

public:
    const char* name = "caixukun";
    int age = 25;

    char* d_name;
    int* d_age;

    __host__ __device__ cxk() {}

    __host__ __device__ cxk(const cxk* c) 
    {
        d_name = c->d_name;
        d_age = c->d_age;
    }

    __host__ void copy_to_device()
    {
        cudaMalloc((void**) &d_name, strlen(name) * sizeof(char));
        cudaMemcpy(d_name, name, strlen(name) * sizeof(char), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &d_age, sizeof(int));
        cudaMemcpy(d_age, &age, sizeof(int), cudaMemcpyHostToDevice);
    }

    __device__ virtual void act() override
    {
        printf("name is: %s.\n", d_name);
        printf("age is %d.\n", *d_age);
        printf("jump rap basketball.\n");
    }
};


class yuqian {

public:
    const char* name = "yuqian";
    int age = 40;

    __device__ void act()
    {
        printf("name is: %s.\n", name);
        printf("age is %d.\n", age);
        printf("smoking drinking fire head\n");
    }
};


__global__ void kernel(cxk* object)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    printf("[device] thread %d running:\n", index);
    cxk* d_object = new cxk(object);
    d_object->act();
    printf("[device] thread %d done.\n", index);
}


int main(int argc, char** argv)
{
    cxk* caixukun = new cxk();
    caixukun->copy_to_device();
    cxk* device_caixukun;

    cudaErrorCheck(cudaMalloc((void**) &device_caixukun, sizeof(cxk)));
    cudaErrorCheck(cudaMemcpy(device_caixukun, caixukun, sizeof(cxk), cudaMemcpyHostToDevice));

    printf("kernel launch.\n");

    kernel<<<3, 1>>>(device_caixukun);
    cudaErrorCheck(cudaDeviceSynchronize());

    printf("kernel done.\n");

    return 0;
}