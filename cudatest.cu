#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>


__device__ int recur(int n)
{
    if (n <= 1) return n;
    return recur(n - 1) + recur(n - 2);
}


__global__ void kernel()
{
    int n = 20;
    int result = recur(n);
    printf("device: %d\n", result);
    // int* array1;
    // cudaMalloc((void**) &array1, sizeof(int) * 3);
    // array1[0] = 1;
    // array1[1] = 2;
    // array1[2] = 3;

    // printf("device: malloc done.\n");

    // int* array2 = (int*) malloc(sizeof(int) * 3);

    // cudaMemcpyFromSymbol(array2, array1, sizeof(int) * 3);
    // free(array1);

    // printf("device: %d\n", array2[0]);
    // printf("device: %d\n", array2[1]);
    // printf("device: %d\n", array2[2]);
}


__host__ int main(int argc, char** argv)
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}