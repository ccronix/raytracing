#pragma once


#include "ray.cuh"
#include "aabb.cuh"
#include "util.cuh"
#include "intersection.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>


class object {

public:
    __device__ virtual ~object() {}
    __device__ virtual bool intersect(const ray& r, intersection& crossover) const = 0;
    __device__ virtual bool bounding_box(double start, double end, aabb& bbox) const = 0;
    __device__ virtual double pdf(const vec3d& origin, const vec3d& value) const { return 0.0; }
    __device__ virtual vec3d rand(const vec3d& origin, curandState* state) const { return vec3d(1, 0, 0); }
};
