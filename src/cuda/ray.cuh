#pragma once


#include "util.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>


struct ray {
    vec3d origin;
    vec3d direction;
    double time = 0;
    double t_min = 1e-3;
    double t_max = INFINITY;

    __device__ ray() {}

    __device__ ray(vec3d origin, vec3d direction, double time=0, double t_min=1e-3, double t_max=INFINITY) 
    {
        this->origin = origin;
        this->direction = direction;
        this->time = time;
        this->t_min = t_min;
        this->t_max = t_max;
    }

   __device__  vec3d at(double t) const { return origin + t * direction; }
};
