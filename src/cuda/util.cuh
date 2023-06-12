#pragma once


#include <limits>

#include <cmath>
#include <cstdlib>

#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>



__device__ bool near_zero(vec3d v)
{
    double e = 1e-8;
    return (fabsf(v.x() < e)) && (fabsf(v.y() < e)) && (fabsf(v.z() < e));
}


__device__ double degree_to_arc(double degree)
{
    return degree * 3.141592653589 / 180;
}


__device__ double random_double(curandState* state)
{
    return curand_uniform_double(state);
}


__device__ double random_double(double min, double max, curandState* state)
{
    return min + (max - min) * random_double(state);
}


__device__ int random_int(int min, int max, curandState* state)
{
    return int(random_double(min, max + 1, state));
}


__device__ vec3d random_vector(curandState* state)
{
    return vec3d(random_double(state), random_double(state), random_double(state));
}


__device__ vec3d random_vector(double min, double max, curandState* state)
{
    return vec3d(random_double(min, max, state), random_double(min, max, state), random_double(min, max, state));
}


__device__ vec3d random_shpere(curandState* state)
{
    while (true) {
        vec3d point = random_vector(-1, 1, state);
        double length = point.length();
        if (length * length >= 1) {
            continue;
        }
        return point.normalize();
    }
}


__device__ vec3d random_hemisphere(const vec3d& normal, curandState* state)
{
    vec3d value = random_shpere(state);
    if (value.dot(normal) > 0.0) {
        return value;
    }
    else {
        return -value;
    }
}


__device__ vec3d random_disk(curandState* state)
{
    while (true) {
        vec3d point = vec3d(random_double(-1, 1, state), random_double(-1, 1, state), 0);
        double length = point.length();
        if (length * length >= 1) {
            continue;
        }
        return point;
    }
}


__device__ vec3d random_cosine(curandState* state) {
    double r1 = random_double(state);
    double r2 = random_double(state);
    double z = sqrt(1 - r2);

    double phi = 2 * 3.141592653589 * r1;
    double x = cos(phi) * sqrt(r2);
    double y = sin(phi) * sqrt(r2);
    return vec3d(x, y, z);
}


__device__ vec3d gamma(vec3d value, double power)
{
    double x = pow(value.x(), power);
    double y = pow(value.y(), power);
    double z = pow(value.z(), power);
    return vec3d(x, y, z);
}


__device__ double clamp(double value, double min, double max)
{
    if (value < min) {
        return min;
    }
    else if (value > max) {
        return max;
    }
    else {
        return value;
    }
}


__device__ vec3d clamp(vec3d value, double min, double max)
{
    double x = clamp(value.x(), min, max);
    double y = clamp(value.y(), min, max);
    double z = clamp(value.z(), min, max);
    return vec3d(x, y, z);
}


__device__ bool is_infinity(vec3d color)
{
    return (fabs(color.x()) == INFINITY) || (fabs(color.y()) == INFINITY) || (fabs(color.z()) == INFINITY);
}


__device__ bool is_nan(vec3d color)
{
    return isnan(color.x()) || isnan(color.y()) || isnan(color.z());
}


__device__ vec3d reflect(const vec3d& input, const vec3d& normal)
{
    return input - 2 * input.dot(normal) * normal;
}


__device__ vec3d refract(const vec3d& input, const vec3d& normal, double refract_ratio)
{
    double cos_theta = fmin(-input.dot(normal), 1.0);
    vec3d out_perp = refract_ratio * (input + cos_theta * normal);
    vec3d out_parallel = -sqrt(fabs(1.0 - out_perp.length() * out_perp.length())) * normal;
    return out_perp + out_parallel;
}
