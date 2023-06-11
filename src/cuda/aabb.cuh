#pragma once


#include "ray.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>
#include <thrust/swap.h>


class aabb {

public:
    __device__ aabb() {}

    __device__ aabb(const vec3d& minimum, const vec3d& maximum)
    {
        this->minimum = minimum;
        this->maximum = maximum;
    }

    __device__ vec3d min() const { return minimum; }

    __device__ vec3d max() const { return maximum; }

    __device__ bool intersect(const ray&r) const
    {
        double t_min = r.t_min;
        double t_max = r.t_max;
        for (int i = 0; i < 3; i++) {
            double invert_d = 1.0 / r.direction[i];
            double t0 = (minimum[i] - r.origin[i]) * invert_d;
            double t1 = (maximum[i] - r.origin[i]) * invert_d;
            if (invert_d < 0.0) {
                thrust::swap(t0, t1);
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) {
                return false;
            }
        }
        return true;
    }

    __device__ friend aabb surrounding_box(aabb bbox0, aabb bbox1)
    {
        double min_x = fmin(bbox0.min().x(), bbox1.min().x());
        double min_y = fmin(bbox0.min().y(), bbox1.min().y());
        double min_z = fmin(bbox0.min().z(), bbox1.min().z());
        vec3d minimum = vec3d(min_x, min_y, min_z);

        double max_x = fmax(bbox0.max().x(), bbox1.max().x());
        double max_y = fmax(bbox0.max().y(), bbox1.max().y());
        double max_z = fmax(bbox0.max().z(), bbox1.max().z());
        vec3d maximum = vec3d(max_x, max_y, max_z);

        return aabb(minimum, maximum);
    }

private:
    vec3d minimum;
    vec3d maximum;
};