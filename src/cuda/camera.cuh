#pragma once


#include "ray.cuh"
#include "util.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>


class camera {

public:
    vec3d position;
    vec3d lookat;
    vec3d up;
    double fov;
    double aspect_ratio;
    double aperture;
    double focus_distance;

    __device__ camera() {}

    __device__ camera(
        vec3d position, 
        vec3d lookat, 
        vec3d up, 
        double fov, 
        double aspect_ratio, 
        double aperture=0, 
        double focus_distance=1, 
        double shutter_start=0, 
        double shutter_end=0
    )
    {
        this->position = position;
        this->lookat = lookat;
        this->up = up;
        this->fov = fov;
        this->aspect_ratio = aspect_ratio;
        this->aperture = aperture;
        this->focus_distance = focus_distance;
        this->shutter_start = shutter_start;
        this->shutter_end = shutter_end;
        setup_camera();
    }

    __device__ void setup_camera()
    {
        double arc_fov = degree_to_arc(fov);
        double sensor_h = tan(arc_fov / 2) * 2;
        double sensor_w = sensor_h * aspect_ratio;

        w = (position - lookat).normalize();
        u = up.cross(w).normalize();
        v = w.cross(u);

        horizontal = focus_distance * sensor_w * u;
        vertical = focus_distance * sensor_h * v;
        left_bottom = position - horizontal / 2 - vertical / 2 - focus_distance * w;
        lens_radius = aperture / 2;
    }

    __device__ ray emit(double x, double y, curandState* state) const
    {
        vec3d defocus = lens_radius * random_disk(state);
        vec3d offset = u * defocus.x() + v * defocus.y();
        vec3d direction = left_bottom + x * horizontal + y * vertical - position - offset;
        return ray(position + offset, direction, random_double(shutter_start, shutter_end, state));
    }

private:
    vec3d left_bottom;
    vec3d horizontal;
    vec3d vertical;
    vec3d u, v, w;
    double lens_radius;
    double shutter_start;
    double shutter_end;
};