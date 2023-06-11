#pragma once


#include "ray.cuh"
#include "util.cuh"
#include "object.cuh"
#include "intersection.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>


class translate : public object {

public:
    __device__ translate(object* obj, vec3d target)
    {
        this->obj = obj;
        this-> target = target;
    }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        ray offset_ray = ray(r.origin - target, r.direction, r.time, r.t_min, r.t_max);
        if (!obj->intersect(offset_ray, crossover)) {
            return false;
        }
        crossover.position += target;
        crossover.set_face_normal(offset_ray, crossover.normal);
        return true;
    }

    __device__ virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        if (!obj->bounding_box(start, end, bbox)) {
            return false;
        }
        bbox = aabb(bbox.min() + target, bbox.max() + target);
        return true;
    }

    __device__ virtual double pdf(const vec3d& origin, const vec3d& value) const override { return obj->pdf(origin, value); }

    __device__ virtual vec3d rand(const vec3d& origin, curandState* state) const override { return obj->rand(origin, state); }

private:
    object* obj;
    vec3d target;
};


class rotatey : public object {

public:
    __device__ rotatey(object* obj, double angle) 
    {
        this->obj = obj;
        double arc = degree_to_arc(angle);
        sin_theta = sin(arc);
        cos_theta = cos(arc);
        has_bbox = obj->bounding_box(0, 1, rotated_bbox);

        vec3d minimum = vec3d(INFINITY, INFINITY, INFINITY);
        vec3d maximum = vec3d(-INFINITY, -INFINITY, -INFINITY);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double x = i * rotated_bbox.max().x() + (1 - i) * rotated_bbox.min().x();
                    double y = j * rotated_bbox.max().y() + (1 - j) * rotated_bbox.min().y();
                    double z = k * rotated_bbox.max().z() + (1 - k) * rotated_bbox.min().z();

                    double new_x = cos_theta * x + sin_theta * z;
                    double new_z = -sin_theta * x + cos_theta * z;

                    vec3d temp = vec3d(new_x, y, new_z);
                    for (int l = 0; l < 3; l++) {
                        minimum[l] = fmin(minimum[l], temp[l]);
                        maximum[l] = fmax(maximum[l], temp[l]);
                    }
                }
            }
        }
        rotated_bbox = aabb(minimum, maximum);
    }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        vec3d origin = r.origin;
        vec3d direction = r.direction;

        origin[0] = cos_theta * r.origin.x() - sin_theta * r.origin.z();
        origin[2] = sin_theta * r.origin.x() + cos_theta * r.origin.z();

        direction[0] = cos_theta * r.direction.x() - sin_theta * r.direction.z();
        direction[2] = sin_theta * r.direction.x() + cos_theta * r.direction.z();

        ray rotated_r = ray(origin, direction, r.time, r.t_min, r.t_max);

        if (!obj->intersect(rotated_r, crossover)) {
            return false;
        }
        vec3d position = crossover.position;
        vec3d normal = crossover.normal;

        position[0] = cos_theta * crossover.position.x() + sin_theta * crossover.position.z();
        position[2] = -sin_theta * crossover.position.x() + cos_theta * crossover.position.z();

        normal[0] = cos_theta * crossover.normal.x() + sin_theta * crossover.normal.z();
        normal[2] = -sin_theta * crossover.normal.x() + cos_theta * crossover.normal.z();

        crossover.position = position;
        crossover.set_face_normal(rotated_r, normal);
        return true;
    }

    __device__ virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        bbox = rotated_bbox;
        return has_bbox;
    }

    __device__ virtual double pdf(const vec3d& origin, const vec3d& value) const override { return obj->pdf(origin, value); }

    __device__ virtual vec3d rand(const vec3d& origin, curandState* state) const override { return obj->rand(origin, state); }

private:
    object* obj;
    double sin_theta;
    double cos_theta;
    bool has_bbox;
    aabb rotated_bbox;
};


class flip : public object {

public:
    __device__ flip(object* obj) { this->obj = obj; }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        if (!obj->intersect(r, crossover)) {
            return false;
        }
        crossover.frontward = !crossover.frontward;
        return true;
    }

    __device__ virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        return obj->bounding_box(start, end, bbox);
    }

    __device__ virtual double pdf(const vec3d& origin, const vec3d& value) const override { return obj->pdf(origin, value); }

    __device__ virtual vec3d rand(const vec3d& origin, curandState* state) const override { return obj->rand(origin, state); }

private:
    object* obj;
};
