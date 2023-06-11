#pragma once

#include <cstdlib>

#include "ray.cuh"
#include "aabb.cuh"
#include "util.cuh"
#include "object.cuh"
#include "intersection.cuh"

#include <cuda_runtime.h>



class group : public object {

public:
    __device__ group() {}

    __device__ group(object* obj) { add(obj); }

    __device__ group(const group& grp) 
    { 
        objects = grp.objects; 
        count = grp.size();
    }

    __device__ group operator = (const group& grp)
    {
        objects = grp.objects;
        count = grp.size();
        return *this;
    }

    __device__ void add(object* obj)
    {
        if (count == 0) {
            objects = (object**) malloc(sizeof(object*));
        }
        else {
            object** temp = (object**) malloc(sizeof(object*) * count);
            memcpy(temp, objects, sizeof(object*) * count);
            free(objects);
            objects = (object**) malloc(sizeof(object*) * (count + 1));
            memcpy(objects, temp, sizeof(object*) * (count + 1));
            free(temp);
        }
        objects[count] = obj;
        count++;
    }

    __device__ void add(const group& grp)
    {
        for (int i = 0; i < grp.size(); i++) {
            object* object = grp.objects[i];
            add(object);
        }
    }

    __device__ void clear() 
    { 
        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            free(object);
            object = nullptr;
        }
        free(objects);
        objects = nullptr;
        count = 0;
    }

    __device__ int size() const { return count; }

    __device__ object** content() const { return objects; }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        // printf("!! intro\n");
        intersection temp_crossover;
        bool has_intersect = false;
        double closest = r.t_max;

        for (int i = 0; i < count; i++) {
            // printf("!! %d %d\n", i, count);
            object* object = objects[i];
            if (object->intersect(ray(r.origin, r.direction, r.time, r.t_min, closest), temp_crossover)) {
                has_intersect = true;
                closest = temp_crossover.t;
                crossover = temp_crossover;
            }
        }
        return has_intersect;
    }

    __device__ virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        if (count == 0) {
            return false;
        }

        aabb temp_bbox;
        bool first = false;
        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            if (!object->bounding_box(start, end, temp_bbox)) {
                return false;
            }
            bbox = first ? temp_bbox : surrounding_box(bbox, temp_bbox);
            first = false;
        }
        return true;
    }

    __device__ virtual double pdf(const vec3d& origin, const vec3d& value) const override
    {
        double weight = 1.0 / count;
        double average = 0;
        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            average += weight * object->pdf(origin, value);
        }
        return average;
    }

    __device__ virtual vec3d rand(const vec3d& origin, curandState* state) const override
    {
        int index = random_int(0, count - 1, state);
        return objects[index]->rand(origin, state);
    }

private:
    object** objects = nullptr;
    int count = 0;
};
