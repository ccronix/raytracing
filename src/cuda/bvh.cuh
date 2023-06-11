#pragma once

#include <cstdlib>

#include "aabb.cuh"
#include "group.cuh"
#include "object.cuh"

#include <cuda_runtime.h>
#include <thrust/sort.h>



class bvh : public object {

public:
    object* left;
    object* right;
    aabb bbox;

    __device__ bvh() {}

    __device__ bvh(const group& grp, double start=0, double end=0) : bvh(grp.content(), 0, grp.size(), start, end) {}

    __device__ bvh(object** objects, int first, int last, double start, double end)
    {
        int axis = 1;
        auto function = (axis == 0) ? comparex : (axis == 1) ? comparey : comparez;
        auto qcompare = (axis == 0) ? qcomparex : (axis == 1) ? qcomparey : qcomparez;
        int span = last - first;

        if (span == 1) {
            right = objects[first];
            left = right;
        }
        else if (span == 2) {
            if (function(objects[first], objects[first + 1])) {
                left = objects[first];
                right = objects[first + 1];
            }
            else {
                left = objects[first + 1];
                right = objects[first];
            }
        }
        else {
            thrust::sort(objects + first, objects + last, function);
            int middle = first + span / 2;
            left = new bvh(objects, first, middle, start, end);
            right = new bvh(objects, middle, last, start, end);
        }

        aabb bbox_left, bbox_right;
        if (!left->bounding_box(start, end, bbox_left) || !right->bounding_box(start, end, bbox_right)) {
            printf("[ERROR] no bounding box in bvh constructor.\n");
        }
        bbox = surrounding_box(bbox_left, bbox_right);
    }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        if (!bbox.intersect(r)) {
            return false;
        }

        bool hit_left = left->intersect(r, crossover);
        double t_max_right = hit_left ? crossover.t : r.t_max;
        bool hit_right = right->intersect(ray(r.origin, r.direction, r.time, r.t_min, t_max_right), crossover);
        return hit_left || hit_right;
    }

    __device__ virtual bool bounding_box(double start, double end, aabb& output) const override
    {
        output = bbox;
        return true;
    }

    __device__ static bool compare(const object* a, const object* b, int axis)
    {
        aabb bbox_a, bbox_b;
        if (!a->bounding_box(0, 0, bbox_a) || !b->bounding_box(0, 0, bbox_b)) {
            printf("[ERROR] no bounding box in bvh constructor.\n");
        }
        return bbox_a.min()[axis] < bbox_b.min()[axis];
    }

    __device__ static bool comparex(const object* a, const object* b)
    {
        return compare(a, b, 0);
    }

    __device__ static bool comparey(const object* a, const object* b)
    {
        return compare(a, b, 1);
    }

    __device__ static bool comparez(const object* a, const object* b)
    {
        return compare(a, b, 2);
    }

    __device__ static int qcomparex(const void* a, const void* b)
    {
        return compare(*(object**) a, *(object**) b, 0) ? 1 : -1;
    }

    __device__ static int qcomparey(const void* a, const void* b)
    {
        return compare(*(object**) a, *(object**) b, 1) ? 1 : -1;
    }

    __device__ static int qcomparez(const void* a, const void* b)
    {
        return compare(*(object**) a, *(object**) b, 2) ? 1 : -1;
    }
};