#pragma once

#include <vector>
#include <algorithm>

#include "aabb.hpp"
#include "scene.hpp"
#include "object.hpp"



class bvh : public object {

public:
    object* left;
    object* right;
    aabb bbox;

    bvh() {}

    bvh(const scene& scn, double start=0, double end=0) : bvh(scn.objects, 0, scn.objects.size(), start, end) {}

    bvh(const std::vector<object*>& sources, int first, int last, double start, double end)
    {
        std::vector<object*> objects = sources;
        int axis = random_int(0, 2);
        auto function = (axis == 0) ? comparex : (axis == 1) ? comparey : comparez;
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
            std::sort(objects.begin() + first, objects.begin() + last, function);
            int middle = first + span / 2;
            left = new bvh(objects, first, middle, start, end);
            right = new bvh(objects, middle, last, start, end);
        }

        aabb bbox_left, bbox_right;
        if (!left->bounding_box(start, end, bbox_left) || !right->bounding_box(start, end, bbox_right)) {
            fprintf(stderr, "[ERROR] no bounding box in bvh constructor.");
        }
        bbox = surrounding_box(bbox_left, bbox_right);
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        if (!bbox.intersect(r)) {
            return false;
        }

        bool hit_left = left->intersect(r, crossover);
        double t_max_right = hit_left ? crossover.t : r.t_max;
        bool hit_right = right->intersect(ray(r.origin, r.direction, r.time, r.t_min, t_max_right), crossover);
        return hit_left || hit_right;
    }

    virtual bool bounding_box(double start, double end, aabb& output) const override
    {
        output = bbox;
        return true;
    }

    static bool compare(const object* a, const object* b, int axis)
    {
        aabb bbox_a, bbox_b;
        if (!a->bounding_box(0, 0, bbox_a) || !b->bounding_box(0, 0, bbox_b)) {
            fprintf(stderr, "[ERROR] no bounding box in bvh constructor.");
        }
        return bbox_a.min()[axis] < bbox_b.min()[axis];
    }

    static bool comparex(const object* a, const object* b)
    {
        return compare(a, b, 0);
    }

    static bool comparey(const object* a, const object* b)
    {
        return compare(a, b, 1);
    }

    static bool comparez(const object* a, const object* b)
    {
        return compare(a, b, 2);
    }
};