#pragma once


#include <vector>

#include "ray.hpp"
#include "aabb.hpp"
#include "object.hpp"
#include "intersection.hpp"



class scene : public object {

public:
    std::vector<object*> objects;

    scene() {}

    scene(object* obj) { add(obj); }

    scene(scene& scn) { objects = scn.objects; }

    scene(scene&& scn) { objects = std::move(scn.objects); }

    void add(object* obj) { objects.push_back(obj); }

    void add(const scene& scn)
    {
        for (auto& object : scn.objects) {
            objects.push_back(object);
        }
    }

    void clear() { objects.clear(); }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        intersection temp_crossover;
        bool has_intersect = false;
        double closest = r.t_max;

        for (auto& object : objects) {
            if (object->intersect(ray(r.origin, r.direction, r.time, r.t_min, closest), temp_crossover)) {
                has_intersect = true;
                closest = temp_crossover.t;
                crossover = temp_crossover;
            }
        }
        return has_intersect;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        if (objects.empty()) {
            return false;
        }

        aabb temp_bbox;
        bool first = false;
        for (auto& object : objects) {
            if (!object->bounding_box(start, end, temp_bbox)) {
                return false;
            }
            bbox = first ? temp_bbox : surrounding_box(bbox, temp_bbox);
            first = false;
        }
        return true;
    }
};
