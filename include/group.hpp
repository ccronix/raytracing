#pragma once


#include <vector>

#include "ray.hpp"
#include "aabb.hpp"
#include "object.hpp"
#include "intersection.hpp"



class group : public object {

public:
    group() {}

    group(object* obj) { add(obj); }

    group(const group& grp) { objects = grp.objects; }

    group(const group&& grp) { objects = std::move(grp.objects); }

    group operator = (const group& grp)
    {
        objects = grp.objects;
        return this;
    }

    void add(object* obj) { objects.push_back(obj); }

    void add(const group& grp)
    {
        for (auto& object : grp.objects) {
            objects.push_back(object);
        }
    }

    void clear() { objects.clear(); }

    std::vector<object*> content() const { return objects; }

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

    virtual double pdf(const vec3d& origin, const vec3d& value) const override
    {
        double weight = 1.0 / objects.size();
        double average = 0;
        for (auto& object : objects) {
            average += weight * object->pdf(origin, value);
        }
        return average;
    }

    virtual vec3d rand(const vec3d& origin) const override
    {
        int index = random_int(0, objects.size() - 1);
        return objects[index]->rand(origin);
    }

private:
    std::vector<object*> objects;
};
