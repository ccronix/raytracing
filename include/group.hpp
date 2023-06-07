#pragma once

#include <cstdlib>

#include "ray.hpp"
#include "aabb.hpp"
#include "object.hpp"
#include "intersection.hpp"



class group : public object {

public:
    group() {}

    group(object* obj) { add(obj); }

    group(const group& grp) 
    { 
        objects = grp.objects; 
        count = grp.size();
    }

    group operator = (const group& grp)
    {
        objects = grp.objects;
        count = grp.size();
        return this;
    }

    void add(object* obj)
    {
        if (count == 0) {
            objects = (object**) malloc(sizeof(object*));
        }
        else {
            objects = (object**) realloc(objects, sizeof(object*) * (count + 1));
        }
        objects[count] = obj;
        count++;
    }

    void add(const group& grp)
    {
        for (int i = 0; i < grp.size(); i++) {
            object* object = grp.objects[i];
            add(object);
        }
    }

    void clear() 
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

    int size() const { return count; }

    object** content() const { return objects; }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        intersection temp_crossover;
        bool has_intersect = false;
        double closest = r.t_max;

        for (int i = 0; i < count; i++) {
            object* object = objects[i];
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

    virtual double pdf(const vec3d& origin, const vec3d& value) const override
    {
        double weight = 1.0 / count;
        double average = 0;
        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            average += weight * object->pdf(origin, value);
        }
        return average;
    }

    virtual vec3d rand(const vec3d& origin) const override
    {
        int index = random_int(0, count - 1);
        return objects[index]->rand(origin);
    }

private:
    object** objects = nullptr;
    int count = 0;
};
