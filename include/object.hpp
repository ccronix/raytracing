#pragma once


#include "ray.hpp"
#include "aabb.hpp"
#include "intersection.hpp"
#include "algebra/algebra.hpp"


class object {

public:
    virtual ~object() {}
    virtual bool intersect(const ray& r, intersection& crossover) const = 0;
    virtual bool bounding_box(double start, double end, aabb& bbox) const = 0;
    virtual double pdf(const vec3d& origin, const vec3d& value) const { return 0.0; }
    virtual vec3d rand(const vec3d& origin) const { return vec3d(1, 0, 0); }
};
