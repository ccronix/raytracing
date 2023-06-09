#pragma once


#include "util.hpp"
#include "algebra/algebra.hpp"


struct ray {
    vec3d origin;
    vec3d direction;
    double time = 0;
    double t_min = 1e-3;
    double t_max = infinity;

    ray() {}

    ray(vec3d origin, vec3d direction, double time=0, double t_min=1e-3, double t_max=infinity) 
    {
        this->origin = origin;
        this->direction = direction;
        this->time = time;
        this->t_min = t_min;
        this->t_max = t_max;
    }

    vec3d at(double t) const { return origin + t * direction; }
};
