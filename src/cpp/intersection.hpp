#pragma once

#include "ray.hpp"
#include "algebra/algebra.hpp"


class pdf;


class material;


struct intersection {

    vec3d position;
    vec3d normal;
    vec2d uv_coord;
    double t;
    material* mat;
    bool frontward;

    void set_face_normal(const ray& r, const vec3d& outward_normal)
    {
        frontward = r.direction.dot(outward_normal) < 0;
        normal = frontward ? outward_normal : -outward_normal;
    }
};


struct scatter {

ray specular;
bool is_spec;
pdf* pdf_ptr;
vec3d attenuation;
};
