#pragma once


#include "algebra/algebra.hpp"


class onb {

public:
    onb() {}

    vec3d operator [] (int index) const {return axis[index]; }

    vec3d u() const { return axis[0]; }

    vec3d v() const { return axis[1]; }

    vec3d w() const { return axis[2]; }

    vec3d local(double x, double y, double z) const { return x * u() + y * v() + z * w(); }

    vec3d local(const vec3d& vec) const { return vec.x() * u() + vec.y() * v() + vec.z() * w(); }

    void build(const vec3d& vec) 
    {
        axis[2] = vec.normalize();
        vec3d x = (fabs(w().x()) > 0.9) ? vec3d(0, 1, 0) : vec3d(1, 0, 0);
        axis[1] = w().cross(x).normalize();
        axis[0] = w().cross(v());
    }

private:
    vec3d axis[3];
};
