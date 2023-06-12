#pragma once

#include "onb.cuh"
#include "util.cuh"
#include "object.cuh"
#include "device_algebra/algebra.hpp"

#include <cuda_runtime.h>


class pdf {

public:
    __device__ virtual ~pdf() {}

    __device__ virtual double value(const vec3d& direction) const = 0;

    __device__ virtual vec3d generate(curandState* state) const = 0;
};


class cos_pdf : public pdf {

public:
    __device__ cos_pdf(const vec3d& w) { uvw.build(w); }

    __device__ virtual double value(const vec3d& direction) const override
    {
        double cosine = direction.normalize().dot(uvw.w());
        return (cosine <=0) ? 0 : cosine / 3.141592653589;
    }

    __device__ virtual vec3d generate(curandState* state) const override
    {
        return uvw.local(random_cosine(state));
    }

private:
    onb uvw;
};


__device__ double cos_pdf_value(const vec3d& normal, const vec3d& direction)
{
    double cosine = direction.normalize().dot(normal.normalize());
    return (cosine <=0) ? 0 : cosine / 3.141592653589;
}


__device__ vec3d cos_pdf_generate(const vec3d normal, curandState* state)
{
    vec3d rand_cos = random_cosine(state);

    vec3d axis[3];
    axis[2] = normal.normalize();
    vec3d x = (fabs(axis[2].x()) > 0.9) ? vec3d(0, 1, 0) : vec3d(1, 0, 0);
    axis[1] = axis[2].cross(x).normalize();
    axis[0] = axis[2].cross(axis[1]);
    return rand_cos.x() * axis[0] + rand_cos.y() * axis[1] + rand_cos.z() * axis[2];
}


class obj_pdf : public pdf {

public:
    __device__ obj_pdf(object* obj, const vec3d& origin)
    {
        this->obj = obj;
        this->origin = origin;
    }

    __device__ virtual double value(const vec3d& direction) const override
    {
        return obj->pdf(origin, direction);
    }

    __device__ virtual vec3d generate(curandState* state) const override
    {
        return obj->rand(origin, state);
    }



private:
    object* obj;
    vec3d origin;
};


class mix_pdf : public pdf {

public:
    __device__ mix_pdf(pdf* pdf0, pdf* pdf1)
    {
        this->pdf0 = pdf0;
        this->pdf1 = pdf1;
    }

    __device__ virtual double value(const vec3d& direction) const override
    {
        return 0.5 * (pdf0->value(direction) + pdf1->value(direction));
    }

    __device__ virtual vec3d generate(curandState* state) const override
    {
        if (random_double(state) < 0.5) {
            return pdf0->generate(state);
        }
        else {
            return pdf1->generate(state);
        }
    }

private:
    pdf* pdf0;
    pdf* pdf1;
};
