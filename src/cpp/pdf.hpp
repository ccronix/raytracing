#pragma once

#include "onb.hpp"
#include "util.hpp"
#include "object.hpp"
#include "algebra/algebra.hpp"


class pdf {

public:
    virtual ~pdf() {}

    virtual double value(const vec3d& direction) const = 0;

    virtual vec3d generate() const = 0;
};


class cos_pdf : public pdf {

public:
    cos_pdf(const vec3d& w) { uvw.build(w); }

    virtual double value(const vec3d& direction) const override
    {
        double cosine = direction.normalize().dot(uvw.w());
        return (cosine <=0) ? 0 : cosine / pi;
    }

    virtual vec3d generate() const override
    {
        return uvw.local(random_cosine());
    }

private:
    onb uvw;
};


class obj_pdf : public pdf {

public:
    obj_pdf(object* obj, const vec3d& origin)
    {
        this->obj = obj;
        this->origin = origin;
    }

    virtual double value(const vec3d& direction) const override
    {
        return obj->pdf(origin, direction);
    }

    virtual vec3d generate() const override
    {
        return obj->rand(origin);
    }



private:
    object* obj;
    vec3d origin;
};


class mix_pdf : public pdf {

public:
    mix_pdf(pdf* pdf0, pdf* pdf1)
    {
        this->pdf0 = pdf0;
        this->pdf1 = pdf1;
    }

    virtual double value(const vec3d& direction) const override
    {
        return 0.5 * (pdf0->value(direction) + pdf1->value(direction));
    }

    virtual vec3d generate() const override
    {
        if (random_double() < 0.5) {
            return pdf0->generate();
        }
        else {
            return pdf1->generate();
        }
    }

private:
    pdf* pdf0;
    pdf* pdf1;
};
