#pragma once


#include "pdf.hpp"
#include "texture.hpp"
#include "intersection.hpp"
#include "algebra/algebra.hpp"



class material {

public:
    virtual vec3d emit(const ray&r, const intersection& crossover, const vec3d& position, const vec2d uv_coord) const { return vec3d(0, 0, 0); }
    virtual bool shading(const ray&r, const intersection& crossover, scatter& scatter) const { return false; };
    virtual double shading_pdf(const ray& r, const intersection& crossover, const ray& scatter) const { return 0.0; };
};


class lambertian : public material {

public:
    texture* base_color;

    lambertian(const vec3d& color) { base_color = new constant(color); }

    lambertian(texture* tex) { base_color = tex; }

    virtual bool shading(const ray& r, const intersection& crossover, scatter& scatter) const override
    {
        scatter.is_spec = false;
        scatter.attenuation = base_color->color(crossover.position, crossover.uv_coord);
        scatter.pdf_ptr = new cos_pdf(crossover.normal);
        return true;
    }

    virtual double shading_pdf(const ray& r, const intersection& crossover, const ray& scatter) const override
    {
        double cosine = crossover.normal.dot(scatter.direction.normalize());
        return cosine < 0 ? 0 : cosine / pi;
    }
};


class metal : public material {

public:
    texture* base_color;
    double roughness;

    metal(const vec3d& color, double roughness) 
    {
        base_color = new constant(color);
        this->roughness = roughness < 1 ? roughness : 1;
    }

    metal(texture* tex, double roughness) 
    {
        base_color = tex;
        this->roughness = roughness < 1 ? roughness : 1;
    }

    virtual bool shading(const ray& r, const intersection& crossover, scatter& scatter) const override
    {
        vec3d next = reflect(r.direction.normalize(), crossover.normal);
        vec3d fuzz = roughness * random_shpere();
        scatter.specular = ray(crossover.position, next + fuzz, r.time, r.t_min, r.t_max);
        scatter.attenuation = base_color->color(crossover.position, crossover.uv_coord);
        scatter.pdf_ptr = nullptr;
        scatter.is_spec = true;
        return true;
    }
};


class dielectric : public material {

public:
    double ior;

    dielectric(double ior) { this->ior = ior; }

    virtual bool shading(const ray& r, const intersection& crossover, scatter& scatter) const override
    {
        scatter.is_spec = true;
        scatter.pdf_ptr = nullptr;
        scatter.attenuation = vec3d(1, 1, 1);
        double refract_ratio = crossover.frontward ? (1.0 / ior) : ior;

        vec3d next = r.direction.normalize();
        double cos_theta = fmin((-next).dot(crossover.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        bool not_refract = refract_ratio * sin_theta > 1.0;

        vec3d referaction;
        if (not_refract || reflectance(cos_theta, refract_ratio) > random()) {
            referaction = reflect(next, crossover.normal);
        }
        else {
            referaction =  refract(next, crossover.normal, refract_ratio);
        }
        scatter.specular = ray(crossover.position, referaction, r.time, r.t_min, r.t_max);
        return true;
    }

private:
    static double reflectance(double cos_theta, double ior)
    {
        double r = (1 - ior) / (1 + ior);
        r = r * r;
        return r + (1 - r) * pow((1 - cos_theta), 5);
    }
};


class light : public material {

public:
    texture* emission;

    light(texture* tex) { emission = tex; }

    light(const vec3d& color) { emission = new constant(color); }

    virtual vec3d emit(const ray&r, const intersection& crossover, const vec3d& position, const vec2d uv_coord) const override
    {        
        if (crossover.frontward) {
            return emission->color(position, uv_coord);
        }
        else {
            return vec3d(0, 0, 0); 
        }
    }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const
    {
        return false;
    }
};


class fog : public material {

public:
    texture* base_color;

    fog(vec3d color) { base_color = new constant(color); }

    fog(texture* tex) { base_color = tex; }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const
    {
        scatter = ray(crossover.position, random_shpere(), r.time, r.t_min, r.t_max);
        attenuation = base_color->color(crossover.position, crossover.uv_coord);
        return true;
    }    
};
