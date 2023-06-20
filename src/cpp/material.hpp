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
    virtual bool has_emission() const { return false; }
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
        return cosine > 0 ? cosine / pi : 0;
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
        if (not_refract || reflectance(cos_theta, refract_ratio) > random_double()) {
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


class emissive : public material {

public:
    texture* emission;

    emissive(texture* tex) { emission = tex; }

    emissive(const vec3d& color) { emission = new constant(color); }

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

    virtual bool has_emission() const override { return true; }
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


class phong : public material {

public:
    vec3d Kd = vec3d(0.9, 0.9, 0.9);
    vec3d Ks = vec3d(0.0, 0.0, 0.0);
    vec3d Ke = vec3d(0.0, 0.0, 0.0);
    vec3d Tr = vec3d(0.0, 0.0, 0.0);
    float Ns = 0.0;
    float Ni = 0.0;

    texture* diffuse_map = nullptr;
    texture* normal_map = nullptr;

    phong(const vec3d& color){ Kd = color; }

    phong(texture* tex) { diffuse_map = tex; }

    virtual vec3d emit(const ray&r, const intersection& crossover, const vec3d& position, const vec2d uv_coord) const override
    {  
        if (is_emissive() && crossover.frontward) {
            return Ke;
        }
        else {
            return vec3d(0, 0, 0); 
        }
    }

    virtual bool shading(const ray& r, const intersection& crossover, scatter& scatter) const override
    {
        if (is_emissive()) {
            return false;
        }

        vec3d base_color;
        if (diffuse_map != nullptr) {
            base_color = diffuse_map->color(crossover.position, crossover.uv_coord);
        }
        else {
            base_color = Kd;
        }

        if (is_specular() && random_double() < 0.5) {
            vec3d next = reflect(r.direction.normalize(), crossover.normal);
            vec3d fuzz = 1 / log(Ns) * random_shpere();

            vec3d normal = crossover.normal.normalize();
            vec3d temp = next.cross(normal);
            vec3d tangent = normal.cross(temp).normalize();
            vec3d bi_tangent = normal.cross(tangent).normalize();
            double fuzz_n = fuzz.dot(normal);
            double fuzz_t = fuzz.dot(tangent);
            double fuzz_bt = fuzz.dot(bi_tangent);
            fuzz_t = fuzz_t / next.normalize().dot(normal);
            fuzz_bt = fuzz_bt * next.normalize().dot(normal);

            fuzz = normal * fuzz_n + tangent * fuzz_t + bi_tangent * fuzz_bt;

            scatter.specular = ray(crossover.position, next + fuzz, r.time, r.t_min, r.t_max);
            scatter.attenuation = Ks;
            scatter.pdf_ptr = nullptr;
            scatter.is_spec = true;
            return true;
        }

        if (is_transmit()) {
            scatter.is_spec = true;
            scatter.pdf_ptr = nullptr;
            scatter.attenuation = Tr;
            double refract_ratio = crossover.frontward ? (1.0 / Ni) : Ni;

            vec3d next = r.direction.normalize();
            double cos_theta = fmin((-next).dot(crossover.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            bool not_refract = refract_ratio * sin_theta > 1.0;

            vec3d referaction;
            if (not_refract || reflectance(cos_theta, refract_ratio) > random_double()) {
                referaction = reflect(next, crossover.normal);
            }
            else {
                referaction =  refract(next, crossover.normal, refract_ratio);
            }
            scatter.specular = ray(crossover.position, referaction, r.time, r.t_min, r.t_max);
            return true;
        }

        scatter.is_spec = false;
        scatter.attenuation = base_color;
        scatter.pdf_ptr = new cos_pdf(crossover.normal);
        return true;
    }

    virtual double shading_pdf(const ray& r, const intersection& crossover, const ray& scatter) const override
    {
        double cosine = crossover.normal.dot(scatter.direction.normalize());
        return cosine > 0 ? cosine / pi : 0;
    }

private:
    bool is_specular() const { return Ks.x() > 0 || Ks.y() || Ks.z(); }

    bool is_emissive() const { return Ke.x() > 0 || Ke.y() || Ke.z(); }

    bool is_transmit() const { return Tr.x() > 0 || Tr.y() || Tr.z(); }

    virtual bool has_emission() const override { return is_emissive(); }

    static double reflectance(double cos_theta, double ior)
    {
        double r = (1 - ior) / (1 + ior);
        r = r * r;
        return r + (1 - r) * pow((1 - cos_theta), 5);
    }
};
