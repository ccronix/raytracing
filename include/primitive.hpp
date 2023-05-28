#pragma once


#include "aabb.hpp"
#include "scene.hpp"
#include "object.hpp"
#include "material.hpp"
#include "intersection.hpp"
#include "algebra/algebra.hpp"



class sphere : public object {

public:
    vec3d center;
    double radius;
    material* mat;

    sphere() {}

    sphere(vec3d center, double radius, material* mat) 
    {
        this->center = center;
        this->radius = radius;
        this->mat = mat;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        vec3d oc = r.origin - center;
        double a = r.direction.dot(r.direction);
        double b = 2.0 * oc.dot(r.direction);
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return false;
        }

        double root = (-b - sqrt(discriminant)) / (a * 2);
        
        if (root < r.t_min || root > r.t_max) {
            root = (-b + sqrt(discriminant)) / (a * 2);
            if (root < r.t_min || root > r.t_max) {
                return false; 
            }
        }
        crossover.t = root;
        crossover.position = r.at(root);
        vec3d outward_normal = (crossover.position - center) / radius;
        crossover.set_face_normal(r, outward_normal);
        crossover.uv_coord = uv(outward_normal);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        vec3d minimum = center - vec3d(radius, radius, radius);
        vec3d maximum = center + vec3d(radius, radius, radius);
        bbox = aabb(minimum, maximum);
        return true;
    }

private:
    static vec2d uv(const vec3d& position)
    {
        double theta = acos(-position.y());
        double phi = atan2(-position.z(), position.x()) + pi;
        double u = phi / (2 * pi);
        double v = theta / pi;
        return vec2d(u, v);
    }
};


class msphere : public object {

public:
    vec3d center0, center1;
    double start, end;
    double radius;
    material* mat;

    msphere() {}

    msphere(vec3d center0, vec3d center1, double start, double end, double radius, material* mat) 
    {
        this->center0 = center0;
        this->center1 = center1;
        this->start = start;
        this->end = end;
        this->radius = radius;
        this->mat = mat;
    }

    vec3d center(double time) const
    {
        return center0 + ((time - start) / (end - start)) * (center1 - center0);
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        vec3d oc = r.origin - center(r.time);
        double a = r.direction.dot(r.direction);
        double b = 2.0 * oc.dot(r.direction);
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return false;
        }

        double root = (-b - sqrt(discriminant)) / (a * 2);
        
        if (root < r.t_min || root > r.t_max) {
            root = (-b + sqrt(discriminant)) / (a * 2);
            if (root < r.t_min || root > r.t_max) {
                return false; 
            }
        }
        crossover.t = root;
        crossover.position = r.at(root);
        vec3d outward_normal = (crossover.position - center(r.time)) / radius;
        crossover.set_face_normal(r, outward_normal);
        crossover.uv_coord = uv(outward_normal);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        vec3d min_start = center(start) - vec3d(radius, radius, radius);
        vec3d max_start = center(start) + vec3d(radius, radius, radius);
        aabb bbox_start = aabb(min_start, max_start);

        vec3d min_end = center(end) - vec3d(radius, radius, radius);
        vec3d max_end = center(end) + vec3d(radius, radius, radius);
        aabb bbox_end = aabb(min_end, max_end);

        bbox = surrounding_box(bbox_start, bbox_end);
        return true;
    }

private:
    static vec2d uv(const vec3d& position)
    {
        double theta = acos(-position.y());
        double phi = atan2(-position.z(), position.x()) + pi;
        double u = phi / (2 * pi);
        double v = theta / pi;
        return vec2d(u, v);
    }
};


class planexy : public object {

public:
    material* mat;
    double x0, x1, y0, y1, k;

    planexy() {}

    planexy(double x0, double x1, double y0, double y1, double k, material* mat)
    {
        this->x0 = x0;
        this->x1 = x1;
        this->y0 = y0;
        this->y1 = y1;
        this->k = k;
        this->mat = mat;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        double t = (k - r.origin.z()) / r.direction.z();
        if (t < r.t_min || t > r.t_max) {
            return false;
        }

        double x = r.origin.x() + t * r.direction.x();
        double y = r.origin.y() + t * r.direction.y();
        if (x < x0 || x > x1 || y < y0 || y > y1) {
            return false;
        }

        double u = (x - x0) / (x1 - x0);
        double v = (y - y0) / (y1 - y0);

        vec3d outward_normal = vec3d(0, 0, 1);
        crossover.set_face_normal(r, outward_normal);
        crossover.uv_coord = vec2d(u, v);
        crossover.t = t;
        crossover.position = r.at(t);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        bbox = aabb(vec3d(x0, y0, k - 0.001), vec3d(x1, y1, k + 0.001));
        return true;
    }
};


class planexz : public object {

public:
    material* mat;
    double x0, x1, z0, z1, k;

    planexz() {}

    planexz(double x0, double x1, double z0, double z1, double k, material* mat)
    {
        this->x0 = x0;
        this->x1 = x1;
        this->z0 = z0;
        this->z1 = z1;
        this->k = k;
        this->mat = mat;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        double t = (k - r.origin.y()) / r.direction.y();
        if (t < r.t_min || t > r.t_max) {
            return false;
        }

        double x = r.origin.x() + t * r.direction.x();
        double z = r.origin.z() + t * r.direction.z();
        if (x < x0 || x > x1 || z < z0 || z > z1) {
            return false;
        }

        double u = (x - x0) / (x1 - x0);
        double v = (z - z0) / (z1 - z0);

        vec3d outward_normal = vec3d(0, 1, 0);
        crossover.set_face_normal(r, outward_normal);
        crossover.uv_coord = vec2d(u, v);
        crossover.t = t;
        crossover.position = r.at(t);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        bbox = aabb(vec3d(x0, k - 0.001, z0), vec3d(x1, k + 0.001, z1));
        return true;
    }

    virtual double pdf(const vec3d& origin, const vec3d& value) const override
    {
        intersection crossover;
        if (!this->intersect(ray(origin, value), crossover)) {
            return 0;
        }

        double area = (x1 - x0) * (z1 - z0);
        double distance_sqr = crossover.t * crossover.t * value.length() * value.length();
        double cosine = fabs(value.dot(crossover.normal) / value.length());
        return distance_sqr / (cosine * area);
    }
    virtual vec3d rand(const vec3d& origin) const override
    {
        vec3d point = vec3d(random_double(x0, x1), k, random_double(z0, z1));
        return point - origin;
    }
};


class planeyz : public object {

public:
    material* mat;
    double y0, y1, z0, z1, k;

    planeyz() {}

    planeyz(double y0, double y1, double z0, double z1, double k, material* mat)
    {
        this->y0 = y0;
        this->y1 = y1;
        this->z0 = z0;
        this->z1 = z1;
        this->k = k;
        this->mat = mat;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        double t = (k - r.origin.x()) / r.direction.x();
        if (t < r.t_min || t > r.t_max) {
            return false;
        }

        double y = r.origin.y() + t * r.direction.y();
        double z = r.origin.z() + t * r.direction.z();
        if (y < y0 || y > y1 || z < z0 || z > z1) {
            return false;
        }

        double u = (y - y0) / (y1 - y0);
        double v = (z - z0) / (z1 - z0);

        vec3d outward_normal = vec3d(1, 0, 0);
        crossover.set_face_normal(r, outward_normal);
        crossover.uv_coord = vec2d(u, v);
        crossover.t = t;
        crossover.position = r.at(t);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        bbox = aabb(vec3d(k - 0.001, y0, z0), vec3d(k + 0.001, y1, z1));
        return true;
    }
};


class volume : public object {

public:
    volume(object* surface, double density, texture* tex)
    {
        this->surface = surface;
        mat = new fog(tex);
        neg_inv_d = -1.0 / density;
    }

    volume(object* surface, double density, const vec3d& color)
    {
        this->surface = surface;
        mat = new fog(color);
        neg_inv_d = -1.0 / density;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        intersection crossover1, crossover2;
        if (!surface->intersect(ray(r.origin, r.direction, r.time, -infinity, infinity), crossover1)) {
            return false;
        }

        if (!surface->intersect(ray(r.origin, r.direction, r.time, crossover1.t + 0.0001, infinity), crossover2)) {
            return false;
        }

        if (crossover1.t < r.t_min) {
            crossover1.t = r.t_min;
        }
        if (crossover2.t > r.t_max) {
            crossover2.t = r.t_max;
        }
        
        if (crossover1.t >= crossover2.t) {
            return false;
        }
        if (crossover1.t < 0) {
            crossover1.t = 0;
        }

        double ray_length = r.direction.length();
        double inner_length = (crossover2.t - crossover1.t) * ray_length;
        double intersect_distance = neg_inv_d * log(random_double());

        if (intersect_distance > inner_length) {
            return false;
        }

        crossover.t = crossover1.t + intersect_distance / ray_length;
        crossover.position = r.at(crossover.t);
        crossover.normal = vec3d(1, 0, 0);
        crossover.frontward = false;
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        return surface->bounding_box(start, end, bbox);
    }

private:
    object* surface;
    material* mat;
    double neg_inv_d;
};


class box : public object {

public:
    vec3d minimum;
    vec3d maximum;

    box() {}

    box(const vec3d& minimum, const vec3d& maximum, material* mat)
    {
        this->minimum = minimum;
        this->maximum = maximum;

        xy0 = new planexy(minimum.x(), maximum.x(), minimum.y(), maximum.y(), maximum.z(), mat);
        xy1 = new planexy(minimum.x(), maximum.x(), minimum.y(), maximum.y(), minimum.z(), mat);

        xz0 = new planexz(minimum.x(), maximum.x(), minimum.z(), maximum.z(), maximum.y(), mat);
        xz1 = new planexz(minimum.x(), maximum.x(), minimum.z(), maximum.z(), minimum.y(), mat);

        yz0 = new planeyz(minimum.y(), maximum.y(), minimum.z(), maximum.z(), maximum.x(), mat);
        yz1 = new planeyz(minimum.y(), maximum.y(), minimum.z(), maximum.z(), minimum.x(), mat);

        sides.add(xy0);
        sides.add(xy1);
        sides.add(xz0);
        sides.add(xz1);
        sides.add(yz0);
        sides.add(yz1);
    }

    virtual ~box() override
    {
        delete xy0;
        delete xy1;
        delete xz0;
        delete xz1;
        delete yz0;
        delete yz1;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        return sides.intersect(r, crossover);
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        bbox = aabb(minimum, maximum);
        return true;
    }

private:
    scene sides;
    object* xy0 = nullptr;
    object* xy1 = nullptr;
    object* xz0 = nullptr;
    object* xz1 = nullptr;
    object* yz0 = nullptr;
    object* yz1 = nullptr;
};
