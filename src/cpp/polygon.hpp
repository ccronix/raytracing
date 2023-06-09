#pragma once


#include <cstdlib>
#include <algorithm>

#include "ray.hpp"
#include "util.hpp"
#include "object.hpp"
#include "material.hpp"
#include "intersection.hpp"
#include "algebra/algebra.hpp"


struct vertex
{
    vec3d position;
    vec2d uv_coord;
    vec3d normal;
};


class triangle : public object {

public:
    triangle(vertex* vertices, material* mat) 
    {
        this->vertices = (vertex*) malloc(sizeof(vertex) * 3);

        this->vertices[0] = vertices[0];
        this->vertices[1] = vertices[1];
        this->vertices[2] = vertices[2];

        p0p1 = vertices[1].position - vertices[0].position;
        p0p2 = vertices[2].position - vertices[0].position;

        this->mat = mat;
    }

    triangle(vertex v0, vertex v1, vertex v2, material* mat)
    {
        this->vertices = (vertex*) malloc(sizeof(vertex) * 3);

        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;

        p0p1 = vertices[1].position - vertices[0].position;
        p0p2 = vertices[2].position - vertices[0].position;

        this->mat = mat;
    }

    triangle(vec3d p0, vec3d p1, vec3d p2, material* mat)
    {
        this->vertices = (vertex*) malloc(sizeof(vertex) * 3);

        vertex v0, v1, v2;

        v0.position = p0;
        v1.position = p1;
        v2.position = p2;

        v0.uv_coord = vec2d(p0.x(), p0.y());
        v1.uv_coord = vec2d(p1.x(), p1.y());
        v2.uv_coord = vec2d(p2.x(), p2.y());

        v0.normal = (p1 - p0).cross(p2 - p0).normalize();
        v1.normal = (p2 - p1).cross(p0 - p1).normalize();
        v2.normal = (p0 - p2).cross(p1 - p2).normalize();

        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;

        p0p1 = vertices[1].position - vertices[0].position;
        p0p2 = vertices[2].position - vertices[0].position;

        this->mat = mat;
    }
 
    virtual ~triangle() override
    {
        free(vertices);
        free(mat);
        vertices = nullptr;
        mat = nullptr;
    }

    virtual bool intersect(const ray& r, intersection& crossover) const override
    {

        vec3d q = r.direction.cross(p0p2);
        double a = p0p1.dot(q);
        if (fabs(a) < epsilon) {
            return false;
        }

        double f = 1.0 / a;
        vec3d s = r.origin - vertices[0].position;
        double u = f * s.dot(q);
        if (u < 0) {
            return false;
        }

        vec3d k = s.cross(p0p1);
        double v = f * r.direction.dot(k);
        if (v < 0 || u + v > 1) {
            return false;
        }

        double t = f * p0p2.dot(k);
        if (t < r.t_min || t > r.t_max) {
            return false;
        }

        crossover.t = t;
        crossover.position = r.at(t);
        crossover.uv_coord = interpret_uv(u, v);
        vec3d outward_normal = interpret_normal(u, v);
        crossover.set_face_normal(r, outward_normal);
        crossover.mat = mat;
        return true;
    }

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        double x_min = std::min({vertices[0].position.x(), vertices[1].position.x(), vertices[2].position.x()}) - epsilon;
        double y_min = std::min({vertices[0].position.y(), vertices[1].position.y(), vertices[2].position.y()}) - epsilon;
        double z_min = std::min({vertices[0].position.z(), vertices[1].position.z(), vertices[2].position.z()}) - epsilon;

        double x_max = std::max({vertices[0].position.x(), vertices[1].position.x(), vertices[2].position.x()}) + epsilon;
        double y_max = std::max({vertices[0].position.y(), vertices[1].position.y(), vertices[2].position.y()}) + epsilon;
        double z_max = std::max({vertices[0].position.z(), vertices[1].position.z(), vertices[2].position.z()}) + epsilon;

        bbox = aabb(vec3d(x_min, y_min, z_min), vec3d(x_max, y_max, z_max));
        return true;
    }

    virtual double pdf(const vec3d& origin, const vec3d& value) const override
    {
        intersection crossover;
        if (!this->intersect(ray(origin, value), crossover)) {
            return 0;
        }

        double area = p0p1.cross(p0p2).length() / 2.0;
        double distance_sqr = crossover.t * crossover.t * value.length() * value.length();
        double cosine = fabs(value.dot(crossover.normal) / value.length());
        return distance_sqr / (cosine * area);
    }

    virtual vec3d rand(const vec3d& origin) const override
    {
        vec3d point;
        double u = random_double();
        double v = random_double();

        if (u + v > 1) {
            point = vertices[0].position + (1 - u) * p0p1 + (1 - v) * p0p2;
        }
        else {
            point = vertices[0].position + u * p0p1 + v * p0p2;
        }

        return point - origin;
    }

private:
    vec3d p0p1, p0p2;
    vertex* vertices;
    material* mat;

    vec2d interpret_uv(double u, double v) const
    {
        return u * vertices[1].uv_coord + v * vertices[2].uv_coord + (1 - u - v) * vertices[0].uv_coord;
    }

    vec3d interpret_normal(double u, double v) const
    {
        return u * vertices[1].normal + v * vertices[2].normal + (1 - u - v) * vertices[0].normal;
    }
};
