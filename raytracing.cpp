#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <omp.h>
#include <vector>
#include <limits>
#include <cstdio>
#include <cstdlib>

#include "algebra/algebra.hpp"
#include "stb/stb_image_write.hpp"

const double infinity = std::numeric_limits<double>::infinity();


bool near_zero(vec3d v)
{
    double e = 1e-8;
    return (fabs(v.x() < e)) && (fabs(v.y() < e)) && (fabs(v.z() < e));
}


double random()
{
    return rand() / double(RAND_MAX);
}


double random(double min, double max)
{
    return min + (max - min) * random();
}


vec3d random_vector()
{
    return vec3d(random(), random(), random());
}


vec3d random_vector(double min, double max)
{
    return vec3d(random(min, max), random(min, max), random(min, max));
}


vec3d random_shpere()
{
    while (true) {
        vec3d point = random_vector(-1, 1);
        double length = point.length();
        if (length * length >= 1) {
            continue;
        }
        return point.normalize();
    }
}


vec3d random_hemisphere(const vec3d& normal)
{
    vec3d value = random_shpere();
    if (value.dot(normal) > 0.0) {
        return value;
    }
    else {
        return -value;
    }
}


vec3d gamma(vec3d value, double power)
{
    double x = pow(value.x(), power);
    double y = pow(value.y(), power);
    double z = pow(value.z(), power);
    return vec3d(x, y, z);
}


double clamp(double value, double min, double max)
{
    if (value < min) {
        return min;
    }
    else if (value > max) {
        return max;
    }
    else {
        return value;
    }
}


vec3d clamp(vec3d value, double min, double max)
{
    double x = clamp(value.x(), min, max);
    double y = clamp(value.y(), min, max);
    double z = clamp(value.z(), min, max);
    return vec3d(x, y, z);
}


vec3d reflect(const vec3d& input, const vec3d& normal)
{
    return input - 2 * input.dot(normal) * normal;
}


class ray {

public:
    ray () {}
    ray (vec3d orig, vec3d dir) 
    {
        this->orig = orig;
        this->dir = dir;
    }

    vec3d origin() const { return orig; }

    vec3d direction() const { return dir; }

    vec3d at(double t) const { return orig + t * dir; }

private:
    vec3d orig;
    vec3d dir;
};


class material;


class intersection {

public:
    vec3d position;
    vec3d normal;
    double t;
    material* mat;
    bool frontward;

    intersection() {}

    void set_face_normal(const ray& r, const vec3d& outward_normal)
    {
        frontward = r.direction().dot(outward_normal) < 0;
        normal = frontward ? outward_normal : -outward_normal;
    }
};


class material {

public:
    virtual bool shading(const ray&r, const intersection& crossover, vec3d& attenuation, ray& scatter) const = 0;
};


class lambertian : public material {

public:
    vec3d albedo;

    lambertian(const vec3d& color) { albedo = color; }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        vec3d next = crossover.normal + random_vector();
        if (near_zero(next)) {
            next = crossover.normal;
        }
        scatter = ray(crossover.position, next);
        attenuation = albedo;
        return true;
    }
};


class metal : public material {

public:
    vec3d albedo;
    double roughness;

    metal(const vec3d& color, double roughness) 
    {
        albedo = color;
        this->roughness = roughness < 1 ? roughness : 1;
    }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        vec3d next = reflect(r.direction().normalize(), crossover.normal);
        vec3d fuzz = roughness * random_shpere();
        scatter = ray(crossover.position, next + fuzz);
        attenuation = albedo;
        return scatter.direction().dot(crossover.normal) > 0;
    }
};


class object {

public:
    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const = 0;
};


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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        vec3d oc = r.origin() - center;
        double a = r.direction().dot(r.direction());
        double b = 2.0 * oc.dot(r.direction());
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return false;
        }

        double root = (-b - sqrt(discriminant)) / (a * 2);
        
        if (root < t_min || root > t_max) {
            root = (-b + sqrt(discriminant)) / (a * 2);
            if (root < t_min || root > t_max) {
                return false; 
            }
        }
        crossover.t = root;
        crossover.position = r.at(root);
        vec3d outward_normal = (crossover.position - center) / radius;
        crossover.set_face_normal(r, outward_normal);
        crossover.mat = mat;
        return true;
    }
};


class scene : public object {

public:
    std::vector<object*> objects;

    scene() {}

    scene(object* obj) { add(obj); }

    void add(object* obj) { objects.push_back(obj); }

    void clear() { objects.clear(); }

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        intersection temp_crossover;
        bool has_intersect = false;
        double closest = t_max;

        for (auto& object : objects) {
            if (object->intersect(r, t_min, closest, temp_crossover)) {
                has_intersect = true;
                closest = temp_crossover.t;
                crossover = temp_crossover;
            }
        }
        return has_intersect;
    }
};


class camera {

public:
    vec3d origin;
    double sensor_w;
    double sensor_h;
    double focal_length;

    camera() {}

    camera(vec3d origin, double sensor_w, double sensor_h, double focal_length)
    {
        this->sensor_w = sensor_w;
        this->sensor_h = sensor_h;
        this->focal_length = focal_length;

        horizontal = vec3d(sensor_w, 0, 0);
        vertical = vec3d(0, sensor_h, 0);
        left_bottom = origin - horizontal / 2 - vertical / 2 - vec3d(0, 0, focal_length);
    }

    ray emit(double x, double y) const
    {
        vec3d direction = left_bottom + x * horizontal + y * vertical - origin;
        return ray(origin, direction);
    }

private:
    vec3d left_bottom;
    vec3d horizontal;
    vec3d vertical;
};


vec3d trace(const scene& scn, const ray& r, int depth)
{
    if (depth <= 0) {
        return vec3d(0, 0, 0);
    }

    intersection crossover;
    if (scn.intersect(r, 0.001, infinity, crossover)) {
        ray next_r;
        vec3d attenuation;
        if (crossover.mat->shading(r, crossover, attenuation, next_r)) {
            return attenuation * trace(scn, next_r, depth - 1);
        }
        else {
            return vec3d(0, 0, 0);
        }
    }
    double t = 0.5 * (r.direction().normalize().y() + 1.0);
    return (1.0 - t) * vec3d(1.0, 1.0, 1.0) + t * vec3d(0.5, 0.7, 1.0);
}


void render_image(const char* path, int width, int height)
{
    material* grey_diffuse = new lambertian(vec3d(0.5, 0.5, 0.5));
    material* silver_metal = new metal(vec3d(0.8, 0.8, 0.8), 0.1);
    material* yellow_metal = new metal(vec3d(0.8, 0.6, 0.2), 0.2);

    object* left_ball = new sphere(sphere(vec3d(1, 0, -1), 0.5, silver_metal));
    object* right_ball = new sphere(sphere(vec3d(-1, 0, -1), 0.5, yellow_metal));
    object* small_ball = new sphere(sphere(vec3d(0, 0, -1), 0.5, grey_diffuse));
    object* large_ball = new sphere(sphere(vec3d(0, -100.5, -1), 100, grey_diffuse));

    scene scn;
    scn.add(left_ball);
    scn.add(right_ball);
    scn.add(small_ball);
    scn.add(large_ball);
    camera cam = camera(vec3d(0, 0, 0), 3.56, 2.0, 1.0);

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] starting write test image...\n");

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < height; i++) {
        printf("\rRendering (%d spp) %5.2f%%", spp, 100. * i / (height - 1));
        for (int j = 0; j < width; j++) {
            vec3d color = vec3d(0, 0, 0);
            for (int k = 0; k < spp; k++) {
                double x = double(j + random()) / (width - 1);
                double y = double(i + random()) / (height - 1);
                ray r = cam.emit(x, y);
                color += trace(scn, r, max_depth);
            }
            color /= spp;
            color = clamp(color, 0, 1);
            color = gamma(color, 0.45);
            int index = (i * width + j) * 3;
            data[index] = int(color.x() * 255);
            data[index + 1] = int(color.y() * 255);
            data[index + 2] = int(color.z() * 255);
        }
    }
    stbi_flip_vertically_on_write(true);
    stbi_write_png(path, width, height, 3, data, 0);
    printf("\n[INFO] write image done.\n");
    free(data);
    data = NULL;
}


int main(int argc, char* argv[])
{
    int width = 1920, height = 1080;
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/output.png", width, height);
    return 0;
}
