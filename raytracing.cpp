#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <omp.h>
#include <vector>
#include <memory>
#include <limits>
#include <cstdio>
#include <cstdlib>

#include "algebra/algebra.hpp"
#include "stb/stb_image_write.hpp"

const float infinity = std::numeric_limits<float>::infinity();


class ray {

public:
    ray () {}
    ray (vec3f orig, vec3f dir) 
    {
        this->orig = orig;
        this->dir = dir;
    }

    vec3f origin() const { return orig; }

    vec3f direction() const { return dir; }

    vec3f at(float t) const { return orig + t * dir; }

private:
    vec3f orig;
    vec3f dir;
};


class intersection {

public:
    vec3f point;
    vec3f normal;
    float t;
    bool frontward;

    intersection() {}

    void set_face_normal(const ray& r, const vec3f& outward_normal)
    {
        frontward = r.direction().dot(outward_normal) < 0;
        normal = frontward ? outward_normal : -outward_normal;
    }
};


class object {

public:
    virtual bool intersect(const ray& r, float t_min, float t_max, intersection& crossover) const = 0;
};


class sphere : public object {

public:
    vec3f center;
    float radius;

    sphere() {}

    sphere(vec3f center, float radius) 
    {
        this->center = center;
        this->radius = radius;
    }

    virtual bool intersect(const ray& r, float t_min, float t_max, intersection& crossover) const override
    {
        vec3f oc = r.origin() - center;
        float a = r.direction().dot(r.direction());
        float b = 2.0 * oc.dot(r.direction());
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return false;
        }

        float root = (-b - sqrt(discriminant)) / (a * 2);
        
        if (root < t_min || root > t_max) {
            root = (-b + sqrt(discriminant)) / (a * 2);
            if (root < t_min || root > t_max) {
                return false; 
            }
        }
        crossover.t = root;
        crossover.point = r.at(root);
        vec3f outward_normal = (crossover.point - center) / radius;
        crossover.set_face_normal(r, outward_normal);
        return true;
    }
};


class scene : public object {

public:
    std::vector<std::shared_ptr<object>> objects;

    scene() {}

    scene(std::shared_ptr<object> obj) { add(obj); }

    void add(std::shared_ptr<object> obj) { objects.push_back(obj); }

    void clear() { objects.clear(); }

    virtual bool intersect(const ray& r, float t_min, float t_max, intersection& crossover) const override
    {
        intersection temp_crossover;
        bool has_intersect = false;
        float closest = t_max;

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
    vec3f origin;
    float sensor_w;
    float sensor_h;
    float focal_length;

    camera() {}

    camera(vec3f origin, float sensor_w, float sensor_h, float focal_length)
    {
        this->sensor_w = sensor_w;
        this->sensor_h = sensor_h;
        this->focal_length = focal_length;

        horizontal = vec3f(sensor_w, 0, 0);
        vertical = vec3f(0, sensor_h, 0);
        left_bottom = origin - horizontal / 2 - vertical / 2 - vec3f(0, 0, focal_length);
    }

    ray emit(float x, float y) const
    {
        vec3f direction = left_bottom + x * horizontal + y * vertical - origin;
        return ray(origin, direction);
    }

private:
    vec3f left_bottom;
    vec3f horizontal;
    vec3f vertical;
};


float random()
{
    return rand() / float(RAND_MAX);
}


float random(float min, float max)
{
    return min + (max - min) * random();
}


vec3f random_vector()
{
    return vec3f(random(), random(), random());
}


vec3f random_vector(float min, float max)
{
    return vec3f(random(min, max), random(min, max), random(min, max));
}


vec3f random_shpere()
{
    while (true) {
        vec3f point = random_vector(-1, 1);
        float length = point.length();
        if (length * length >= 1) {
            continue;
        }
        return point;
    }
}


vec3f gamma(vec3f v, float power)
{
    float x = pow(v.x(), power);
    float y = pow(v.y(), power);
    float z = pow(v.z(), power);
    return vec3f(x, y, z);
}


float clamp(float value, float min, float max)
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


vec3f shading_ray(const scene& scn, const ray& r, int depth)
{
    if (depth <= 0) {
        return vec3f(0, 0, 0);
    }

    intersection crossover;
    if (scn.intersect(r, 0, infinity, crossover)) {
        vec3f next = crossover.point + crossover.normal + random_shpere();
        ray next_r =  ray(crossover.point, next - crossover.point);
        return 0.5 * shading_ray(scn, next_r, depth - 1);
    }
    float t = 0.5 * (r.direction().normalize().y() + 1.0);
    return (1.0 - t) * vec3f(1.0, 1.0, 1.0) + t * vec3f(0.5, 0.7, 1.0);
}


void render_image(const char* path, int width, int height)
{
    scene scn;
    scn.add(std::make_shared<sphere>(sphere(vec3f(0, 0, -1), 0.5)));
    scn.add(std::make_shared<sphere>(sphere(vec3f(0, -100.5, -1), 100)));
    camera cam = camera(vec3f(0, 0, 0), 3.56, 2.0, 1.0);

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] starting write test image...\n");

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < height; i++) {
        printf("\rRendering (%d spp) %5.2f%%", spp, 100. * i / (height - 1));
        for (int j = 0; j < width; j++) {
            vec3f color = vec3f(0, 0, 0);
            for (int k = 0; k < spp; k++) {
                float x = float(j + random()) / (width - 1);
                float y = float(i + random()) / (height - 1);
                ray r = cam.emit(x, y);
                color += shading_ray(scn, r, max_depth);
            }
            color /= spp;
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
