#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <omp.h>
#include <vector>
#include <limits>
#include <algorithm>

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include "algebra/algebra.hpp"
#include "stb/stb_image.hpp"
#include "stb/stb_image_write.hpp"

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.141592653589;


bool near_zero(vec3d v)
{
    double e = 1e-8;
    return (fabs(v.x() < e)) && (fabs(v.y() < e)) && (fabs(v.z() < e));
}


double degree_to_arc(double degree)
{
    return degree * pi / 180;
}


double random()
{
    return rand() / double(RAND_MAX);
}


double random(double min, double max)
{
    return min + (max - min) * random();
}


int random_int(int min, int max)
{
    return int(random(min, max + 1));
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


vec3d random_disk()
{
    while (true) {
        vec3d point = vec3d(random(-1, 1), random(-1, 1), 0);
        double length = point.length();
        if (length * length >= 1) {
            continue;
        }
        return point;
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


vec3d refract(const vec3d& input, const vec3d& normal, double refract_ratio)
{
    double cos_theta = fmin(-input.dot(normal), 1.0);
    vec3d out_perp = refract_ratio * (input + cos_theta * normal);
    vec3d out_parallel = -sqrt(fabs(1.0 - out_perp.length() * out_perp.length())) * normal;
    return out_perp + out_parallel;
}


class ray {

public:
    ray () {}
    ray (vec3d orig, vec3d dir, double tm=0) 
    {
        this->orig = orig;
        this->dir = dir;
        this->tm = tm;
    }

    vec3d origin() const { return orig; }

    vec3d direction() const { return dir; }

    double time() const { return tm; }

    vec3d at(double t) const { return orig + t * dir; }

private:
    vec3d orig;
    vec3d dir;
    double tm;
};


class material;


class intersection {

public:
    vec3d position;
    vec3d normal;
    vec2d uv_coord;
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


class texture {

public:
    virtual ~texture() {}
    virtual vec3d color(const vec3d& position, const vec2d& uv_coord) const = 0;
};


class constant : public texture {

public:
    vec3d base_color;

    constant() {}

    constant(vec3d base_color) { this->base_color = base_color; }

    constant(double r, double g, double b) { this->base_color = vec3d(r, g, b); }

    virtual vec3d color(const vec3d& position, const vec2d& uv_coord) const override
    {
        return base_color;
    }
};


class checker : public texture {

public:
    texture* even = new constant(0, 0, 0);
    texture* odds = new constant(1, 1, 1);

    checker() {}

    virtual ~checker() override
    {
        if (even != nullptr) delete even;
        if (odds != nullptr) delete odds;
    }

    checker(texture* even, texture* odds)
    {
        this->even = even;
        this->odds = odds;
    }

    checker(const vec3d& even_color, const vec3d odds_color)
    {
        even = new constant(even_color);
        odds = new constant(odds_color);
    }

    virtual vec3d color(const vec3d& position, const vec2d& uv_coord) const override
    {
        double sine = sin(10 * position.x()) * sin(10 * position.y()) * sin(10 * position.z());
        if (sine < 0) {
            return even->color(position, uv_coord);
        }
        else {
            return odds->color(position, uv_coord);
        }
    }
};


class image : public texture {

public:
    image() {}

    image(const char* path)
    {
        this->path = path;
        data = stbi_load(path, &width, &height, &channels, 0);
        if (!data) {
            fprintf(stderr, "[ERROR] can not load image: %s\n", path);
        }
    }

    virtual ~image() override { stbi_image_free(data); }

    virtual vec3d color(const vec3d& position, const vec2d& uv_coord) const override
    {
        if (data == nullptr) {
            return vec3d(0, 1, 1);
        }

        double u = clamp(uv_coord.u(), 0, 1);
        double v = 1.0 - clamp(uv_coord.v(), 0, 1);

        int x = int(u * width);
        int y = int(v * height);

        if (x >= width ) {
            x = width - 1;
        }
        if (y >= height) {
            y = height - 1;
        }

        const double scale = 1.0 / 255;
        int index = y * width * channels + x * channels;
        unsigned char* base = data + index;
        return vec3d(base[0], base[1], base[2]) * scale;
    }

private:
    const char* path;
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
};


class material {

public:
    virtual vec3d emit(const vec3d& position, const vec2d uv_coord) const { return vec3d(0, 0, 0); }
    virtual bool shading(const ray&r, const intersection& crossover, vec3d& attenuation, ray& scatter) const = 0;
};


class lambertian : public material {

public:
    texture* base_color;

    lambertian(const vec3d& color) { base_color = new constant(color); }

    lambertian(texture* tex) { base_color = tex; }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        vec3d next = crossover.normal + random_shpere().normalize();
        scatter = ray(crossover.position, next, r.time());
        attenuation = base_color->color(crossover.position, crossover.uv_coord);
        return true;
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

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        vec3d next = reflect(r.direction().normalize(), crossover.normal);
        vec3d fuzz = roughness * random_shpere();
        scatter = ray(crossover.position, next + fuzz, r.time());
        attenuation = base_color->color(crossover.position, crossover.uv_coord);
        return scatter.direction().dot(crossover.normal) > 0;
    }
};


class dielectric : public material {

public:
    double ior;

    dielectric(double ior) { this->ior = ior; }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        attenuation = vec3d(1, 1, 1);
        double refract_ratio = crossover.frontward ? (1.0 / ior) : ior;

        vec3d next = r.direction().normalize();
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
        scatter = ray(crossover.position, referaction, r.time());
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

    virtual vec3d emit(const vec3d& position, const vec2d uv_coord) const override
    {
        return emission->color(position, uv_coord);
    }

    virtual bool shading(const ray& r, const intersection& crossover, vec3d& attenuation, ray& scatter) const override
    {
        return false;
    }
};


class aabb {

public:
    aabb() {}

    aabb(const vec3d& minimum, const vec3d& maximum)
    {
        this->minimum = minimum;
        this->maximum = maximum;
    }

    vec3d min() const { return minimum; }

    vec3d max() const { return maximum; }

    bool intersect(const ray&r, double t_min, double t_max) const
    {
        for (int i = 0; i < 3; i++) {
            double invert_d = 1.0 / r.direction()[i];
            double t0 = (minimum[i] - r.origin()[i]) * invert_d;
            double t1 = (maximum[i] - r.origin()[i]) * invert_d;
            if (invert_d < 0.0) {
                std::swap(t0, t1);
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) {
                return false;
            }
        }
        return true;
    }

    friend aabb surrounding_box(aabb bbox0, aabb bbox1)
    {
        double min_x = fmin(bbox0.min().x(), bbox1.min().x());
        double min_y = fmin(bbox0.min().y(), bbox1.min().y());
        double min_z = fmin(bbox0.min().z(), bbox1.min().z());
        vec3d minimum = vec3d(min_x, min_y, min_z);

        double max_x = fmax(bbox0.max().x(), bbox1.max().x());
        double max_y = fmax(bbox0.max().y(), bbox1.max().y());
        double max_z = fmax(bbox0.max().z(), bbox1.max().z());
        vec3d maximum = vec3d(max_x, max_y, max_z);

        return aabb(minimum, maximum);
    }

private:
    vec3d minimum;
    vec3d maximum;
};


class object {

public:
    virtual ~object() {}
    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const = 0;
    virtual bool bounding_box(double start, double end, aabb& bbox) const = 0;
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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        vec3d oc = r.origin() - center(r.time());
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
        vec3d outward_normal = (crossover.position - center(r.time())) / radius;
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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        double t = (k - r.origin().z()) / r.direction().z();
        if (t < t_min || t > t_max) {
            return false;
        }

        double x = r.origin().x() + t * r.direction().x();
        double y = r.origin().y() + t * r.direction().y();
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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        double t = (k - r.origin().y()) / r.direction().y();
        if (t < t_min || t > t_max) {
            return false;
        }

        double x = r.origin().x() + t * r.direction().x();
        double z = r.origin().z() + t * r.direction().z();
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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        double t = (k - r.origin().x()) / r.direction().x();
        if (t < t_min || t > t_max) {
            return false;
        }

        double y = r.origin().y() + t * r.direction().y();
        double z = r.origin().z() + t * r.direction().z();
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

    virtual bool bounding_box(double start, double end, aabb& bbox) const override
    {
        if (objects.empty()) {
            return false;
        }

        aabb temp_bbox;
        bool first = false;
        for (auto& object : objects) {
            if (!object->bounding_box(start, end, temp_bbox)) {
                return false;
            }
            bbox = first ? temp_bbox : surrounding_box(bbox, temp_bbox);
            first = false;
        }
        return true;
    }
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

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        return sides.intersect(r, t_min, t_max, crossover);
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


class bvh : public object {

public:
    object* left;
    object* right;
    aabb bbox;

    bvh() {}

    bvh(const scene& scn, double start=0, double end=0) : bvh(scn.objects, 0, scn.objects.size(), start, end) {}

    bvh(const std::vector<object*>& sources, int first, int last, double start, double end)
    {
        std::vector<object*> objects = sources;
        int axis = random_int(0, 2);
        auto function = (axis == 0) ? comparex : (axis == 1) ? comparey : comparez;
        int span = last - first;

        if (span == 1) {
            right = objects[first];
            left = right;
        }
        else if (span == 2) {
            if (function(objects[first], objects[first + 1])) {
                left = objects[first];
                right = objects[first + 1];
            }
            else {
                left = objects[first + 1];
                right = objects[first];
            }
        }
        else {
            std::sort(objects.begin() + first, objects.begin() + last, function);
            int middle = first + span / 2;
            left = new bvh(objects, first, middle, start, end);
            right = new bvh(objects, middle, last, start, end);
        }

        aabb bbox_left, bbox_right;
        if (!left->bounding_box(start, end, bbox_left) || !right->bounding_box(start, end, bbox_right)) {
            fprintf(stderr, "[ERROR] no bounding box in bvh constructor.");
        }
        bbox = surrounding_box(bbox_left, bbox_right);
    }

    virtual bool intersect(const ray& r, double t_min, double t_max, intersection& crossover) const override
    {
        if (!bbox.intersect(r, t_min, t_max)) {
            return false;
        }

        bool hit_left = left->intersect(r, t_min, t_max, crossover);
        double t_max_right = hit_left ? crossover.t : t_max;
        bool hit_right = right->intersect(r, t_min, t_max_right, crossover);
        return hit_left || hit_right;
    }

    virtual bool bounding_box(double start, double end, aabb& output) const override
    {
        output = bbox;
        return true;
    }

    static bool compare(const object* a, const object* b, int axis)
    {
        aabb bbox_a, bbox_b;
        if (!a->bounding_box(0, 0, bbox_a) || !b->bounding_box(0, 0, bbox_b)) {
            fprintf(stderr, "[ERROR] no bounding box in bvh constructor.");
        }
        return bbox_a.min()[axis] < bbox_b.min()[axis];
    }

    static bool comparex(const object* a, const object* b)
    {
        return compare(a, b, 0);
    }

    static bool comparey(const object* a, const object* b)
    {
        return compare(a, b, 1);
    }

    static bool comparez(const object* a, const object* b)
    {
        return compare(a, b, 2);
    }
};


class camera {

public:
    vec3d position;
    vec3d lookat;
    vec3d up;
    double fov;
    double aspect_ratio;
    double aperture;
    double focus_distance;

    camera() {}

    camera(
        vec3d position, 
        vec3d lookat, 
        vec3d up, 
        double fov, 
        double aspect_ratio, 
        double aperture=0, 
        double focus_distance=1, 
        double shutter_start=0, 
        double shutter_end=0
    )
    {
        this->position = position;
        this->lookat = lookat;
        this->up = up;
        this->fov = fov;
        this->aspect_ratio = aspect_ratio;
        this->aperture = aperture;
        this->focus_distance = focus_distance;
        this->shutter_start = shutter_start;
        this->shutter_end = shutter_end;

        double arc_fov = degree_to_arc(fov);
        double sensor_h = tan(arc_fov / 2) * 2;
        double sensor_w = sensor_h * aspect_ratio;

        w = (position - lookat).normalize();
        u = up.cross(w).normalize();
        v = w.cross(u);

        horizontal = focus_distance * sensor_w * u;
        vertical = focus_distance * sensor_h * v;
        left_bottom = position - horizontal / 2 - vertical / 2 - focus_distance * w;
        lens_radius = aperture / 2;
    }

    ray emit(double x, double y) const
    {
        vec3d defocus = lens_radius * random_disk();
        vec3d offset = u * defocus.x() + v * defocus.y();
        vec3d direction = left_bottom + x * horizontal + y * vertical - position - offset;
        return ray(position + offset, direction, random(shutter_start, shutter_end));
    }

private:
    vec3d left_bottom;
    vec3d horizontal;
    vec3d vertical;
    vec3d u, v, w;
    double lens_radius;
    double shutter_start;
    double shutter_end;
};


vec3d trace(const scene& scn, const ray& r, int depth)
{
    if (depth <= 0) {
        return vec3d(0, 0, 0);
    }

    intersection crossover;
    if (!scn.intersect(r, 0.001, infinity, crossover)) {
        return vec3d(0, 0, 0);
    }

    ray next_r;
    vec3d attenuation;
    vec3d emission = crossover.mat->emit(crossover.position, crossover.uv_coord);
    if (!crossover.mat->shading(r, crossover, attenuation, next_r)) {
        return emission;
    }
    return emission + attenuation * trace(scn, next_r, depth - 1);
}


scene random_scene()
{
    scene scn;
    texture* checker_tex = new checker(vec3d(0, 0, 0), vec3d(1, 1, 1));
    material* ground_mat = new lambertian(checker_tex);
    scn.add(new sphere(sphere(vec3d(0, -1000, 0), 1000, ground_mat)));

    for(int i = -11; i < 11; i++) {
        for (int j = -11; j < 11; j++) {
            double choose = random();
            vec3d center = vec3d(i + 0.9 * random(), 0.2, j + 0.9 * random());

            if ((center - vec3d(4, 0.2, 0)).length() > 0.9) {
                if (choose < 0.8) {
                    vec3d albedo = random_vector() * random_vector();
                    material* sphere_mat = new lambertian(albedo);
                    vec3d center2 = center + vec3d(0, random(0, 0.5), 0);
                    scn.add(new msphere(center, center2, 0, 1, 0.2, sphere_mat));
                }
                else if (choose < 0.95)
                {
                    vec3d albedo = random_vector(0.5, 1);
                    double roughness = random(0, 0.5);
                    material* sphere_mat = new metal(albedo, roughness);
                    scn.add(new sphere(center, 0.2, sphere_mat));
                }
                else {
                    material* sphere_mat = new dielectric(1.5);
                    scn.add(new sphere(center, 0.2, sphere_mat));
                }
            }
        }
    }
    material* glass = new dielectric(1.5);
    scn.add(new sphere(vec3d(0, 1, 0), 1.0, glass));

    material* diffuse = new lambertian(vec3d(0.4, 0.2, 0.1));
    scn.add(new sphere(vec3d(-4, 1, 0), 1.0, diffuse));

    material* gold = new metal(vec3d(0.7, 0.6, 0.5), 0);
    scn.add(new sphere(vec3d(4, 1, 0), 1.0, gold));
    bvh* bvh_scn = new bvh(scn, 0, 1);
    return scene(bvh_scn);
    // return scn;
}


scene cornell_box()
{
    scene scn;

    material* red_diffuse = new lambertian(vec3d(0.65, 0.05, 0.05));
    material* green_diffuse = new lambertian(vec3d(0.12, 0.45, 0.15));
    material* white_diffuse = new lambertian(vec3d(0.75, 0.75, 0.75));
    material* area_light = new light(vec3d(30, 30, 30));

    scn.add(new planeyz(0, 555, 0, 555, 555, green_diffuse));
    scn.add(new planeyz(0, 555, 0, 555, 0, red_diffuse));
    scn.add(new planexz(0, 555, 0, 555, 0, white_diffuse));
    scn.add(new planexz(0, 555, 0, 555, 555, white_diffuse));
    scn.add(new planexy(0, 555, 0, 555, 555, white_diffuse));

    scn.add(new box(vec3d(130, 0, 65), vec3d(295, 165, 230), white_diffuse));
    scn.add(new box(vec3d(265, 0, 295), vec3d(430, 330, 460), white_diffuse));

    scn.add(new planexz(213, 343, 227, 332, 554, area_light));
    bvh* bvh_scn = new bvh(scn, 0, 1);
    return scene(bvh_scn);
    // return scn;
}


void render_image(const char* path, int width, int height)
{
    // material* blue_diffuse = new lambertian(vec3d(0.5, 0.5, 0.5));
    // material* grey_diffuse = new lambertian(vec3d(0.5, 0.4, 0.3));
    // material* glass = new dielectric(1.5);
    // material* yellow_metal = new metal(vec3d(0.8, 0.6, 0.2), 0.2);

    // object* right_ball = new sphere(sphere(vec3d(1, 0, -1), 0.5, glass));
    // object* left_ball = new sphere(sphere(vec3d(-1, 0, -1), 0.5, yellow_metal));
    // object* small_ball = new sphere(sphere(vec3d(0, 0, -1), 0.5, blue_diffuse));
    // object* large_ball = new sphere(sphere(vec3d(0, -100.5, -1), 100, grey_diffuse));

    scene scn = cornell_box();
    // scn.add(left_ball);
    // scn.add(right_ball);
    // scn.add(small_ball);
    // scn.add(large_ball);
    // camera cam = camera(vec3d(3, 3, 2), vec3d(0, 0, -1), vec3d(0, 1, 0), 45, 1.78, 2.0, 5.2);
    // camera cam = camera(vec3d(13, 2, 3), vec3d(0, 0, 0), vec3d(0, 1, 0), 30, 1.78, 0.1, 10, 0, 1);
    camera cam = camera(vec3d(278, 278, -800), vec3d(278, 278, 0), vec3d(0, 1, 0), 40, 1, 0, 1, 0, 1);

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] start render...\n");
    clock_t start = clock();

    for (int i = 0; i < height; i++) {
        printf("\rRendering (%d spp) %5.2f%%", spp, 100. * i / (height - 1));
    #pragma omp parallel for
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
    printf("\n[INFO] render done, using %f sec.\n", float(clock() - start) / CLOCKS_PER_SEC);
    free(data);
    data = NULL;
}


int main(int argc, char* argv[])
{
    int width = 1024, height = 1024;
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/output.png", width, height);
    return 0;
}
