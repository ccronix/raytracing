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
        vec3d next = crossover.normal + random_shpere().normalize();
        if (near_zero(next)) {
            next = crossover.normal;
        }
        scatter = ray(crossover.position, next, r.time());
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
        scatter = ray(crossover.position, next + fuzz, r.time());
        attenuation = albedo;
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


class bvh : public object{

public:
    object* left;
    object* right;
    aabb bbox;

    bvh() {}

    bvh(const scene& scn, double start, double end) : bvh(scn.objects, 0, scn.objects.size(), start, end) {}

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
        double aperture, 
        double focus_distance, 
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


scene random_scene()
{
    scene scn;
    material* ground_mat = new lambertian(vec3d(0.5, 0.5, 0.5));
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

    scene scn = random_scene();
    // scn.add(left_ball);
    // scn.add(right_ball);
    // scn.add(small_ball);
    // scn.add(large_ball);
    // camera cam = camera(vec3d(3, 3, 2), vec3d(0, 0, -1), vec3d(0, 1, 0), 45, 1.78, 2.0, 5.2);
    camera cam = camera(vec3d(13, 2, 3), vec3d(0, 0, 0), vec3d(0, 1, 0), 30, 1.78, 0.1, 10, 0, 1);

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] start render...\n");
    clock_t start = clock();

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
    printf("\n[INFO] render done, using %f sec.\n", float(clock() - start) / CLOCKS_PER_SEC);
    free(data);
    data = NULL;
}


int main(int argc, char* argv[])
{
    int width = 1920, height = 1080;
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/output.png", width, height);
    return 0;
}
