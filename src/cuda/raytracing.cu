#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "stb/stb_image_write.hpp"
#include "device_algebra/algebra.hpp"


struct ray {
    vec3d origin;
    vec3d direction;
    double time = 0;
    double t_min = 1e-3;
    double t_max = INFINITY;

    __device__ ray() {}

    __device__ ray(vec3d origin, vec3d direction, double time=0, double t_min=1e-3, double t_max=INFINITY) 
    {
        this->origin = origin;
        this->direction = direction;
        this->time = time;
        this->t_min = t_min;
        this->t_max = t_max;
    }

    __device__ vec3d at(double t) const { return origin + t * direction; }
};


struct intersection {

    vec3d position;
    vec3d normal;
    vec2d uv_coord;
    double t;
    bool frontward;

    __device__ void set_face_normal(const ray& r, const vec3d& outward_normal)
    {
        frontward = r.direction.dot(outward_normal) < 0;
        normal = frontward ? outward_normal : -outward_normal;
    }
};


class object {

public:
    __device__ virtual ~object() {}
    __device__ virtual bool intersect(const ray& r, intersection& crossover) const = 0;
};


class sphere : public object {

public:
    vec3d center;
    double radius;

    __host__ __device__ sphere() {}

    __host__ __device__ sphere(vec3d center, double radius) 
    {
        this->center = center;
        this->radius = radius;
    }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
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
        return true;
    }

private:
    __device__ static vec2d uv(const vec3d& position)
    {
        double pi = 3.141592653589;
        double theta = acos(-position.y());
        double phi = atan2(-position.z(), position.x()) + pi;
        double u = phi / (2 * pi);
        double v = theta / pi;
        return vec2d(u, v);
    }
};


class group : public object {

public:
    __host__ __device__ group() {}

    __host__ __device__ group(object* obj) { add(obj); }

    __host__ __device__ group(const group& grp) 
    { 
        objects = grp.objects; 
        count = grp.size();
    }

    __host__ __device__ group operator = (const group& grp)
    {
        objects = grp.objects;
        count = grp.size();
        return this;
    }

    __host__ __device__ void add(object* obj)
    {
        if (count == 0) {
            objects = (object**) malloc(sizeof(object*));
        }
        else {
            objects = (object**) realloc(objects, sizeof(object*) * (count + 1));
        }
        objects[count] = obj;
        count++;
    }

    __host__ __device__ void add(const group& grp)
    {
        for (int i = 0; i < grp.size(); i++) {
            object* object = grp.objects[i];
            add(object);
        }
    }

    __host__ __device__ void clear() 
    { 
        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            free(object);
            object = nullptr;
        }
        free(objects);
        objects = nullptr;
        count = 0;
    }

    __host__ __device__ int size() const { return count; }

    __host__ __device__ object** content() const { return objects; }

    __device__ virtual bool intersect(const ray& r, intersection& crossover) const override
    {
        intersection temp_crossover;
        bool has_intersect = false;
        double closest = r.t_max;

        for (int i = 0; i < count; i++) {
            object* object = objects[i];
            if (object->intersect(ray(r.origin, r.direction, r.time, r.t_min, closest), temp_crossover)) {
                has_intersect = true;
                closest = temp_crossover.t;
                crossover = temp_crossover;
            }
        }
        return has_intersect;
    }


private:
    object** objects = nullptr;
    int count = 0;
};


__global__ void trace(unsigned char* buffer, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    double r = double(i) / width;
    double g = double(j) / height;
    double b = 0.25;
    vec3d color = vec3d(r, g, b);

    int index = (j * width + i) * 3;
    buffer[index] = int(color.x() * 255);
    buffer[index + 1] = int(color.y() * 255);
    buffer[index + 2] = int(color.z() * 255);
}


__host__ void render_image(const char* path, int width, int height)
{
    int size = width * height * sizeof(unsigned char) * 3;
    unsigned char* data;
    cudaError_t error = cudaMallocManaged((void**) &data, size);
    if (error) {
        printf("[CUDA] can not allocate gpu memory: %d\n", int(error));
        cudaDeviceReset();
        exit(1);
    }

    dim3 blocks(width / 8, height / 8);
    dim3 threads(8, 8);

    printf("[CUDA] kernel launch...\n");

    trace<<<blocks, threads>>>(data, width, height);
    cudaDeviceSynchronize();

    printf("[CUDA] kernel done, writing data...\n");

    stbi_flip_vertically_on_write(true);
    stbi_write_png(path, width, height, 3, data, 0);
    printf("[INFO] write image done.\n");
    cudaFree(data);
}

__host__ int main(int argc, char* argv[])
{
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/cuda_output.png", 1920, 1080);
    return 0;
}
