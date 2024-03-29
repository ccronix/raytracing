#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <omp.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include "algebra/algebra.hpp"
#include "stb/stb_image_write.hpp"

#include "ray.hpp"
#include "bvh.hpp"
#include "util.hpp"
#include "demo.hpp"
#include "group.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "loader.hpp"
#include "object.hpp"
#include "texture.hpp"
#include "material.hpp"
#include "polygon.hpp"
#include "primitive.hpp"
#include "transform.hpp"
#include "intersection.hpp"


vec3d trace(const scene& scn, const ray& wi,  int depth)
{
    if (depth <= 0) {
        return vec3d(0, 0, 0);
    }

    intersection crossover;
    if (!scn.objects.intersect(wi, crossover)) {
        return vec3d(0, 0, 0);
    }

    ray wo;
    vec3d attenuation;
    bool is_spec = false;
    double pdf, mat_pdf;
    vec3d emission = crossover.mat->emit(wi, crossover, crossover.position, crossover.uv_coord);
    if (!crossover.mat->shading(wi, crossover, wo, attenuation, is_spec)) {
        return emission;
    }

    if (true) {
        if (is_spec) {
            mat_pdf = crossover.mat->pdf(wi, crossover, wo);
            pdf =  mat_pdf;
        }
        else {
            if (random_double() < 0.5) {
                wo = ray(crossover.position, scn.lights->rand(crossover.position), wi.time, wi.t_min, wi.t_max);
            }
            mat_pdf = crossover.mat->pdf(wi, crossover, wo);
            pdf = 0.5 * (scn.lights->pdf(crossover.position, wo.direction) + mat_pdf);
        }
    }
    else {
        wo = ray(crossover.position, crossover.mat->sample(wi, crossover), wi.time, wi.t_min, wi.t_max);
        mat_pdf = crossover.mat->pdf(wi, crossover, wo);
        pdf = mat_pdf;
    }

    return emission + attenuation * mat_pdf * trace(scn, wo, depth - 1) / pdf;
}


void render_image(const char* path, int width, int height)
{
    // loader obj_loader = loader("C:/Users/Cronix/Documents/cronix_dev/raytracing/object/fruit/fruit.obj");

    camera cam =  camera(vec3d(278, 278, -800), vec3d(278, 278, 0), vec3d(0, 1, 0), 40, 1, 0, 1, 0, 1);
    // camera cam = camera(vec3d(0, 10, 40), vec3d(0, 10, 0), vec3d(0, 1, 0), 40, 1.78, 0, 1, 0, 1);
    // camera cam = camera(vec3d(3, 2, -5), vec3d(1.5, 0.2, -2), vec3d(0, 1, 0), 40, 1.78, 0, 1, 0, 1);
    // camera cam = camera(vec3d(-0.5, 1, 4), vec3d(0.5, 0, 0), vec3d(0, 1, 0), 55, 1.78, 0, 1, 0, 1);
    object* light = new flip(new planexz(213, 343, 227, 332, 554, new emissive(vec3d(30, 30, 30))));

    scene scn = scene(1024, 1024);
    scn.add_object(cornell_box());
    scn.add_light(light);
    scn.set_camera(cam);
    scn.build_bvh();

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] start render...\n");
    clock_t start = clock();

    for (int i = 0; i < height; i++) {
        printf("\rRendering (%d spp) %5.2f%%", spp, 100. * i / (height - 1));
    #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < width; j++) {
            vec3d color = vec3d(0, 0, 0);
            for (int k = 0; k < spp; k++) {
                double x = double(j + random_double()) / (width - 1);
                double y = double(i + random_double()) / (height - 1);

                ray r = scn.render_camera.emit(x, y);
                vec3d sample = trace(scn, r, max_depth);

                if (is_nan(sample) || is_infinity(sample)) {
                    sample = vec3d(0, 0, 0);
                }

                color += sample;
            }

            color /= spp;
            color = gamma(color, 0.45);
            color = clamp(color, 0, 1);
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
    render_image("./output.png", width, height);
    return 0;
}
