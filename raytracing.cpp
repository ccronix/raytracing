#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <omp.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include "algebra/algebra.hpp"
#include "stb/stb_image_write.hpp"

#include "ray.hpp"
#include "pdf.hpp"
#include "bvh.hpp"
#include "util.hpp"
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


vec3d trace(const scene& scn, const ray& r, object* lights,  int depth)
{
    if (depth <= 0) {
        return vec3d(0, 0, 0);
    }

    intersection crossover;
    if (!scn.intersect(r, crossover)) {
        return vec3d(0, 0, 0);
    }

    ray next_r;
    scatter scattering;
    vec3d emission = crossover.mat->emit(r, crossover, crossover.position, crossover.uv_coord);
    if (!crossover.mat->shading(r, crossover, scattering)) {
        return emission;
    }
    if (scattering.is_spec) {
        return scattering.attenuation * trace(scn, scattering.specular, lights, depth - 1);
    }

    obj_pdf* direct_pdf = new obj_pdf(lights, crossover.position);
    mix_pdf mixture_pdf = mix_pdf(direct_pdf, scattering.pdf_ptr);
    next_r = ray(crossover.position, mixture_pdf.generate(), r.time, r.t_min, r.t_max);
    double pdf_value = mixture_pdf.value(next_r.direction);

    if (scattering.pdf_ptr != nullptr) {
        delete scattering.pdf_ptr;
    }
    delete direct_pdf;

    return emission + scattering.attenuation * crossover.mat->shading_pdf(r, crossover, next_r) * trace(scn, next_r, lights, depth - 1) / pdf_value;
}


scene random_scene()
{
    scene scn;
    texture* checker_tex = new checker(vec3d(0, 0, 0), vec3d(1, 1, 1));
    material* ground_mat = new lambertian(checker_tex);
    scn.add(new sphere(sphere(vec3d(0, -1000, 0), 1000, ground_mat)));

    for(int i = -11; i < 11; i++) {
        for (int j = -11; j < 11; j++) {
            double choose = random_double();
            vec3d center = vec3d(i + 0.9 * random_double(), 0.2, j + 0.9 * random_double());

            if ((center - vec3d(4, 0.2, 0)).length() > 0.9) {
                if (choose < 0.8) {
                    vec3d albedo = random_vector() * random_vector();
                    material* sphere_mat = new lambertian(albedo);
                    vec3d center2 = center + vec3d(0, random_double(0, 0.5), 0);
                    scn.add(new msphere(center, center2, 0, 1, 0.2, sphere_mat));
                }
                else if (choose < 0.95)
                {
                    vec3d albedo = random_vector(0.5, 1);
                    double roughness = random_double(0, 0.5);
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
    material* mirror = new metal(vec3d(1, 1, 1), 0);

    scn.add(new planeyz(0, 555, 0, 555, 555, green_diffuse));
    scn.add(new planeyz(0, 555, 0, 555, 0, red_diffuse));
    scn.add(new planexz(0, 555, 0, 555, 0, white_diffuse));
    scn.add(new planexz(0, 555, 0, 555, 555, white_diffuse));
    scn.add(new planexy(0, 555, 0, 555, 555, white_diffuse));

    object* box1 = new box(vec3d(0, 0, 0), vec3d(165, 330, 165), mirror);
    box1 = new rotatey(box1, 15);
    box1 = new translate(box1, vec3d(265, 0, 295));;
    scn.add(box1);

    object* box2 = new box(vec3d(0, 0, 0), vec3d(165, 165, 165), white_diffuse);
    box2 = new rotatey(box2, -18);
    box2 = new translate(box2, vec3d(130, 0, 65));;
    scn.add(box2);

    // scn.add(new flip(new planexz(213, 343, 227, 332, 554, area_light)));
    // bvh* bvh_scn = new bvh(scn, 0, 1);
    // return scene(bvh_scn);
    return scn;
}


scene all_feature_test()
{
    scene scn;
    scene ground_boxes;
    material* white_mat = new lambertian(vec3d(0.73, 0.73, 0.73));
    material* ground_mat = new lambertian(vec3d(0.48, 0.83, 0.53));
    material* area_light = new light(vec3d(7, 7, 7));
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            double w = 100;
            double x0 = -1000 + i * w;
            double z0 = -1000 + j * w;
            double y0 = 0;
            double x1 = x0 + w;
            double z1 = z0 + w;
            double y1 = random_double(1, 101);
            ground_boxes.add(new box(vec3d(x0, y0, z0), vec3d(x1, y1, z1), ground_mat));
        }
    }

    scn.add(ground_boxes);
    scn.add(new planexz(123, 423, 147, 412, 554, area_light));

    vec3d center_start = vec3d(400, 400, 200);
    vec3d center_end = center_start + vec3d(100, 0, 0);
    material* motion_ball_mat = new lambertian(vec3d(0.7, 0.3, 0.1));
    scn.add(new msphere(center_start, center_end, 0, 1, 50, motion_ball_mat));

    scn.add(new sphere(vec3d(260, 150, 45), 50, new dielectric(1.5)));
    scn.add(new sphere(vec3d(0, 150, 145), 50, new metal(vec3d(0.8, 0.8, 0.9), 1)));

    object* background = new sphere(vec3d(0, 0, 0), 5000, new dielectric(1.5));
    scn.add(new volume(background, 0.0002, vec3d(1, 1, 1)));

    background = new sphere(vec3d(360,150,145), 70, new dielectric(1.5));
    scn.add(background);
    scn.add(new volume(background, 0.2, vec3d(0.2, 0.4, 0.9)));

    material* img_mat = new lambertian(new image("C:/Users/Cronix/Pictures/image.jpg"));
    scn.add(new sphere(vec3d(400, 200, 400), 100, img_mat));
    scn.add(new sphere(vec3d(220, 280, 300), 50, new metal(vec3d(1, 1, 1), 0)));

    scene vol_boxes;
    for (int i = 0; i < 1000; i++) {
        vol_boxes.add(new sphere(random_vector(0, 165), 10, white_mat));
    }
    scn.add(new translate(new rotatey(new bvh(vol_boxes), 15), vec3d(-100, 270, 395)));
    bvh* bvh_scn = new bvh(scn, 0, 1);
    return scene(bvh_scn);
}


void render_image(const char* path, int width, int height)
{
    scene scn;

    loader obj_loader = loader("C:/Users/Cronix/Documents/cronix_dev/raytracing/object/fruit2.obj");
    bvh* bvh_obj = new bvh(obj_loader.meshes());
    scn.add(bvh_obj);

    material* area_light = new light(vec3d(15, 15, 15));
    object* lights = new flip(new planexz(-5, 5, -5, 5, 20, area_light));
    scn.add(lights);

    camera cam = camera(vec3d(-1, 4, 2), vec3d(1, 0, -1), vec3d(0, 1, 0), 55, 1.78, 0, 1, 0, 1);
    // camera cam = camera(vec3d(278, 278, -800), vec3d(278, 278, 0), vec3d(0, 1, 0), 40, 1, 0, 1, 0, 1);
    // camera cam = camera(vec3d(478, 278, -600), vec3d(278, 278, 0), vec3d(0, 1, 0), 40, 1.78, 0, 1, 0, 1);

    int spp = 100;
    int max_depth = 10;

    unsigned char* data = (unsigned char*) malloc(width * height * sizeof(unsigned char) * 3);
    printf("[INFO] start render.../n");
    clock_t start = clock();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < height; i++) {
        printf("\rRendering (%d spp) %5.2f%%", spp, 100. * i / (height - 1));
        for (int j = 0; j < width; j++) {
            vec3d color = vec3d(0, 0, 0);
            for (int k = 0; k < spp; k++) {
                double x = double(j + random_double()) / (width - 1);
                double y = double(i + random_double()) / (height - 1);

                ray r = cam.emit(x, y);
                vec3d sample = trace(scn, r, lights, max_depth);

                if (is_nan(sample) || is_infinity(sample)) {
                    sample = vec3d(0, 0, 0);
                }

                color += sample;
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
    render_image("C:/Users/Cronix/Documents/cronix_dev/raytracing/output3.png", width, height);
    return 0;
}
