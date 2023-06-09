#include "ray.hpp"
#include "bvh.hpp"
#include "util.hpp"
#include "group.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "object.hpp"
#include "texture.hpp"
#include "material.hpp"
#include "primitive.hpp"
#include "transform.hpp"
#include "intersection.hpp"

#include "algebra/algebra.hpp"


group random_scene()
{
    group grp;
    texture* checker_tex = new checker(vec3d(0, 0, 0), vec3d(1, 1, 1));
    material* ground_mat = new lambertian(checker_tex);
    grp.add(new sphere(sphere(vec3d(0, -1000, 0), 1000, ground_mat)));

    for(int i = -11; i < 11; i++) {
        for (int j = -11; j < 11; j++) {
            double choose = random_double();
            vec3d center = vec3d(i + 0.9 * random_double(), 0.2, j + 0.9 * random_double());

            if ((center - vec3d(4, 0.2, 0)).length() > 0.9) {
                if (choose < 0.8) {
                    vec3d albedo = random_vector() * random_vector();
                    material* sphere_mat = new lambertian(albedo);
                    vec3d center2 = center + vec3d(0, random_double(0, 0.5), 0);
                    grp.add(new msphere(center, center2, 0, 1, 0.2, sphere_mat));
                }
                else if (choose < 0.95)
                {
                    vec3d albedo = random_vector(0.5, 1);
                    double roughness = random_double(0, 0.5);
                    material* sphere_mat = new metal(albedo, roughness);
                    grp.add(new sphere(center, 0.2, sphere_mat));
                }
                else {
                    material* sphere_mat = new dielectric(1.5);
                    grp.add(new sphere(center, 0.2, sphere_mat));
                }
            }
        }
    }
    material* glass = new dielectric(1.5);
    grp.add(new sphere(vec3d(0, 1, 0), 1.0, glass));

    material* diffuse = new lambertian(vec3d(0.4, 0.2, 0.1));
    grp.add(new sphere(vec3d(-4, 1, 0), 1.0, diffuse));

    material* gold = new metal(vec3d(0.7, 0.6, 0.5), 0);
    grp.add(new sphere(vec3d(4, 1, 0), 1.0, gold));
    bvh* bvh_grp = new bvh(grp, 0, 1);
    return group(bvh_grp);
}


group cornell_box()
{
    group grp;

    material* red_diffuse = new lambertian(vec3d(0.65, 0.05, 0.05));
    material* green_diffuse = new lambertian(vec3d(0.12, 0.45, 0.15));
    material* white_diffuse = new lambertian(vec3d(0.75, 0.75, 0.75));
    material* area_light = new emissive(vec3d(30, 30, 30));
    material* mirror = new metal(vec3d(1, 1, 1), 0);

    grp.add(new planeyz(0, 555, 0, 555, 555, green_diffuse));
    grp.add(new planeyz(0, 555, 0, 555, 0, red_diffuse));
    grp.add(new planexz(0, 555, 0, 555, 0, white_diffuse));
    grp.add(new planexz(0, 555, 0, 555, 555, white_diffuse));
    grp.add(new planexy(0, 555, 0, 555, 555, white_diffuse));

    object* box1 = new box(vec3d(0, 0, 0), vec3d(165, 330, 165), mirror);
    box1 = new rotatey(box1, 15);
    box1 = new translate(box1, vec3d(265, 0, 295));;
    grp.add(box1);

    object* box2 = new box(vec3d(0, 0, 0), vec3d(165, 165, 165), white_diffuse);
    box2 = new rotatey(box2, -18);
    box2 = new translate(box2, vec3d(130, 0, 65));;
    grp.add(box2);
    return grp;
}


group all_feature_test()
{
    group grp;
    group ground_boxes;
    material* white_mat = new lambertian(vec3d(0.73, 0.73, 0.73));
    material* ground_mat = new lambertian(vec3d(0.48, 0.83, 0.53));
    material* area_light = new emissive(vec3d(7, 7, 7));
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

    grp.add(ground_boxes);
    grp.add(new planexz(123, 423, 147, 412, 554, area_light));

    vec3d center_start = vec3d(400, 400, 200);
    vec3d center_end = center_start + vec3d(100, 0, 0);
    material* motion_ball_mat = new lambertian(vec3d(0.7, 0.3, 0.1));
    grp.add(new msphere(center_start, center_end, 0, 1, 50, motion_ball_mat));

    grp.add(new sphere(vec3d(260, 150, 45), 50, new dielectric(1.5)));
    grp.add(new sphere(vec3d(0, 150, 145), 50, new metal(vec3d(0.8, 0.8, 0.9), 1)));

    object* background = new sphere(vec3d(0, 0, 0), 5000, new dielectric(1.5));
    grp.add(new volume(background, 0.0002, vec3d(1, 1, 1)));

    background = new sphere(vec3d(360,150,145), 70, new dielectric(1.5));
    grp.add(background);
    grp.add(new volume(background, 0.2, vec3d(0.2, 0.4, 0.9)));

    material* img_mat = new lambertian(new image("C:/Users/Cronix/Pictures/image.jpg"));
    grp.add(new sphere(vec3d(400, 200, 400), 100, img_mat));
    grp.add(new sphere(vec3d(220, 280, 300), 50, new metal(vec3d(1, 1, 1), 0)));

    group vol_boxes;
    for (int i = 0; i < 1000; i++) {
        vol_boxes.add(new sphere(random_vector(0, 165), 10, white_mat));
    }
    grp.add(new translate(new rotatey(new bvh(vol_boxes), 15), vec3d(-100, 270, 395)));
    bvh* bvh_grp = new bvh(grp, 0, 1);
    return group(bvh_grp);
}