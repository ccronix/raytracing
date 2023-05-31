#pragma once


#include <vector>
#include <string>

#include "bvh.hpp"
#include "group.hpp"
#include "camera.hpp"
#include "object.hpp"


class scene {

public:
    group objects;
    group* lights = new group();
    camera render_camera;

    scene() {}

    scene(int width, int height) { set_size(width, height); }

    void set_size(int width, int height)
    {
        this->width = width;
        this->height = height;
        setup_camera_aspect_ratio();
    }

    void set_camera(const camera& cam)
    {
        render_camera = cam;
        setup_camera_aspect_ratio();
    }

    void add_object(const group& grp) { objects.add(grp); }

    void add_object(object* obj) { objects.add(obj); }

    void add_light(object* lgt) { lights->add(lgt); }

    void add_light(const group& lgt) { lights->add(lgt); }

    void build_bvh()
    {
        printf("[INFO] building bvh acceleration architecture...\n");
        objects = group(new bvh(objects));
        printf("[INFO] bvh done.\n");
    }
    
private:
    int width = 1280;
    int height = 720;

    void setup_camera_aspect_ratio()
    {
        double aspect_ratio = double(width) / height;
        render_camera.aspect_ratio = aspect_ratio;
    }
};
