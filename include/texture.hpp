#pragma once

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION


#include "stb/stb_image.hpp"

#include "util.hpp"
#include "algebra/algebra.hpp"


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

    image(const char* path, bool color_based=false)
    {
        this->path = path;
        this->color_based = color_based;
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
        
        if (color_based) {
            return gamma(vec3d(base[0], base[1], base[2]) * scale, 2.2);
        }
        else {
            return vec3d(base[0], base[1], base[2]) * scale;
        }
    }

private:
    const char* path;
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    bool color_based=false;
};
