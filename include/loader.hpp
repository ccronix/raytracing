#define TINYOBJLOADER_IMPLEMENTATION


#include <cstdio>
#include <vector>
#include <string>

#include "group.hpp"
#include "polygon.hpp"
#include "texture.hpp"
#include "material.hpp"
#include "algebra/algebra.hpp"
#include "loader/tiny_obj_loader.hpp"


class loader {

public:
    loader(const char* path)
    {
        scene_path = path;
        tinyobj::ObjReader reader;
        tinyobj::ObjReaderConfig config;
        config.mtl_search_path = dirname(path).c_str();

        if (!reader.ParseFromFile(path, config)) {
            printf("[TinyOBJLoader] error: %s\n", reader.Error().c_str());
            return;
        }

        if (!reader.Warning().empty()) {
            printf("[TinyOBJLoader] warning: %s\n", reader.Warning().c_str());
        }

        load_material(reader);
        load_mesh(reader);
    }

    group meshes() { return grp; }

    group* lights() { return lgt; }


private:
    group grp;
    group* lgt = new group();
    std::string scene_path;
    std::vector<material*> mats;

    std::string dirname(const std::string& path) const
    {
        std::string::size_type slash_index = path.find_last_of("/\\");
        std::string scene_dir;

        if (slash_index == std::string::npos) {
            scene_dir = ".";
        }
        else if (slash_index == 0) {
            scene_dir = "/";
        }
        else {
            scene_dir = path.substr(0, slash_index);
        }
        return scene_dir;
    }

    void load_mesh(tinyobj::ObjReader reader)
    {
        tinyobj::attrib_t attrib = reader.GetAttrib();
        std::vector<tinyobj::shape_t> shapes = reader.GetShapes();
        printf("[INFO] start loading mesh...\n");
        for (auto& shape : shapes) {
            printf("[TinyOBJLoader] load shape: %s\n", shape.name.c_str());
            int mat_id;
            int offset = 0;
            for (int face_id = 0; face_id < shape.mesh.num_face_vertices.size(); face_id++) {
                int face_num_vertices = shape.mesh.num_face_vertices[face_id];

                if (face_num_vertices != 3) {
                    continue;
                }

                std::vector<vertex> vertices(3);
                int mat_id = shape.mesh.material_ids[face_id];

                for (int vertex_id = 0; vertex_id < face_num_vertices; vertex_id++) {
                    tinyobj::index_t index = shape.mesh.indices[offset + vertex_id];

                    float px = attrib.vertices[3 * index.vertex_index];
                    float py = attrib.vertices[3 * index.vertex_index + 1];
                    float pz = attrib.vertices[3 * index.vertex_index + 2];

                    vertices[vertex_id].position = vec3d(px, py, pz);

                    if (index.normal_index >= 0) {
                        float nx = attrib.normals[3 * index.normal_index];
                        float ny = attrib.normals[3 * index.normal_index + 1];
                        float nz = attrib.normals[3 * index.normal_index + 2];

                        vertices[vertex_id].normal = vec3d(nx, ny, nz);
                    }
                    else {
                        vec3d p0p1 = vertices[1].position - vertices[0].position;
                        vec3d p0p2 = vertices[2].position - vertices[0].position;
                        vec3d normal = p0p1.cross(p0p2).normalize();

                        vertices[vertex_id].normal = normal;
                    }

                    if (index.texcoord_index >= 0) {
                        float u = attrib.texcoords[2 * index.texcoord_index];
                        float v = attrib.texcoords[2 * index.texcoord_index + 1];

                        vertices[vertex_id].uv_coord = vec2d(u, v);
                    }
                    else {
                        vertices[vertex_id].uv_coord = vec2d(0, 0);
                    }
                }
                offset += face_num_vertices;
                material* triangle_mat = mats[mat_id];
                triangle* mesh_triangle = new triangle(vertices, triangle_mat);
                grp.add(mesh_triangle);
                if (triangle_mat->has_emission()) {
                    lgt->add(mesh_triangle);
                }
            }
        }
        printf("[INFO] load mesh done.\n");
    }

    void load_material(tinyobj::ObjReader reader)
    {
        texture* base_color;
        std::vector<tinyobj::material_t> materials = reader.GetMaterials();
        std::string scene_dir = dirname(scene_path);
        printf("[INFO] start loading material...\n");
        for (auto& obj_mat : materials) {
            vec3d Kd = vec3d(obj_mat.diffuse[0], obj_mat.diffuse[1], obj_mat.diffuse[2]);
            vec3d Ks = vec3d(obj_mat.specular[0], obj_mat.specular[1], obj_mat.specular[2]);
            vec3d Ke = vec3d(obj_mat.emission[0], obj_mat.emission[1], obj_mat.emission[2]);
            vec3d Tr = vec3d(obj_mat.transmittance[0], obj_mat.transmittance[1], obj_mat.transmittance[2]);
            float Ns = obj_mat.shininess;
            float Ni = obj_mat.ior;

            if (!obj_mat.diffuse_texname.empty()) {
                std::string image_path = scene_dir + "/" + obj_mat.diffuse_texname;
                printf("[TinyOBJLoader] load diffuse image: %s\n", image_path.c_str());
                base_color = new image(image_path.c_str(), true);
            }
            else {
                printf("[TinyOBJLoader] load constant: vec3(%f, %f, %f)\n", Kd.x(), Kd.y(), Kd.z());
                base_color = nullptr;
            }

            phong* mat = new phong(base_color);
            mat->Kd = Kd;
            mat->Ks = Ks;
            mat->Ke = Ke;
            mat->Tr = Tr;
            mat->Ns = Ns;
            mat->Ni = Ni;

            mats.push_back(mat);
        }
        printf("[INFO] load material done.\n");
    }
};
