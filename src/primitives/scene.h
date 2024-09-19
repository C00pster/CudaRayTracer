#ifndef SCENE_H
#define SCENE_H

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector>
#include <string>
#include <iostream>
#include "math/vec4.h"

class Scene {
    public:
        struct Vertex {
            Vec4 position;
            Vec4 normal;
            float texCoord[2];
        };

        Scene(const std::string& filename) {
            loadObjFile(filename);
        }

        const std::vector<Vertex>& getVertices() const {
            return vertices;
        }

        const std::vector<unsigned int>& getIndices() const {
            return indices;
        }

    private:
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        void loadObjFile(const std::string& filename) {
            Assimp::Importer importer;

            const aiScene* scene = importer.ReadFile(filename, 
                aiProcess_Triangulate | 
                aiProcess_FlipUVs |
                aiProcess_JoinIdenticalVertices |
                aiProcess_GenNormals);

            if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
                std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
                return;
            }

            aiMesh* mesh = scene->mMeshes[0];
            processMesh(mesh);
        }

        void processMesh(aiMesh* mesh) {
            for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
                Vertex vertex;

                //Position
                vertext.position = Vec4(mesh->mVertices[i].x,
                                        mesh->mVertices[i].y,
                                        mesh->mVertices[i].z,
                                        0.0f);

                //Normal
                if (mesh->HasNormals()) {
                    vertex.normal = Vec4(mesh->mNormals[i].x,
                                         mesh->mNormals[i].y,
                                         mesh->mNormals[i].z,
                                         1.0f);
                }

                //Texture coordinates
                if (mesh->HasTextureCoords(0)) {
                    vertex.texCoord[0] = mesh->mTextureCoords[0][i].x;
                    vertex.texCoord[1] = mesh->mTextureCoords[0][i].y;
                } else {
                    vertex.texCoord[0] = 0.0f;
                    vertex.texCoord[1] = 0.0f;
                }

                vertices.push_back(vertex);
            }

            for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
                aiFace face = mesh->mFaces[i];
                for (unsigned int j = 0; j < face.mNumIndices; j++) {
                    indices.push_back(face.mIndices[j]);
                }
            }
        }
};

#endif // SCENE_H