#ifndef WORLD_CUH
#define WORLD_CUH

#include "acceleration/bvh_node.cuh"
#include "primitives/primitive.cuh"
#include <vector>
#include <cstdint>

class World {
    public:
        __device__
        World() {}

        __device__
        World(Primitive** primitives, Primitive** sorted_primitives, size_t n_primitives, int64_t* morton_codes, int64_t* sorted_codes) {
            list_size = n_primitives * 2 - 1;

            buildLBVH(primitives, sorted_primitives, morton_codes, sorted_codes, n_primitives, &root_idx);
            world_list = sorted_primitives;
        }

        __device__
        virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
            return world_list[root_idx]->hit(r, t_min, t_max, rec);
        }

        Primitive** world_list;
        size_t list_size;
        size_t root_idx;
};

#endif // WORLD_CUH