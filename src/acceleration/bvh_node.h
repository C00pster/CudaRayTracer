#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "aabb.h"
#include "primitives/primitive.h"
#include <cuda_runtime.h>
#include <cmath>

class BVHNode : public Primitive {
    public:

        __device__
        BVHNode() {}

        __device__
        BVHNode(AABB* bounding_box, size_t left, size_t right, Primitive** w) {
            bbox = *bounding_box;
            left_idx = left;
            right_idx = right;
            world = w;
        }

        __device__
        virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
            if (!bbox.hit(r, t_min, t_max)) return false;
            bool hit_left = false;
            bool hit_right = false;

            if (left_idx != -1 && world[left_idx]->hit(r, t_min, t_max, rec)) {
                hit_left = true;
            }
            if (right_idx != -1 && world[right_idx]->hit(r, t_min, hit_left ? rec.t : t_max, rec)) {
                hit_right = true;
            }
            return hit_left || hit_right;
        }

        __device__
        virtual bool bounding_box(AABB& bounding_box) const override {
            bounding_box = bbox;
            return true;
        }

        __device__
        virtual Point3 get_centroid() const override {
            // This should never be called
            return Point3();
        }

    private:
        AABB bbox;
        int left_idx;
        int right_idx;
        Primitive** world;
};

__device__
int64_t morton_code(const Point3& p) {
    int64_t x = (int64_t)(p.x() * 1024);
    int64_t y = (int64_t)(p.y() * 1024);
    int64_t z = (int64_t)(p.z() * 1024);

    x = (x | (x << 16)) & 0x030000FF00FF;
    x = (x | (x << 8)) & 0x0300F00F00F;
    x = (x | (x << 4)) & 0x030C30C30C3;
    x = (x | (x << 2)) & 0x09249249249;

    y = (y | (y << 16)) & 0x030000FF00FF;
    y = (y | (y << 8)) & 0x0300F00F00F;
    y = (y | (y << 4)) & 0x030C30C30C3;
    y = (y | (y << 2)) & 0x09249249249;

    z = (z | (z << 16)) & 0x030000FF00FF;
    z = (z | (z << 8)) & 0x0300F00F00F;
    z = (z | (z << 4)) & 0x030C30C30C3;
    z = (z | (z << 2)) & 0x09249249249;

    return x | (y << 1) | (z << 2);
}

__device__ void radix_sort(int64_t* morton_codes, Primitive** primitives, int64_t* sorted_codes, Primitive** sorted_primitives, size_t n) {
    const int BITS_PER_PASS = 8;
    const int MASK = 0xFF;
    const int NUM_PASSES = 8; // 64 bits / 8 bits per pass

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        // Step 1: Counting Phase
        int count[256] = {0};
        for (size_t i = 0; i < n; i++) {
            int byte = (morton_codes[i] >> (BITS_PER_PASS * pass)) & MASK;
            count[byte]++;
        }

        // Step 2: Prefix Sum (Exclusive Scan) to determine offsets
        int offset[256] = {0};
        int sum = 0;
        for (int i = 0; i < 256; i++) {
            offset[i] = sum;
            sum += count[i];
        }

        // Step 3: Scatter Phase
        for (size_t i = 0; i < n; i++) {
            int byte = (morton_codes[i] >> (BITS_PER_PASS * pass)) & MASK;
            size_t idx = offset[byte]++;
            sorted_codes[idx] = morton_codes[i];
            sorted_primitives[idx] = primitives[i];
        }
    }
}

__device__
void buildLBVH(
    Primitive** primitives,
    Primitive** sorted_primitives,
    int64_t* morton_codes,
    int64_t* sorted_morton_codes,
    size_t n_initial_primitives,
    size_t* root_idx
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (size_t i = 0; i < n_initial_primitives; i++) {
        morton_codes[i] = morton_code(primitives[i]->get_centroid());
    }

    radix_sort(morton_codes, primitives, sorted_morton_codes, sorted_primitives, n_initial_primitives);

    size_t num_nodes_at_level = n_initial_primitives;
    size_t start_counter = 0;
    while (num_nodes_at_level > 1) {
        for (size_t i = 0; i < (num_nodes_at_level + 1) / 2; i++) {
            size_t child_left = 2*i + start_counter;
            size_t child_right = child_left + 1;
            size_t parent = start_counter + num_nodes_at_level + i;
            if (2*i + 1 >= num_nodes_at_level) {
                sorted_primitives[parent] = sorted_primitives[child_left];
            } else {
                AABB box_left, box_right;
                sorted_primitives[child_left]->bounding_box(box_left);
                sorted_primitives[child_right]->bounding_box(box_right);
                AABB box = surrounding_box(box_left, box_right);
                sorted_primitives[parent] = 
                    new BVHNode(&box, child_left, child_right, sorted_primitives);
            }
        }
        start_counter += num_nodes_at_level;
        num_nodes_at_level = (num_nodes_at_level + 1) / 2;
    }

    *root_idx = n_initial_primitives * 2 - 2;
}

#endif // BVH_NODE_H