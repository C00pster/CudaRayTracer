#ifndef BVH_NODE_CUH
#define BVH_NODE_CUH

#include "aabb.cuh"
#include "primitives/primitive.cuh"
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
int64_t morton_code(const Point3& p);

__device__
void radix_sort(
    int64_t* morton_codes, 
    Primitive** primitives, 
    int64_t* sorted_codes, 
    Primitive** sorted_primitives, 
    size_t n
);

__device__
void buildLBVH(
    Primitive** primitives,
    Primitive** sorted_primitives,
    int64_t* morton_codes,
    int64_t* sorted_morton_codes,
    size_t n_initial_primitives,
    size_t* root_idx
);

#endif // BVH_NODE_H