#ifndef BVH_NODE_H
#define BVH_NODE_H

#include "aabb.h"
#include "primitives/primitive.h"

class BVHNode : public Primitive {
    public:

        BVHNode(AABB* bounding_box, size_t left_idx, size_t right_idx) {
            bbox = *bounding_box;
            left_idx = left_idx;
            right_idx = right_idx;
        }

        bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
            if (left_idx != -1 && !bbox.hit(r, t_min, t_max)) return false;

            if (left->hit(r, t_min, t_max, rec)) { // hit_left
                return true;
            }
            return (right_idx != -1 && right->hit(r, t_min, hit_left ? rec.t : t_max, rec)); // hit_right
        }

        AABB bounding_box() const override {
            return bbox;
        }

    private:
        AABB bbox;
        int left_idx;
        int right_idx;
}

thrust::device_vector<Primitive*>& computeBVH(thrust::device_vector<Primitive*>& primitives, size_t start, size_t end) {
    while (end - start > 1) {
        size_t old_end = end;
        size_t counter = end + 1;
        for (int i = start; i < old_end - 1; i += 2) {
            AABB box = surrounding_box(primitives[i]->bounding_box(), primitives[i + 1]->bounding_box());
            primitives[end++](new BVHNode(&box, i, i + 1));
        }

        if (old_end % 2 == 1) {
            primitives[end++](primitives[old_end - 1]);
        }

        start = old_end + 1;
    }

    return primitives;
}

#endif // BVH_NODE_H