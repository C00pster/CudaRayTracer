#ifndef PRIMITIVE_LIST_H
#define PRIMITIVE_LIST_H

#include "primitives/primitive.h"

class PrimitiveList : public Primitive {
    public:
        __device__ PrimitiveList() {}
                __device__ PrimitiveList(Primitive **l, int n) : list(l), list_size(n) {
            if (list_size == 0) {
                // Handle empty list appropriately
            } else {
                AABB temp_box;
                bool first_box = true;
                for (int i = 0; i < list_size; i++) {
                    if (!list[i]->bounding_box(temp_box)) {
                        // Handle error: one of the primitives doesn't have a bounding box
                    }
                    bbox = first_box ? temp_box : surrounding_box(bbox, temp_box);
                    first_box = false;
                }
            }
        }

        __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const;
        __device__ virtual bool bounding_box(AABB& output_box) const;

        Primitive **list;
        int list_size;
        AABB bbox;
};

__device__ bool PrimitiveList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ bool PrimitiveList::bounding_box(AABB& bounding_box) const {
    bounding_box = bbox;
    return true;
}

#endif // PRIMITIVE_LIST_H