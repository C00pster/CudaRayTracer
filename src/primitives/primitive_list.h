#ifndef PRIMITIVE_LIST_H
#define PRIMITIVE_LIST_H

#include "primitives/primitive.h"
#include <thrust/device_vector.h>

class PrimitiveList : public Primitive {
    public:
        __device__ 
        PrimitiveList() {}

        __device__ 
        PrimitiveList(Primitive **l, size_t n) : list(l), list_size(n) {}

        __device__
        virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
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

        __device__
        virtual bool bounding_box(AABB& box) const {
            if (list_size < 1) return false;

            AABB temp_box;
            bool first_true = list[0]->bounding_box(temp_box);

            if (!first_true) return false;
            else box = temp_box;

            for (int i = 1; i < list_size; i++) {
                if (list[i]->bounding_box(temp_box)) {
                    box = surrounding_box(box, temp_box);
                } else {
                    return false;
                }
            }

            return true;
        }

        Primitive** list;
        size_t list_size;
        AABB bbox;
};

#endif // PRIMITIVE_LIST_H