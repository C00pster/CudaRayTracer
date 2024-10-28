#ifndef PRIMITIVE_LIST_CUH
#define PRIMITIVE_LIST_CUH

#include "primitive.cuh"

class PrimitiveList : public Primitive {
public:
    __device__ PrimitiveList() {}
    __device__ PrimitiveList(Primitive **lst, int n) { list = lst; list_size = n; allocated_list_size = n; }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
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

    __device__ virtual bool bounding_box(float t0, float t1, AABB& bbox) const {
        if (list_size < 1) return false;

        AABB temp_box;
        bool first_true = list[0]->bounding_box(t0, t1, temp_box);
        if (!first_true) return false;
        bbox = temp_box;

        for (int i = 1; i < list_size; i++) {
            if (list[i]->bounding_box(t0, t1, temp_box)) {
                bbox = surrounding_box(bbox, temp_box);
            } else return false;
        }
        return true;
    }

    __device__ void add(Primitive* p) {
        if (allocated_list_size <= list_size) {
            Primitive** new_list = new Primitive*[allocated_list_size * 2];
            for (int i = 0; i < list_size; i++) {
                new_list[i] = list[i];
            }
            list = new_list;
            allocated_list_size *= 2;
        }
        list[list_size++] = p;
    }

    Primitive** list;
    int list_size;
    int allocated_list_size;
};

#endif // PRIMITIVE_LIST_CUH