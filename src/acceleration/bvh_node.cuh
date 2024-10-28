#ifndef BVH_NODE_CUH
#define BVH_NODE_CUH

#include "math/ray.cuh"
#include "primitives/primitive_list.cuh"

enum Axis { X, Y, Z };

__device__ void swap(Primitive*& a, Primitive*& b) {
    Primitive* temp = a;
    a = b;
    b = temp;
}

template<Axis axis>
__device__ void bubble_sort(Primitive** primitives, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            AABB box_left, box_right;
            Primitive *ah = primitives[j];
            Primitive *bh = primitives[j + 1];

            ah->bounding_box(0, 0, box_left);
            bh->bounding_box(0, 0, box_right);

            if ((axis == X && (box_left.min().x() - box_right.min().x()) < 0.0f) ||
                (axis == Y && (box_left.min().y() - box_right.min().y()) < 0.0f) ||
                (axis == Z && (box_left.min().z() - box_right.min().z()) < 0.0f)) {
                swap(primitives[j], primitives[j + 1]);
            }
        }
    }
}

class BVHNode : public Primitive {
public:
    __device__ BVHNode() {}
    __device__ BVHNode(Primitive **primitives, int n, float time0, float time1, curandState& local_rand_state) {
        int axis = int(3 * curand_uniform(&local_rand_state));

        if (axis == 0) {
            bubble_sort<X>(primitives, n);
        } else if (axis == 1) {
            bubble_sort<Y>(primitives, n);
        } else {
            bubble_sort<Z>(primitives, n);
        }

        if (n == 1) {
            left = right = primitives[0];
        } else if (n == 2) {
            left = primitives[0];
            right = primitives[1];
        } else {
            left = new BVHNode(primitives, n / 2, time0, time1, local_rand_state);
            right = new BVHNode(primitives + n / 2, n - n / 2, time0, time1, local_rand_state);
        }

        AABB box_left, box_right;
        bbox = surrounding_box(box_left, box_right);
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        if (!bbox.hit(r, t_min, t_max)) return false;
        HitRecord left_rec, right_rec;
        bool hit_left = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, hit_left ? left_rec.t : t_max, right_rec);
        if (hit_left && hit_right) {
            rec = left_rec.t < right_rec.t ? left_rec : right_rec;
            return true;
        } else if (hit_left) {
            rec = left_rec;
            return true;
        } else if (hit_right) {
            rec = right_rec;
            return true;
        } else return false;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& bounding_box) const override {
        bounding_box = bbox;
        return true;
    }

private:
    Primitive* left;
    Primitive* right;
    AABB bbox;
};

#endif // BVH_NODE_H