#ifndef BOX_CUH
#define BOX_CUH

class Box : public Primitive {
public:
    __device__ Box() {}
    __device__ Box(const Point3& min, const Point3& max, Material* material) {
        box_min = min;
        box_max = max;

        sides.add(new XYRect(min.x(), max.x(), min.y(), max.y(), max.z(), material));
        sides.add(new FlipFace(new XYRect(min.x(), max.x(), min.y(), max.y(), min.z(), material)));
        sides.add(new XZRect(min.x(), max.x(), min.z(), max.z(), max.y(), material));
        sides.add(new FlipFace(new XZRect(min.x(), max.x(), min.z(), max.z(), min.y(), material)));
        sides.add(new YZRect(min.y(), max.y(), min.z(), max.z(), max.x(), material));
        sides.add(new FlipFace(new YZRect(min.y(), max.y(), min.z(), max.z(), min.x(), material)));
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        return sides.hit(r, t_min, t_max, rec);
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = AABB(box_min, box_max);
        return true;
    }

    Point3 box_min;
    Point3 box_max;
    PrimitiveList sides;
};

#endif