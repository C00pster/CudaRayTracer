#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#define DEGREES_TO_RADIANS(degrees)((M_PI * degrees) / 180.0f)

class Translate : public Primitive {
public:
    __device__ Translate(Primitive* p, const Vec4& displacement) : ptr(p), offset(displacement) {}
    
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Ray moved_r(r.origin() - offset, r.direction(), r.time());
        if (!ptr->hit(moved_r, t_min, t_max, rec)) return false;

        rec.p += offset;
        rec.set_face_normal(moved_r, rec.normal);

        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        if (!ptr->bounding_box(t0, t1, output_box)) return false;

        output_box = AABB(output_box.min() + offset, output_box.max() + offset);
        return true;
    }

    Primitive* ptr;
    Vec4 offset;
};

class RotateY : public Primitive {
public:
    __device__ RotateY(Primitive* p, float angle) : ptr(p) {
        float radians = DEGREES_TO_RADIANS(angle);
        sin_theta = sin(radians);
        cos_theta = cos(radians);
        has_box = ptr->bounding_box(0, 1, bbox);

        Point3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        Point3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
                    float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
                    float z = k * bbox.max().z() + (1 - k) * bbox.min().z();

                    float new_x = cos_theta * x + sin_theta * z;
                    float new_z = -sin_theta * x + cos_theta * z;

                    Point3 tester(new_x, y, new_z);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], tester[c]);
                        max[c] = fmaxf(max[c], tester[c]);
                    }
                }
            }
        }

        bbox = AABB(min, max);
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec4 origin = r.origin();
        Vec4 direction = r.direction();

        origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
        origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

        direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
        direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

        Ray rotated_r(origin, direction, r.time());
        if (!ptr->hit(rotated_r, t_min, t_max, rec)) return false;

        Vec4 p = rec.p;
        Vec4 normal = rec.normal;

        p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
        p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

        normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
        normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

        rec.p = p;
        rec.set_face_normal(rotated_r, normal);

        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = bbox;
        return has_box;
    }

    Primitive* ptr;
    float sin_theta;
    float cos_theta;
    bool has_box;
    AABB bbox;
};

#endif // TRANSFORM_CUH