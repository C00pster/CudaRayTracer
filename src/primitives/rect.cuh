#ifndef RECT_CUH
#define RECT_CUH

class XYRect : public Primitive {
public:
    __device__ XYRect() {}
    __device__ XYRect(float _x0, float _x1, float _y0, float _y1, float _k, Material* mat) 
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {}

    __device__ virtual bool hit(const Ray& r, float t0, float t1, HitRecord& rec) const {
        float t = (k - r.origin().z()) / r.direction().z();
        if (t < t0 || t > t1) return false;
        float x = r.origin().x() + t * r.direction().x();
        float y = r.origin().y() + t * r.direction().y();
        if (x < x0 || x > x1 || y < y0 || y > y1) return false;
        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);
        rec.t = t;
        Vec4 outward_normal(0, 0, 1);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.at(t);
        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = AABB(Vec4(x0, y0, k - 0.0001), Vec4(x1, y1, k + 0.0001));
        return true;
    }

    Material* mat_ptr;
    float x0, x1, y0, y1, k;
};

class XZRect : public Primitive {
public:
    __device__ XZRect() {}
    __device__ XZRect(float _x0, float _x1, float _z0, float _z1, float _k, Material* mat) 
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {}

    __device__ bool hit(const Ray& r, float t0, float t1, HitRecord& rec) const {
        float t = (k - r.origin().y()) / r.direction().y();
        if (t < t0 || t > t1)
            return false;
        float x = r.origin().x() + t*r.direction().x();
        float z = r.origin().z() + t*r.direction().z();
        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;
        rec.u = (x - x0) / (x1 - x0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        Vec4 outward_normal = Vec4(0, 1, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.at(t);

        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = AABB(Vec4(x0, k - 0.0001, z0), Vec4(x1, k + 0.0001, z1));
        return true;
    }

    Material* mat_ptr;
    float x0, x1, z0, z1, k;
};

class YZRect : public Primitive {
public:
    __device__ YZRect() {}
    __device__ YZRect(float y_start, float y_end, float z_start, float z_end, float x_position, Material* mat) 
        : y0(y_start), y1(y_end), z0(z_start), z1(z_end), k(x_position), mat_ptr(mat) {}

    __device__ bool hit(const Ray& r, float t0, float t1, HitRecord& rec) const {
        float t = (k - r.origin().x()) / r.direction().x();
        if (t < t0 || t > t1)
            return false;
        float y = r.origin().y() + t*r.direction().y();
        float z = r.origin().z() + t*r.direction().z();
        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;
        rec.u = (y - y0) / (y1 - y0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        Vec4 outward_normal = Vec4(1, 0, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        rec.p = r.at(t);

        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = AABB(Vec4(k - 0.0001, y0, z0), Vec4(k + 0.0001, y1, z1));
        return true;
    }

    Material* mat_ptr;
    float y0, y1, z0, z1, k;
};

class FlipFace : public Primitive {
public:
    __device__ FlipFace(Primitive* p) : ptr(p) {}

    __device__ virtual bool hit(const Ray& r, float t0, float t1, HitRecord& rec) const {
        if (!ptr->hit(r, t0, t1, rec)) return false;
        rec.set_face_normal(r, -rec.normal);
        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        return ptr->bounding_box(t0, t1, output_box);
    }

    Primitive* ptr;
};

#endif // RECT_CUH