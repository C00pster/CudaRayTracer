#ifndef QUAD_CUH
#define QUAD_CUH

#include "primitive.cuh"
#include "primitive_list.cuh"
#include "math/ray.cuh"

class Quad : public Primitive {
public:
    __device__ Quad() {}
    __device__ Quad(const Point3& Q, const Vec4& u, const Vec4& v, Material* mat)
        : Q(Q), u(u), v(v), mat_ptr(mat)
    {
        Vec4 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n,n);

        set_bounding_box();
    }

    __device__ virtual void set_bounding_box() {
        AABB bbox_diagonal1(Q, Q + u + v);
        AABB bbox_diagonal2(Q + u, Q + v);
        bbox = surrounding_box(bbox_diagonal1, bbox_diagonal2);
    }

    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = bbox;
        return true;
    }

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float denom = dot(normal, r.direction());

        if (fabs(denom) < 1e-6) return false; // Ray is parallel to the plane

        float t = (D - dot(normal, r.origin())) / denom;
        if (t < t_min || t > t_max) return false;

        Vec4 intersection = r.at(t);
        Vec4 planar_hitpt_vector = intersection - Q;
        float alpha = dot(w, cross(planar_hitpt_vector, v));
        float beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec)) return false;

        rec.t = t;
        rec.p = intersection;
        rec.mat_ptr = mat_ptr;
        rec.set_face_normal(r, normal);

        return true;
    }

    __device__ virtual bool is_interior(float alpha, float beta, HitRecord& rec) const {
        if (alpha < 0 || beta < 0 || alpha > 1 || beta > 1) return false;

        rec.u = alpha;
        rec.v = beta;
        return true;
    }

    private:
        Point3 Q;
        Vec4 u, v, w, normal;
        Material* mat_ptr;
        float D;
        AABB bbox;
};

__device__ inline PrimitiveList* box(const Point3& a, const Point3& b, Material* mat) {
    PrimitiveList* sides = new PrimitiveList();

    Point3 min = Point3(fmin(a.x(),b.x()), fmin(a.y(),b.y()), fmin(a.z(),b.z()));
    Point3 max = Point3(fmax(a.x(),b.x()), fmax(a.y(),b.y()), fmax(a.z(),b.z()));

    Vec4 dx = Vec4(max.x() - min.x(), 0, 0);
    Vec4 dy = Vec4(0, max.y() - min.y(), 0);
    Vec4 dz = Vec4(0, 0, max.z() - min.z());

    sides->add(new Quad(Point3(min.x(), min.y(), max.z()),  dx,  dy, mat)); // front
    sides->add(new Quad(Point3(max.x(), min.y(), max.z()), -dz,  dy, mat)); // right
    sides->add(new Quad(Point3(max.x(), min.y(), min.z()), -dx,  dy, mat)); // back
    sides->add(new Quad(Point3(min.x(), min.y(), min.z()),  dz,  dy, mat)); // left
    sides->add(new Quad(Point3(min.x(), max.y(), max.z()),  dx, -dz, mat)); // top
    sides->add(new Quad(Point3(min.x(), min.y(), min.z()),  dx,  dz, mat)); // bottom

    return sides;
}

#endif // QUAD_CUH