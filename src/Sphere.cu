//
// Created by 24977 on 2024/11/26.
//

#include "Sphere.cuh"
__device__ bool Sphere::hit(const Ray& r, Interval ray_t, Hit_info& rec) const
{
    printf("func get\n");
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = h * h - a * c;
    if (discriminant < 0) {
        return false;
    }
    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.p = r.at(root);
    rec.normal = unit_vector(rec.p- vec3(0, 0, -1));
    rec.t = root;
    // rec.mat = mat;


    vec3 outward_normal = (rec.p - center) / radius;
    vec3 normal;

    rec.set_face_normal(r, outward_normal);
    //rec.set_face_normal(r, normal);

    return true;
}
