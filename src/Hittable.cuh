//
// Created by 24977 on 2024/11/26.
//

#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "interval.cuh"


struct Hit_info
{
    __host__ __device__ Hit_info()=default;
    point3 p;
    vec3 normal;
    // std::shared_ptr<Material> mat;
    vec3 color;
    float t;
    bool front_face;

    __host__ __device__ void set_face_normal(const Ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    __host__ __device__ virtual ~Hittable() = default;

    __host__ __device__ virtual bool hit(const Ray& r, Interval ray_t, Hit_info& rec) const = 0;
};



#endif //HITTABLE_CUH
