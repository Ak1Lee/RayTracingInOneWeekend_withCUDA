//
// Created by 24977 on 2024/11/26.
//

#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"

class Sphere :public Hittable
{
public:
    // Sphere(const point3& center, const float& radius, std::shared_ptr<Material> mat) :center(center), radius(radius) ,mat(mat){};
    __host__ __device__ Sphere(const point3& center, const float& radius) :center(center), radius(radius){};
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, Hit_info& rec) const override;

private:
    point3 center;
    float radius;
    vec3 color = {1,1,1};
    // std::shared_ptr<Material> mat;
};



#endif //SPHERE_CUH
