//
// Created by 24977 on 2024/11/26.
//

#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "material.cuh"

class Sphere :public Hittable
{
public:
    __host__ __device__ Sphere(const point3& center, const float& radius, Material* mat) :center(center), radius(radius) ,mat(mat){};
    __host__ __device__ Sphere(const point3& center, const float& radius) :center(center), radius(radius) {
        mat = new Lambertion(vec3(0.5f,0.9f,0.5f));
    };
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, Hit_info& rec) const override;

private:
    point3 center;
    float radius;
    vec3 color = {1,1,1};
    Material* mat;
};



#endif //SPHERE_CUH
