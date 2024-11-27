//
// Created by 24977 on 2024/11/27.
//

#ifndef MATERIAL_CUH
#define MATERIAL_CUH



// #include "hittable.cuh"
#include "color.cuh"
#include "randf.cuh"

struct Hit_info;

class Material
{
public:
	__host__ __device__ virtual ~Material() = default;
    __host__ __device__ virtual bool scatter(const Ray& r_in, const Hit_info& rec, color& attenuation, Ray& scattered, curandState* state) const {
        return false;
    }
};

class Lambertion : public Material
{
public:
    __host__ __device__ Lambertion(const color& albedo) :albedo(albedo) {};
    __host__ __device__ virtual bool scatter(const Ray& r_in, const Hit_info& rec, color& attenuation, Ray& scattered,curandState* state) const override;

private:
    color albedo;
};

class Metal : public Material
{
public:
    __host__ __device__ Metal(const color& albedo, float fuzz) :albedo(albedo), fuzz(fuzz<1?fuzz:1) {};
    __host__ __device__ virtual bool scatter(const Ray& r_in, const Hit_info& rec, color& attenuation, Ray& scattered, curandState* state) const override;

private:
    color albedo;
    float fuzz;
};


class Dielectric : public Material
{
public:
    __host__ __device__ Dielectric(float ref_idx) :refraction_index(ref_idx) {};
    __host__ __device__ virtual bool scatter(const Ray& r_in, const Hit_info& rec, color& attenuation, Ray& scattered, curandState* state) const override;


private:
    float refraction_index;

    __host__ __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};



#endif //MATERIAL_CUH
