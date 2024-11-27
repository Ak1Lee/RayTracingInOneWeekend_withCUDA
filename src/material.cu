//
// Created by 24977 on 2024/11/27.
//

#include "material.cuh"
#include "Hittable.cuh"



bool Lambertion:: scatter(const Ray& r_in, const Hit_info& rec, color& attenuation, Ray& scattered, curandState* state) const
{

    auto tmp_dir = rec.normal + random_unit_vector(state);

    while (tmp_dir.near_zero())
    {
        tmp_dir = rec.normal + random_unit_vector(state);
    }
    scattered = Ray(rec.p, tmp_dir);
    attenuation = albedo;
    return true;
}

bool Metal::scatter(const Ray &r_in, const Hit_info &rec, color &attenuation, Ray &scattered, curandState* state) const{
    auto reflec = reflect(r_in.direction(),rec.normal);
    // reflec = unit_vector(reflec) + (fuzz * random_unit_vector(state));
    scattered = Ray(rec.p, reflec);
    attenuation = albedo;
    return true;
}

bool Dielectric::scatter(const Ray &r_in, const Hit_info &rec, color &attenuation, Ray &scattered, curandState* state) const {
    attenuation = color(1.f, 1.f, 1.f);
    float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

    vec3 unit_dir = unit_vector(r_in.direction());
    // vec3 refracted = refract(unit_dir, rec.normal, ri);
    float cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);


    auto tmp_f = ri * sin_theta;
    bool cannot_refract = (tmp_f > 1.f);
    vec3 direction;

    if (cannot_refract||reflectance(cos_theta, ri) > random_float(0,1,state))
    {
        direction = reflect(unit_dir, rec.normal);

    }
    else
        direction = refract(unit_dir, rec.normal, ri);

    scattered = Ray(rec.p, direction);

    //scattered = Ray(rec.p, refracted);

    return true;
}
