//
// Created by 24977 on 2024/11/25.
//

#ifndef RAY_CUH
#define RAY_CUH
#include "vec3.cuh"


using point3 = vec3;
class Ray
{
public:
    __host__ __device__ Ray(){}

    __host__ __device__ Ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {};

    __host__ __device__ const point3& origin()const
    {
        return orig;
    }

    __host__ __device__ const vec3& direction()const
    {
        return dir;
    }

    __host__ __device__ point3 at(float t) const
    {
        return orig + (dir * t);
    }

private:
    point3 orig;
    vec3 dir;

};



#endif //RAY_CUH
