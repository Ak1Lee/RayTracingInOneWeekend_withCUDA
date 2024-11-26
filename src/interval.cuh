//
// Created by 24977 on 2024/11/26.
//

#ifndef INTERVAL_CUH
#define INTERVAL_CUH



#include <limits>


class Interval {
public:
    float min, max;
    const float infinity = std::numeric_limits<float>::infinity();
    __host__ __device__ Interval() :min(-infinity), max(infinity) {};

    __host__ __device__ Interval(float min, float max) :min(min), max(max) {};

    __host__ __device__ float size()const {
        return max - min;
    }

    __host__ __device__ bool contains(float x)const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x)const {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const
    {
        if (x < min)return min;
        if (x > max)return max;
        return x;
    }

    // static const Interval empty, universe;
};



#endif //INTERVAL_CUH
