//
// Created by 24977 on 2024/11/25.
//

#include "vec3.cuh"

#include "randf.cuh"

// #include "util/randf.h"
__host__ __device__ vec3 vec3::random()
{
    // return vec3(random_float(), random_float(), random_float());
    return vec3();
}

vec3 vec3::random(float min, float max, curandState* state) {
#ifdef __CUDA_ARCH__
    return vec3(
        random_float(min, max, state),
        random_float(min, max, state),
        random_float(min, max, state));
#else
    return vec3(
        random_float(min, max),
        random_float(min, max),
        random_float(min, max));
#endif
}
