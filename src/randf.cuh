//
// Created by 24977 on 2024/11/26.
//

#pragma once

#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>  // 需要包含 cuRAND 头文件
#include <curand.h>
// using Random123

static std::mt19937 generator(std::random_device{}());
static std::uniform_real_distribution<float> distribution(0.0, 1.0);

inline float random_float() {
    // Returns a random real in [0,1).
    return distribution(generator);
}
__host__ inline float random_float(float min, float max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

__host__ __device__ inline float random_float(float min, float max, curandState* state) {
#ifdef __CUDA_ARCH__
    float random = curand_uniform(state);
    return min + random * (max - min);

#else

    return random_float(min, max);
#endif
}









__device__ __host__ inline float random_float(float min, float max, unsigned int seed) {
#ifdef __CUDA_ARCH__
    seed = (1103515245 * seed + 12345) & 0x7fffffff;
    float random = static_cast<float>(seed) / static_cast<float>(0x7fffffff);
#else

    static thread_local std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float random = distribution(generator);
#endif

    return min + (max - min) * random;
}