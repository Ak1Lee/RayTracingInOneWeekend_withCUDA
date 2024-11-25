//
// Created by 24977 on 2024/11/26.
//

#pragma once

#include <cstdlib>
#include <random>


// 使用静态局部变量来存储生成器和分布，确保它们只被初始化一次
static std::mt19937 generator(std::random_device{}());
static std::uniform_real_distribution<float> distribution(0.0, 1.0);

//inline float random_float() {
//    // Returns a random real in [0,1).
//    return std::rand() / (RAND_MAX + 1.0);
//}

inline float random_float() {
    // Returns a random real in [0,1).
    return distribution(generator);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

