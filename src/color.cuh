//
// Created by 24977 on 2024/11/25.
//

#ifndef COLOR_CUH
#define COLOR_CUH


#include "vec3.cuh"
#include "ray.cuh"

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color);

inline float color_linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
    {
        return std::sqrt(linear_component);
    }
    return 0;
}




#endif //COLOR_CUH
