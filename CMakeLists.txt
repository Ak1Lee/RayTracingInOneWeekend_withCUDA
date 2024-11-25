cmake_minimum_required(VERSION 3.29)
project(RayTraceCuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(RayTraceCuda src/main.cu
        src/vec3.cu
        src/vec3.cuh
        src/color.cu
        src/color.cuh
        src/ray.cu
        src/ray.cuh
        src/camera.cu
        src/camera.cuh
        src/film.cu
        src/film.cuh
        src/interval.cu
        src/interval.cuh
        src/randf.cuh
)

set_target_properties(RayTraceCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
