#include <iostream>
#include <book.h>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>


#include "film.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "Hittable.cuh"
#include "Hittable_list.cuh"
#include "Sphere.cuh"
#include "camera.cuh"



#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
//const Ray& r, Interval ray_t, Hit_info& rec
//point3(0.0, -100.5, -1.0), 100.0
__host__ __device__ constexpr float infinity() {
    return 1e30f;
}
__device__ color ray_color_sample(const Ray& r, Hittable_list& world, int depth)
{

    if (depth <= 0)
        return vec3{ 0,0,0 };

    Hit_info rec;

    float infinity = 1e30f;
    Interval ray_t(0.000001, infinity );

    if (world.hit(r, ray_t, rec)) {
        //TODO
        Ray scat_ray;
        color attenuation;
        // if (rec.mat->scatter(r, rec, attenuation, scat_ray))
        // {
        //     return attenuation * ray_color_multiple_sample(scat_ray, world, depth - 1);
        // }
        return color(0, 0, 0);
        // return rec.normal;
        //vec3 direction = rec.normal + random_unit_vector();
        //return 0.5 * ray_color_multiple_sample(Ray(rec.p, direction), world, depth -1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}
__device__ color ray_color_sample_test(const Ray& r, Hittable_list& world, int depth)
{
    if (depth <= 0)
        return vec3{ 0,0,0 };
    Hit_info rec;
    float infinity = 1e30f;
    Interval ray_t(0.000001, infinity );
    if ((world).hit(r, ray_t, rec)) {
        //TODO
        Ray scat_ray;
        color attenuation;
        // if (rec.mat->scatter(r, rec, attenuation, scat_ray))
        // {
        //     return attenuation * ray_color_multiple_sample(scat_ray, world, depth - 1);
        // }
        // return color(1, 1, 1);
        return rec.normal;
        //vec3 direction = rec.normal + random_unit_vector();
        // return 0.5 * ray_color_multiple_sample(Ray(rec.p, direction), world, depth -1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    // return color(0, 0, 0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}
__device__ vec3 get_color(const Ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}
__global__ void render(unsigned char *buffer, int max_x, int max_y, point3 pixel_00_loc, point3 center, point3 pixel_delta_u, point3 pixel_delta_v,Hittable_list& world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //i j
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    point3 pixel_center = pixel_00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);


    vec3 ray_dir = pixel_center - center;
    Ray r(center, ray_dir);
    color pixel_color{ 0,0,0 };

    pixel_color += ray_color_sample(r, world, 10);
    pixel_color += vec3(0.5f, 0.f, 1.f);
    printf("kernel get j:%d, i:%d \n",j,i);
    Interval interv{ 0.0f,0.9f };
    //
    unsigned char uc_r = unsigned char(255.999 * interv.clamp(pixel_color.x()));
    unsigned char uc_g = unsigned char(255.999 * interv.clamp(pixel_color.y()));
    unsigned char uc_b = unsigned char(255.999 * interv.clamp(pixel_color.z()));

    buffer[pixel_index*3 + 0] = uc_r;
    buffer[pixel_index*3 + 1] = uc_g;
    buffer[pixel_index*3 + 2] = uc_g;

}

__global__ void render_test(unsigned char *buffer, int max_x, int max_y, camera_to_renderer_info renderer_info,Hittable_list& world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;



    __shared__ camera_to_renderer_info local_renderer_info;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_renderer_info = renderer_info;
    }
    printf("kernel get j:%d, i:%d",j,i);
    __syncthreads();

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    point3 pixel_center = local_renderer_info.pixel00_loc+ (i * local_renderer_info.pixel_delta_u) + (j * local_renderer_info.pixel_delta_v);

    vec3 ray_dir = pixel_center - local_renderer_info.center;
    Ray r(local_renderer_info.center, ray_dir);
    color pixel_color{ 0,0,0 };

    pixel_color += ray_color_sample_test(r, world, 10);

    // printf("kernel get j:%d, i:%d, color_r:%f, color_g:%f,color_b:%f \n",j,i,pixel_color.x(),pixel_color.y(),pixel_color.z());
    Interval interv{ 0.0f,0.9f };

    unsigned char uc_r = unsigned char(255.999 * interv.clamp(pixel_color.x()));
    unsigned char uc_g = unsigned char(255.999 * interv.clamp(pixel_color.y()));
    unsigned char uc_b = unsigned char(255.999 * interv.clamp(pixel_color.z()));

    buffer[pixel_index*3 + 0] = uc_r;
    buffer[pixel_index*3 + 1] = uc_g;
    buffer[pixel_index*3 + 2] = uc_b;
}

int main(){
    printf("Hello CUDA\n");
    int image_width = 64;
    int image_height = 64;

    Film film(image_width, image_height);

    //creat world
    Hittable_list* d_world;
    cudaMalloc(&d_world, sizeof(Hittable_list));
    init_hittable_list<<<1, 1>>>(d_world);
    cudaDeviceSynchronize();
    //add sphere
    point3 center = {0.0, -100.5, -1.0};
    float radius = 100.f;
    add_sphere_to_list<<<1, 1>>>(d_world, center, radius);
    cudaDeviceSynchronize();
    //add sphere
    center = {0.0f, 0.0f, -1.2f};
    radius = 0.5f;
    add_sphere_to_list<<<1, 1>>>(d_world, center, radius);
    cudaDeviceSynchronize();


    //set camera
    Camera camera(image_height,image_width);


    //set render
    int tx = 8;
    int ty = 8;
    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);

    std::clog << "\rStart.                 \n";
    render_test<<<blocks, threads>>>(film.get_data(), image_width, image_height, *camera.get_render_info(),*d_world);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    checkCudaErrors(cudaDeviceSynchronize());

    std::clog << "\rEnd.                 \n";

    // Output FB as Image
    film.save_as_png("film.png");

    checkCudaErrors(cudaFree(film.get_data()));

    return 0;
}