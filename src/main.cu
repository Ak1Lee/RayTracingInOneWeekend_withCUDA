#include <iostream>
#include <book.h>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>  // 包含 CUDA 数学常量


#include "film.cuh"
#include "vec3.cuh"
#include "ray.cuh"
#include "Hittable.cuh"
#include "Hittable_list.cuh"
#include "Sphere.cuh"


#include <math_constants.h>

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
__global__ void create_function(
    Hittable** hittable,
    point3 center,
    float radius
    )
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        *hittable = new Sphere(center, radius);
    }
}
__global__ void delete_function(Hittable** hittable)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        delete *hittable;
    }
}
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
        printf("Func get\n");
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
__device__ color ray_color_sample_test(const Ray& r, Hittable const* const* __restrict__ hittables, int depth)
{
    if (depth <= 0)
        return vec3{ 0,0,0 };
    Hit_info rec;
    float infinity = 1e30f;
    Interval ray_t(0.000001, infinity );
    if ((*hittables)->hit(r, ray_t, rec)) {
        //TODO
        Ray scat_ray;
        printf("Func get\n");
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

__global__ void render_test(unsigned char *buffer, int max_x, int max_y, point3 pixel_00_loc, point3 center, point3 pixel_delta_u, point3 pixel_delta_v,Hittable const* const* __restrict__ hittables) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //i j
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    point3 pixel_center = pixel_00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);


    vec3 ray_dir = pixel_center - center;
    Ray r(center, ray_dir);
    color pixel_color{ 0,0,0 };

    pixel_color += ray_color_sample_test(r, hittables, 10);
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

int main(){
    printf("Hello CUDA\n");
    int image_width = 32;
    int image_height = 32;

    Film film(image_width, image_height);

    int num_pixels = 3 * image_width * image_height;
    size_t film_buffer_size = num_pixels * sizeof(unsigned char);
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(film.get_data()),film_buffer_size));

    // Hittable_list world;
    // world.add(make_shared<Sphere>(point3(0.0, -100.5, -1.0), 100.0));
    // world.add(make_shared<Sphere>(point3(0.0, 0.0, -1.2), 0.5));
    Hittable_list world;
    create_function<<<1,1>>>(world.objects,point3(0.0, -100.5, -1.0),100.f);
    create_function<<<1,1>>>(world.objects,point3(0.0, 0.0, -1.2),0.5f);





    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);



    //camera
    float aspect_ratio = 1.f;
    double vfov_degree = 90;  // Vertical view angle (field of view)
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);             //"up"dir
    float defocus_angle = 0.f;
    float focus_dist = 10;
    image_height = image_height < 1 ? 1 : image_height;
    auto center = lookfrom;
    auto focal_length = (lookfrom - lookat).length();


    float theta = vfov_degree * 3.1415926f / 180.f;
    float h = std::tan(theta / 2);
    //float viewport_height = 2 * h * focal_length;
    float viewport_height = 2 * h * focus_dist;

    //cal w u v;
    auto w = unit_vector(lookfrom - lookat);
    auto u = unit_vector(cross(vup, w));
    auto v = cross(w, u);
    auto viewport_width = viewport_height * (float(static_cast<float>(image_width) / image_height));
    auto viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;

    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = center
        - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    auto pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    auto defocus_radius = focus_dist * std::tan((defocus_angle / 2) * (3.1415926f / 180.0));
    vec3 defocus_disk_u = u * defocus_radius;
    vec3 defocus_disk_v = v * defocus_radius;




    std::clog << "\rStart.                 \n";
    render_test<<<blocks, threads>>>(film.get_data(), image_width, image_height, pixel00_loc,center,pixel_delta_u,pixel_delta_v,world.objects);

    // checkCudaErrors(cudaGetLastError());
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