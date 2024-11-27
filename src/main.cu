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

__device__ color ray_color_sample_multiple(const Ray& r, Hittable_list& world, const int MAXDEPTH,curandState& d_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float dec = 1.f;
    Ray current_ray = r;
    color accumulated_color(1.f, 1.f, 1.f);
    color accumulated_attenuation(1.0, 1.0, 1.0);

    int depth = MAXDEPTH;
    float infinity = 1e30f;
    Interval ray_t(0.001f, infinity );
    Hit_info rec;
    while (depth > 0) {
    //     printf("start at (%f,%f,%f),dir:(%f,%f,%f)\n",current_ray.origin().x(),current_ray.origin().y(),current_ray.origin().z(),
    // current_ray.direction().x(),current_ray.direction().y(),current_ray.direction().z());
        if(world.hit(current_ray,ray_t,rec)) {
            vec3 direction = random_on_hemisphere(rec.normal,&d_state);

            Ray scat_ray;
            color attenuation(1.f,1.f,1.f);
            dec *= 0.5f;
//             printf("ray %d %d Hit at (%f,%f,%f),Normal = (%f,%f,%f)\n,start at (%f,%f,%f),dir:(%f,%f,%f)\n",i,j,rec.p.x(), rec.p.y(), rec.p.z(),rec.normal.x(), rec.normal.y(), rec.normal.z(),
// current_ray.origin().x(),current_ray.origin().y(),current_ray.origin().z(),current_ray.direction().x(),current_ray.direction().y(),current_ray.direction().z());
            // printf("fun get,depth : %d\n",depth);
            if(rec.mat->scatter(current_ray,rec,attenuation,scat_ray,&d_state)) {
                // attenuation *= 0.5f;
                current_ray = scat_ray;
                accumulated_attenuation *= attenuation;
            }else {
                return color(0, 0, 0);
            }
        }else {
            // break;
            // if (depth < MAXDEPTH)return accumulated_color * attenuation;;

            vec3 unit_direction = unit_vector(current_ray.direction());
            auto a = 0.5f * (unit_direction.y() + 1.0f);
            return accumulated_attenuation * ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
        }
        --depth;

    }
    // vec3 unit_direction = unit_vector(r.direction());
    // auto a = 0.5f * (unit_direction.y() + 1.0f);
    // return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    vec3 unit_direction = unit_vector(current_ray.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return accumulated_attenuation * ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
}

__global__ void render_test(unsigned char *buffer, int max_x, int max_y, camera_to_renderer_info renderer_info,Hittable_list& world,curandState* d_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = j*max_x+i;
    // printf("i : %d j : %d\n",i,j);
    if (i >= max_x || j >= max_y) return;

    curandState local_state = d_state[idx];

    __shared__ camera_to_renderer_info local_renderer_info;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        local_renderer_info = renderer_info;
    }
    // printf("kernel get j:%d, i:%d",j,i);
    __syncthreads();

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    point3 pixel_center = local_renderer_info.pixel00_loc+ (i * local_renderer_info.pixel_delta_u) + (j * local_renderer_info.pixel_delta_v);

    vec3 ray_dir = pixel_center - local_renderer_info.center;
    Ray r(local_renderer_info.center, ray_dir);
    color pixel_color{ 0,0,0 };

    int multisample = 100;
    for (int k = 0; k < multisample; ++k) {
        pixel_color += ray_color_sample_multiple(r, world, 20, local_state);
    }
    pixel_color /= multisample;



    // printf("kernel get j:%d, i:%d, color_r:%f, color_g:%f,color_b:%f \n",j,i,pixel_color.x(),pixel_color.y(),pixel_color.z());
    Interval interv{ 0.0f,0.95f };

    unsigned char uc_r = unsigned char(255.999 * interv.clamp(pixel_color.x()));
    unsigned char uc_g = unsigned char(255.999 * interv.clamp(pixel_color.y()));
    unsigned char uc_b = unsigned char(255.999 * interv.clamp(pixel_color.z()));

    buffer[pixel_index*3 + 0] = uc_r;
    buffer[pixel_index*3 + 1] = uc_g;
    buffer[pixel_index*3 + 2] = uc_b;

    d_state[idx] = local_state;
}
__global__ void init_curand_states(curandStateXORWOW_t* states, unsigned int seed, int N,int image_width) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = j * image_width + i;
    if (idx >= N) return;
    // printf("idx rand : %d\n",idx);
    curand_init(seed, idx, 0, &states[idx]);  // 初始化每个线程的状态
}
__device__ inline float random_float_XORWOW(float min, float max, curandStateXORWOW_t* state) {
    float random = curand_uniform(state);  // 生成 [0, 1) 范围内的随机数
    return min + random * (max - min);     // 缩放到 [min, max) 范围
}

__global__ void generate_random_in_range(curandStateXORWOW_t* states, float* results, float min, float max, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    curandStateXORWOW_t local_state = states[idx];  // 获取当前线程的状态

    // 生成 [min, max) 范围内的随机浮点数
    float random = random_float(min, max, &local_state);

    results[idx] = random;  // 将生成的随机数存储到结果数组

    states[idx] = local_state;  // 保存状态
}
void test_rand() {
    const int N = 256;  // 随机数个数
    float* d_results;   // 结果数组
    curandStateXORWOW_t* d_states;  // cuRAND 状态

    // 分配内存
    cudaMalloc(&d_results, N * sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandStateXORWOW_t));

    // 初始化 cuRAND 状态
    int threads = 32;
    int blocks = (N + threads - 1) / threads;
    init_curand_states<<<blocks, threads>>>(d_states, 1234, N,16);

    // 生成随机数
    generate_random_in_range<<<blocks, threads>>>(d_states, d_results, 0.0f, 1.0f, N);

    // 同步设备
    cudaDeviceSynchronize();

    // 将结果拷贝回主机
    float h_results[N];
    cudaMemcpy(h_results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; i++) {
        std::cout << "Random[" << i << "]: " << h_results[i] << std::endl;
    }

    generate_random_in_range<<<blocks, threads>>>(d_states, d_results, 0.0f, 1.0f, N);

    // 同步设备
    cudaDeviceSynchronize();

    // 将结果拷贝回主机
    cudaMemcpy(h_results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; i++) {
        std::cout << "Random[" << i << "]: " << h_results[i] << std::endl;
    }

    // 清理
    cudaFree(d_results);
    cudaFree(d_states);
}

enum sharp_set {
    Sphere
};



int main(){
    // test_rand();


    printf("Hello CUDA\n");
    int image_width = 1200;
    int image_height = 675;

    Film film(image_width, image_height);


    //creat world
    Hittable_list* d_world;
    cudaMalloc(&d_world, sizeof(Hittable_list));
    init_hittable_list<<<1, 1>>>(d_world);
    cudaDeviceSynchronize();

    //set camera
    Camera camera(image_width,image_height);



    //set render
    int tx = 16;
    int ty = 16;
    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);

    //rand
    curandState* d_states;
    cudaMalloc(&d_states, image_height * image_width * sizeof(curandState));


    std::clog << "\rStart.                 \n";
    init_curand_states<<<blocks, threads>>>(d_states, 1234, image_width*image_height,image_width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    render_test<<<blocks, threads>>>(film.get_data(), image_width, image_height, *camera.get_render_info(),*d_world, d_states);
    err = cudaGetLastError();
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