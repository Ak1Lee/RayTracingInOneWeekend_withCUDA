//
// Created by 24977 on 2024/11/26.
//

#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "hittable.cuh"
#include "Sphere.cuh"
#include <vector>
#include <memory>

class Hittable_list;

__global__ void kernel_add_Sphere(Hittable_list& world,point3 center,float radius);

class Hittable_list
{
public:
    // std::vector<shared_ptr<Hittable>> objects;

    static const int max_objects = 20;
    Hittable** objects;
    int num_objects = 0;


    __host__ __device__ Hittable_list() {
        cudaMalloc(&objects, max_objects * sizeof(Hittable*));
    }

    // __host__ __device__ Hittable_list(Hittable* object){add(object); }
    __host__ __device__ ~Hittable_list() {
        for (int i = 0; i < num_objects; i++) {
            delete objects[i]; // 释放设备端对象
        }
        cudaFree(objects); // 释放设备端数组内存
    }

    __host__ __device__ void clear()
    {
        num_objects = 0;
    }
    __host__ __device__ void add(Hittable* other)
    {
        if (num_objects < max_objects) {
            objects[num_objects++] = other;
        }
    }
    __device__ bool hit(const Ray& r, Interval ray_t, Hit_info& rec)
    {

        Hit_info temp_info;
        bool is_hit = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < num_objects; i++)
        {
            Hittable* object = objects[i];

            if (object->hit(r, {ray_t.min,closest_so_far}, temp_info))
            {
                is_hit = true;
                closest_so_far = temp_info.t;
                rec = temp_info;
            }
        }
        return is_hit;
    }


    __host__ __device__ void AddSphere(point3 center,float radius)
    {
        printf("start add\n");

        kernel_add_Sphere<<<1,1>>>(*this, center,radius);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return ;
        }
        printf("end add\n");
    }
    __device__ void DeviceAddSphere(const point3& center, float radius){
        if (num_objects < max_objects) {
            Sphere* sphere = new Sphere(center, radius);
            objects[num_objects++] = sphere;
        } else {
            printf("Hittable_list is full, cannot add more objects.\n");
        }
    }

};

__global__ void init_hittable_list(Hittable_list* d_world);

__global__ void add_sphere_to_list(Hittable_list* d_world, point3 center, float radius) ;


#endif //HITTABLE_LIST_CUH
