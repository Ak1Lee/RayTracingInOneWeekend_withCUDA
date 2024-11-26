//
// Created by 24977 on 2024/11/26.
//

#include "Hittable_list.cuh"
// __global__ void kernel_add_Sphere(Hittable_list& world,point3 center,float radius)
// {
//     if(world.num_objects>=world.max_objects)return;
//     if (threadIdx.x == 0 && blockIdx.x == 0)
//     {
//         world.objects[world.num_objects++] = new Sphere(center, radius);
//         // printf("num_objects: %d\n", world.num_objects);
//     }
// }

__global__ void kernel_add_Sphere(Hittable_list* world, point3 center, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (world->num_objects >= world->max_objects) return;
        Sphere* sphere = new Sphere(center, radius); // 动态分配设备对象
        world->objects[world->num_objects++] = sphere;
    }
}

__global__ void add_sphere_to_list(Hittable_list* d_world, point3 center, float radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world->DeviceAddSphere(center, radius);
    }
}

__global__ void init_hittable_list(Hittable_list* d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(d_world) Hittable_list();
    }
}