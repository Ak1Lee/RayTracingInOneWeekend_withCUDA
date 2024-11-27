//
// Created by 24977 on 2024/11/26.
//

#include "Hittable_list.cuh"
#include "randf.cuh"
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
    curandState curand_state;
    curand_init(1132, threadIdx.x, 0, &curand_state);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(d_world) Hittable_list();
        // Sphere* sphere1 = new Sphere({0.f, -100.5f, -1.f}, 100.f); // 动态分配设备对象
        auto* material_ground = new Lambertion({0.5, 0.5, 0.5});
        // d_world->objects[d_world->num_objects++] = new Sphere({0.f, -1000.f, 0.f}, 1000.f,material_ground);
        // auto* material_center = new Lambertion({0.1, 0.2, 0.5});
        // auto* material_left   = new Dielectric(1.5f);
        // // auto* material_bubble  = new Dielectric(1.f/1.5f);
        // auto* material_right  = new Metal({0.8, 0.6, 0.2},1);
        //
        d_world->objects[d_world->num_objects++] = new Sphere({0.f, -1000.f, -1.f}, 1000.f,material_ground);
        // d_world->objects[d_world->num_objects++] = new Sphere({0.0f, 0.0f, -1.2f}, 0.5f,material_center);
        // d_world->objects[d_world->num_objects++] = new Sphere({-1.f, 0.0f, -1.0f}, 0.5f,material_left);
        // d_world->objects[d_world->num_objects++] = new Sphere({-1.f, 0.0f, -1.0f}, 0.4f,material_bubble);
        //
        // d_world->objects[d_world->num_objects++] = new Sphere({1.0f, 0.0f, -1.0f}, 0.5f,material_right);

        for(int a = -8;a<8;a++) {
            for(int b = -8;b<8;b++) {
                auto choose_mat = random_float(0.f,1.f,&curand_state);
                point3 center(a + 0.9*random_float(0.f,1.f,&curand_state), 0.2, b + 0.9*random_float(0.f,1.f,&curand_state));
                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    if(choose_mat < 0.8f) {
                        auto albedo = vec3::random(0.f,1.f,&curand_state);
                        auto* material = new Lambertion(albedo);
                        d_world->objects[d_world->num_objects++] = new Sphere(center, 0.2f,material);
                    } else if (choose_mat < 0.95) {

                        auto albedo = vec3::random(0.5f,1.f,&curand_state);
                        auto fuzz = random_float(0.f,0.5f,&curand_state);
                        auto* material = new Metal(albedo, fuzz);
                        d_world->objects[d_world->num_objects++] = new Sphere(center, 0.2f,material);

                    }else {
                        auto* material = new Dielectric(1.5f);
                        d_world->objects[d_world->num_objects++] = new Sphere(center, 0.2f,material);


                    }
                }

            }
        }
        auto material1 = new Dielectric(1.5f);
        d_world->objects[d_world->num_objects++] = new Sphere(vec3(0.f,1.f,0.f), 1.f,material1);
        auto material2 = new Lambertion(vec3(0.4f,0.2f,0.1f));
        d_world->objects[d_world->num_objects++] = new Sphere(vec3(-4.f,1.f,0.f), 1.f,material2);
        auto material3 = new Metal(vec3(0.7f,0.6f,0.5f),0.0f);
        d_world->objects[d_world->num_objects++] = new Sphere(vec3(4.f,1.f,0.f), 1.f,material3);


    }
}