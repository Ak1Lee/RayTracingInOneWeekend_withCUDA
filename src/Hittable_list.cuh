//
// Created by 24977 on 2024/11/26.
//

#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "hittable.cuh"
#include <vector>
#include <memory>

using std::make_shared;
using std::shared_ptr;

class Hittable_list
{
public:
    // std::vector<shared_ptr<Hittable>> objects;

    static const int max_objects = 20;
    Hittable** objects;
    int num_objects = 0;


    __host__ __device__ Hittable_list() {
        cudaMalloc(&objects, max_objects * sizeof(Hittable**));
    };

    // __host__ __device__ Hittable_list(Hittable* object){add(object); }
    __host__ __device__ ~Hittable_list() = default;


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
                printf("world hit fun get \n");
                is_hit = true;
                closest_so_far = temp_info.t;
                rec = temp_info;
            }
        }
        return is_hit;


    }

};



#endif //HITTABLE_LIST_CUH
