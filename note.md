
```
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
```

```
__global__ void delete_function(Hittable** hittable)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        delete *hittable;
    }
}
```

```
__device__ color ray_color_sample_test(const Ray& r, Hittable const* const* __restrict__ hittables, int depth)
{

    if ((*hittables)->hit(r, ray_t, rec)) 
    ...
```