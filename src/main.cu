#include <iostream>
#include <book.h>
#include <cstdio>
#include <fstream>

#include "vec3.cuh"
#include "ray.cuh"


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
__device__ vec3 get_color(const Ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}
__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = get_color(r);
}
int main(){
    printf("Hello CUDA\n");
    int nx = 128;
    int ny = 128;
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float);
    vec3 *fb;

    checkCudaErrors(cudaMallocManaged((void**)&fb,fb_size));

    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny,
                            vec3(-2.0, -1.0, -1.0),
                            vec3(4.0, 0.0, 0.0),
                            vec3(0.0, 2.0, 0.0),
                            vec3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Output FB as Image
    std::ofstream fout;
    fout.open("output.ppm");
    fout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {

            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].x());
            int ig = int(255.99*fb[pixel_index].y());
            int ib = int(255.99*fb[pixel_index].z());
            fout << ir << " " << ig << " " << ib << "\n";
            std::cout << ir << " " << ig << " " << ib << "\n";

        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}