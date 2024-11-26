//
// Created by 24977 on 2024/11/26.
//

#include "film.cuh"
#include "interval.cuh"
#include "../thirdparty/svpng.inc"

Film::Film(int width, int height):width(width),height(height)
{
    size_t buffer_size = 3 * width * height * sizeof(unsigned char); // 每个像素 3 个通道
    cudaError_t err = cudaMallocManaged(&buffer, buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate unified memory for Film buffer: "
                  << cudaGetErrorString(err) << std::endl;
        buffer = nullptr;
        exit(EXIT_FAILURE);
    }


}

void Film::write_color(int x, int y, color& pixel_color)
{
    int index = (y * width + x) * 3;
    buffer[index]     = static_cast<unsigned char>(255.999f * pixel_color.x());
    buffer[index + 1] = static_cast<unsigned char>(255.999f * pixel_color.y());
    buffer[index + 2] = static_cast<unsigned char>(255.999f * pixel_color.z());
}

void Film::save_as_png(std::string filename)
{
    // open file
    FILE* fp;
    fopen_s(&fp, filename.c_str(), "wb");
    svpng(fp, width, height, buffer, 0);
    fclose(fp);
    std::clog << "\rSave File in " << filename << "\n";

}

unsigned char * Film::get_data() {
    return buffer;
}
