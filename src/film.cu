//
// Created by 24977 on 2024/11/26.
//

#include "film.cuh"
#include "interval.cuh"
#include "../thirdparty/svpng.inc"

Film::Film(int width, int height):width(width),height(height)
{
    buffer = new unsigned char[width*height*3];


}

void Film::write_color(int x, int y, color& pixel_color)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r = color_linear_to_gamma(r);
    g = color_linear_to_gamma(g);
    b = color_linear_to_gamma(b);

    static const Interval interv{ 0.0f,0.9f };

    unsigned char uc_r = unsigned char(255.999 * interv.clamp(r));
    unsigned char uc_g = unsigned char(255.999 * interv.clamp(g));
    unsigned char uc_b = unsigned char(255.999 * interv.clamp(b));
    int idx = (y * width + x) * 3;

    buffer[idx] = uc_r;
    buffer[idx + 1] = uc_g;
    buffer[idx + 2] = uc_b;
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
