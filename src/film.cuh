//
// Created by 24977 on 2024/11/26.
//

#ifndef FILM_CUH
#define FILM_CUH

#include <string>

#include "color.cuh"

class Film
{
public:
    Film(int width,int height);
    ~Film() {
        if (buffer) {
            cudaFree(buffer);
        }
    };


    void write_color(int x, int y, color& pixel_color);

    void save_as_png(std::string filename);

    unsigned char* get_data();
private:
    unsigned char* buffer;
    int width;
    int height;

};



#endif //FILM_CUH
