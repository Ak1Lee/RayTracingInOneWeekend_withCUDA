//
// Created by 24977 on 2024/11/26.
//

#include "camera.cuh"

Camera::Camera(int image_width, int image_height) :image_width(image_width), image_height(image_height), film(image_width, image_height) {
    initialize();
    renderer_info.center = center;
    renderer_info.pixel00_loc = pixel00_loc;
    renderer_info.pixel_delta_u = pixel_delta_u;
    renderer_info.pixel_delta_v = pixel_delta_v;
};

void Camera::initialize()
{
    image_height = image_height < 1 ? 1 : image_height;


    center = lookfrom;
    focal_length = (lookfrom - lookat).length();


    float theta = vfov_degree * 3.1415926f / 180.f;
    float h = std::tan(theta / 2);
    //float viewport_height = 2 * h * focal_length;
    float viewport_height = 2 * h * focus_dist;

    //cal w u v;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);


    //auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (float(static_cast<float>(image_width) / image_height));

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    //auto viewport_u = vec3(viewport_width, 0.f, 0.f);
    //auto viewport_v = vec3(0.f, -viewport_height, 0.f);

    auto viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;

    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = center
        - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);


    // Calculate the camera defocus disk basis vectors.
    auto defocus_radius = focus_dist * std::tan((defocus_angle / 2) * (3.1415926f / 180.0));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
}