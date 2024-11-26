//
// Created by 24977 on 2024/11/26.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "vec3.cuh"
#include "color.cuh"
#include "film.cuh"
#include "randf.cuh"


struct camera_to_renderer_info {
	vec3 pixel00_loc,center,pixel_delta_u,pixel_delta_v;
	camera_to_renderer_info(vec3 pixel00_loc, vec3, vec3 pixel_delta_u, vec3 pixel_delta_v) {};
	camera_to_renderer_info() {};
};

class Camera
{
public:

	Camera(int image_width, int image_height);
	~Camera(){};
	float aspect_ratio = 1.f;
	int image_width = 100;
	int image_height = 100;
	int samples_per_pixel = 10;   // Count of random samples for each pixel
	double vfov_degree = 90;  // Vertical view angle (field of view)
	point3 lookfrom = point3(0, 0, 0);
	point3 lookat = point3(0, 0, -1);
	vec3 vup = vec3(0, 1, 0);             //"up"dir
	float defocus_angle = 0.f;
	float focus_dist = 10;

	// void render(const Hittable_list& world);

	void write_color_(color& pixel_color, std::shared_ptr<unsigned char[]>& buffer_copy, int i, int j);

	void write_color(unsigned char*& p, const color& pixel_color) {
		auto r = pixel_color.x();
		auto g = pixel_color.y();
		auto b = pixel_color.z();

		// Translate the [0,1] component values to the byte range [0,255].
		unsigned char uc_r = unsigned char(255.999 * r);
		unsigned char uc_g = unsigned char(255.999 * g);
		unsigned char uc_b = unsigned char(255.999 * b);

		// Write out the pixel color components.
		*p++ = uc_r;
		*p++ = uc_g;
		*p++ = uc_b;
	}

	Ray get_ray(int i, int j)const;

	vec3 sample_square() const {
		// Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
		return vec3(random_float() - 0.5, random_float() - 0.5, 0.f);
	}

	void GeneratePNG(std::string filename);

	camera_to_renderer_info* get_render_info() {
		return &renderer_info;
	};

private:

	Film film;

	float focal_length = 1.0f;
	point3 pixel00_loc;
	point3 center;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;
	int max_depth = 10;
	vec3   u, v, w;              // Camera frame basis vectors
	vec3   defocus_disk_u;       // Defocus disk horizontal radius
	vec3   defocus_disk_v;       // Defocus disk vertical radius

	camera_to_renderer_info renderer_info;


	void initialize();

	point3 defocus_disk_sample() const {

		// TODO
		// Returns a random point in the camera defocus disk.
		// auto p = random_in_unit_disk();
		// return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
		return point3(1,1,1);
	}
	// color ray_color(const Ray&r, const Hittable& hittable);
	// color ray_color_multiple_sample(const Ray& r, const Hittable& world,int depth);

};



#endif //CAMERA_CUH
