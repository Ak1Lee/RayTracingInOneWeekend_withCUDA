//
// Created by 24977 on 2024/11/25.
//

#ifndef VEC3_CUH
#define VEC3_CUH

#include <iostream>
#include <cmath>
#include <curand_kernel.h>

#include "randf.cuh"


class vec3
{
public:
	float e[3];

	__host__ __device__ vec3() : e{ 0, 0, 0 } {};
	__host__ __device__ vec3(float x, float y, float z) :e{ x,y,z } {};

	__host__ __device__ static vec3 random();

	__host__ __device__ static vec3 random(float min, float max, curandState* state = nullptr);


	__host__ __device__ float x()const {
		return e[0];
	}
	__host__ __device__ float y()const {
		return e[1];
	}
	__host__ __device__ float z()const {
		return e[2];
	}

	__host__ __device__ vec3 operator - ()const
	{
		return vec3(-e[0], -e[1], -e[2]);
	}

	__host__ __device__ float operator[](int i)const
	{
		return e[i];
	}

	__host__ __device__ float& operator[](int i)
	{
		return e[i];
	}

	__host__ __device__ vec3& operator+=(const vec3& other)
	{
		e[0] += other.e[0];
		e[1] += other.e[1];
		e[2] += other.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator+=(const float& other)
	{
		e[0] += other;
		e[1] += other;
		e[2] += other;
		return *this;
	}

	__host__ __device__ vec3& operator-=(const vec3& other)
	{
		e[0] -= other.e[0];
		e[1] -= other.e[1];
		e[2] -= other.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator-=(const float& other)
	{
		e[0] -= other;
		e[1] -= other;
		e[2] -= other;
		return *this;
	}

	__host__ __device__ vec3& operator*=(float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	__host__ __device__ vec3& operator*=(vec3& other) {
		e[0] *= other.e[0];
		e[1] *= other.e[1];
		e[2] *= other.e[2];
		return *this;
	}
	__host__ __device__ vec3& operator/=(float t) {
		return *this *= 1 / t;
	}

	__host__ __device__ float length() const {
		return std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}

	__host__ __device__ float length_squared()const
	{
		return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}

	__host__ __device__ bool near_zero()const
	{
		float s = 1e-8;
		return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
	}

	__host__ __device__ vec3 normalize()
	{
		*this /= ((*this).length());
		return *this;
	}
};


using point3 = vec3;

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3& u, const float& v) {
	return vec3(u.e[0] + v, u.e[1] + v, u.e[2] + v);
}
__host__ __device__ inline vec3 operator+(const float& v, const vec3& u) {
	return vec3(u.e[0] + v, u.e[1] + v, u.e[2] + v);
}
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
	return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
	return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
	return v / v.length();
}

__host__ __device__ inline vec3 random_unit_vector(curandState* state = nullptr) {
	float theta, phi;
	float M_PI = 3.14159f;
#ifdef __CUDA_ARCH__
	theta = random_float(0, 2 * M_PI, state);
	phi = random_float(0, M_PI, state);
#else

	theta = random_float(0, 2 * M_PI);
	phi = random_float(0, M_PI);
#endif

	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	return vec3(x, y, z);
}

__host__ __device__ inline vec3 random_on_hemisphere(const vec3& normal,curandState* state = nullptr) {
	vec3 on_unit_sphere = random_unit_vector(state);
	if (dot(on_unit_sphere, normal) > 0.f) // In the same hemisphere as the normal
		return on_unit_sphere;
	else
		return -on_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& normal) {
	return v - 2 * dot(v, normal) * normal;
}

__host__ __device__ inline vec3 refract(const vec3& ray_in, const vec3& n, double etai_over_etat)
{
	auto cos_theta1 = std::fmin(dot(-ray_in, n),1.0f);
	auto sin_theta1 = std::sqrt(1.0 - cos_theta1 * cos_theta1);


	auto ray_ref_pal = etai_over_etat * (ray_in + cos_theta1 * n);
	auto ray_ref_hor = -std::sqrt(1 - etai_over_etat * etai_over_etat * (1 - cos_theta1 * cos_theta1)) * n;

	return ray_ref_hor + ray_ref_pal;
}






#endif //VEC3_CUH
