#include <cmath>
#include <iostream>

#include <cuda_runtime.h>


template<typename T>
class vec3
{
public:
    T content[3];

public:
    __host__ __device__ vec3() : content{0., 0., 0.} {}
    __host__ __device__ vec3(T x, T y, T z) : content{x, y, z} {}
    __host__ __device__ vec3(const vec3& v) {
        content[0] = v.content[0];
        content[1] = v.content[1];
        content[2] = v.content[2];
    }
    __host__ __device__ ~vec3() {}

    __host__ __device__ T x() const { return content[0]; }
    __host__ __device__ T y() const { return content[1]; }
    __host__ __device__ T z() const { return content[2]; }

    __host__ __device__ T length() const {
        return sqrt(pow(content[0], 2) + pow(content[1], 2) + pow(content[2], 2));
    }

    __host__ __device__ vec3 operator - () const {
            return vec3(- content[0], - content[1], - content[2]);
        }
    __host__ __device__ T operator [] (int i) const { return content[i]; }
    __host__ __device__ T& operator [] (int i) { return content[i]; }
    
    __host__ __device__ vec3& operator += (const vec3 &v) {
        content[0] += v.content[0];
        content[1] += v.content[1];
        content[2] += v.content[2];
        return *this;
    }

    __host__ __device__ vec3& operator -= (const vec3 &v) {
        content[0] -= v.content[0];
        content[1] -= v.content[1];
        content[2] -= v.content[2];
        return *this;
    }

    __host__ __device__ vec3& operator *= (const T v) {
        content[0] *= v;
        content[1] *= v;
        content[2] *= v;
        return *this;
    }

    __host__ __device__ vec3& operator /= (const T v) {
        return *this *= 1 / v;
    }

    __host__ __device__ vec3 normalize() const {
        return *this / length(); 
    }

    __host__ __device__ T dot(const vec3 v) const {
        T r1 = content[0] * v.content[0];
        T r2 = content[1] * v.content[1];
        T r3 = content[2] * v.content[2];
        return r1 + r2 + r3;
    }

    __host__ __device__ vec3 cross(const vec3 v) const {
        T r1 = content[1] * v.content[2] - content[2] * v.content[1];
        T r2 = content[2] * v.content[0] - content[0] * v.content[2];
        T r3 = content[0] * v.content[1] - content[1] * v.content[0];
        return vec3(r1, r2, r3);
    }

    friend std::ostream& operator << (std::ostream &out, const vec3 v) {
        return out << "vec3(" << v.content[0] << ", " << v.content[1] << ", " << v.content[2] << ")";
    }

    __host__ __device__ friend vec3 operator + (const vec3 v1, const vec3 v2) {
        return vec3(
            v1.content[0] + v2.content[0], 
            v1.content[1] + v2.content[1],
            v1.content[2] + v2.content[2]
        );
    }

    __host__ __device__ friend vec3 operator - (const vec3 v1, const vec3 v2) {
        return vec3(
            v1.content[0] - v2.content[0], 
            v1.content[1] - v2.content[1],
            v1.content[2] - v2.content[2]
        );
    }

    __host__ __device__ friend vec3 operator * (const vec3 v1, const vec3 v2) {
        return vec3(
            v1.content[0] * v2.content[0], 
            v1.content[1] * v2.content[1],
            v1.content[2] * v2.content[2]
        );
    }

    __host__ __device__ friend vec3 operator * (const T v1, const vec3 v2) {
        return vec3(
            v1 * v2.content[0], 
            v1 * v2.content[1],
            v1 * v2.content[2]
        );
    }

    __host__ __device__ friend vec3 operator * (const vec3 v1, const T v2) {
        return v2 * v1;
    }

    __host__ __device__ friend vec3 operator / (const vec3 v1, const T v2) {
        return (1 / v2) * v1;
    }
};
