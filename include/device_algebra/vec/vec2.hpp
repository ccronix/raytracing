#include <cmath>
#include <iostream>

#include <cuda_runtime.h>


template<typename T>
class vec2
{
public:
    T content[2];

public:
    __host__ __device__ vec2() : content{0., 0.} {}
    __host__ __device__ vec2(T x, T y) : content{x, y} {}
    __host__ __device__ vec2(const vec2& v) {
        content[0] = v.content[0];
        content[1] = v.content[1];
    }
    ~vec2() {}

    __host__ __device__ T x() const { return content[0]; }
    __host__ __device__ T y() const { return content[1]; }

    __host__ __device__ T u() const { return content[0]; }
    __host__ __device__ T v() const { return content[1]; }


    __host__ __device__ T length() const {
        return sqrt(pow(content[0], 2) + pow(content[1], 2));
    }

    __host__ __device__ vec2 operator - () const {
            return vec2(- content[0], - content[1]);
        }
    __host__ __device__ T operator [] (int i) const { return content[i]; }
    __host__ __device__ T& operator [] (int i) { return content[i]; }
    
    __host__ __device__ vec2& operator += (const vec2 &v) {
        content[0] += v.content[0];
        content[1] += v.content[1];
        return *this;
    }

    __host__ __device__ vec2& operator -= (const vec2 &v) {
        content[0] -= v.content[0];
        content[1] -= v.content[1];
        return *this;
    }

    __host__ __device__ vec2& operator *= (const T v) {
        content[0] *= v;
        content[1] *= v;
        return *this;
    }

    __host__ __device__ vec2& operator /= (const T v) {
        return *this *= 1 / v;
    }

    __host__ __device__ vec2 normalize() const {
        return *this / length(); 
    }

    __host__ __device__ T dot(const vec2 v) const {
        T r1 = content[0] * v.content[0];
        T r2 = content[1] * v.content[1];
        return r1 + r2;
    }

    friend std::ostream& operator << (std::ostream &out, const vec2 v) {
        return out << "vec2(" << v.content[0] << ", " << v.content[1] << ")";
    }

    __host__ __device__ friend vec2 operator + (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] + v2.content[0], 
            v1.content[1] + v2.content[1]
        );
    }

    __host__ __device__ friend vec2 operator - (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] - v2.content[0], 
            v1.content[1] - v2.content[1]
        );
    }

    __host__ __device__ friend vec2 operator * (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] * v2.content[0], 
            v1.content[1] * v2.content[1]
        );
    }

    __host__ __device__ friend vec2 operator * (const T v1, const vec2 v2) {
        return vec2(
            v1 * v2.content[0], 
            v1 * v2.content[1]
        );
    }

    __host__ __device__ friend vec2 operator * (const vec2 v1, const T v2) {
        return v2 * v1;
    }

    __host__ __device__ friend vec2 operator / (const vec2 v1, const T v2) {
        return (1 / v2) * v1;
    }
};
