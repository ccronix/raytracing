#include <cmath>
#include <iostream>

#include <cuda_runtime.h>


template<typename T>
class vec4
{
public:
    T content[4];

public:
    __host__ __device__ vec4() : content{0., 0., 0., 0.} {}
    __host__ __device__ vec4(T x, T y, T z, T w) : content{x, y, z, w} {}
    __host__ __device__ vec4(const vec4& v) {
        content[0] = v.content[0];
        content[1] = v.content[1];
        content[2] = v.content[2];
        content[3] = v.content[3];
    }
    __host__ __device__ ~vec4() {}

    __host__ __device__ T x() const { return content[0]; }
    __host__ __device__ T y() const { return content[1]; }
    __host__ __device__ T z() const { return content[2]; }
    __host__ __device__ T w() const { return content[3]; }

    __host__ __device__ T length() const {
        return sqrt(pow(content[0], 2) + pow(content[1], 2) + pow(content[2], 2) + pow(content[3], 2));
    }

    __host__ __device__ vec4 operator - () const {
            return vec4(- content[0], - content[1], - content[2], - content[3]);
        }
    __host__ __device__ T operator [] (int i) const { return content[i]; }
    __host__ __device__ T& operator [] (int i) { return content[i]; }
    
    __host__ __device__ vec4& operator += (const vec4 &v) {
        content[0] += v.content[0];
        content[1] += v.content[1];
        content[2] += v.content[2];
        content[3] += v.content[3];
        return *this;
    }

    __host__ __device__ vec4& operator -= (const vec4 &v) {
        content[0] -= v.content[0];
        content[1] -= v.content[1];
        content[2] -= v.content[2];
        content[3] -= v.content[3];
        return *this;
    }

    __host__ __device__ vec4& operator *= (const T v) {
        content[0] *= v;
        content[1] *= v;
        content[2] *= v;
        content[3] *= v;
        return *this;
    }

    __host__ __device__ vec4& operator /= (const T v) {
        return *this *= 1 / v;
    }

    __host__ __device__ vec4 normalize(vec4 v ) const {
        return v / v.length(); 
    }

    __host__ __device__ T dot(const vec4 v) const {
        T r1 = content[0] * v.content[0];
        T r2 = content[1] * v.content[1];
        T r3 = content[2] * v.content[2];
        T r4 = content[3] * v.content[3];
        return r1 + r2 + r3 + r4;
    }

    __host__ __device__ vec4 cross(const vec4 v) const {
        T r1 = content[1] * v.content[2] - content[2] * v.content[1];
        T r2 = content[2] * v.content[0] - content[0] * v.content[2];
        T r3 = content[0] * v.content[1] - content[1] * v.content[0];
        return vec4(r1, r2, r3, 1.);
    }

    friend std::ostream& operator << (std::ostream &out, const vec4 v) {
        return out << "Vector3f(" << v.content[0] << ", " << v.content[1] << ", " << v.content[2] << ", " << v.content[3] << ")";
    }

    __host__ __device__ friend vec4 operator + (const vec4 v1, const vec4 v2) {
        return vec4(
            v1.content[0] + v2.content[0], 
            v1.content[1] + v2.content[1],
            v1.content[2] + v2.content[2],
            v1.content[3] + v2.content[3]
        );
    }

    __host__ __device__ friend vec4 operator - (const vec4 v1, const vec4 v2) {
        return vec4(
            v1.content[0] - v2.content[0], 
            v1.content[1] - v2.content[1],
            v1.content[2] - v2.content[2],
            v1.content[3] - v2.content[3]
        );
    }

    __host__ __device__ friend vec4 operator * (const vec4 v1, const vec4 v2) {
        return vec4(
            v1.content[0] * v2.content[0], 
            v1.content[1] * v2.content[1],
            v1.content[2] * v2.content[2],
            v1.content[3] * v2.content[3]
        );
    }

    __host__ __device__ friend vec4 operator * (const T v1, const vec4 v2) {
        return vec4(
            v1 * v2.content[0], 
            v1 * v2.content[1],
            v1 * v2.content[2],
            v1 * v2.content[3]
        );
    }

    __host__ __device__ friend vec4 operator * (const vec4 v1, const T v2) {
        return v2 * v1;
    }

    __host__ __device__ friend vec4 operator / (const vec4 v1, const T v2) {
        return (1 / v2) * v1;
    }
};
