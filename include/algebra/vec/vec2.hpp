#include <cmath>
#include <iostream>


template<typename T>
class vec2
{
public:
    T content[2];

public:
    vec2() : content{0., 0.} {}
    vec2(T x, T y) : content{x, y} {}
    vec2(const vec2& v) {
        content[0] = v.content[0];
        content[1] = v.content[1];
    }
    ~vec2() {}

    T x() const { return content[0]; }
    T y() const { return content[1]; }

    T u() const { return content[0]; }
    T v() const { return content[1]; }


    T length() const {
        return sqrt(pow(content[0], 2) + pow(content[1], 2));
    }

    vec2 operator - () const {
            return vec2(- content[0], - content[1]);
        }
    T operator [] (int i) const { return content[i]; }
    T& operator [] (int i) { return content[i]; }
    
    vec2& operator += (const vec2 &v) {
        content[0] += v.content[0];
        content[1] += v.content[1];
        return *this;
    }

    vec2& operator -= (const vec2 &v) {
        content[0] -= v.content[0];
        content[1] -= v.content[1];
        return *this;
    }

    vec2& operator *= (const T v) {
        content[0] *= v;
        content[1] *= v;
        return *this;
    }

    vec2& operator /= (const T v) {
        return *this *= 1 / v;
    }

    vec2 normalize() const {
        return *this / length(); 
    }

    T dot(const vec2 v) const {
        T r1 = content[0] * v.content[0];
        T r2 = content[1] * v.content[1];
        return r1 + r2;
    }

    friend std::ostream& operator << (std::ostream &out, const vec2 v) {
        return out << "vec2(" << v.content[0] << ", " << v.content[1] << ")";
    }

    friend vec2 operator + (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] + v2.content[0], 
            v1.content[1] + v2.content[1]
        );
    }

    friend vec2 operator - (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] - v2.content[0], 
            v1.content[1] - v2.content[1]
        );
    }

    friend vec2 operator * (const vec2 v1, const vec2 v2) {
        return vec2(
            v1.content[0] * v2.content[0], 
            v1.content[1] * v2.content[1]
        );
    }

    friend vec2 operator * (const T v1, const vec2 v2) {
        return vec2(
            v1 * v2.content[0], 
            v1 * v2.content[1]
        );
    }

    friend vec2 operator * (const vec2 v1, const T v2) {
        return v2 * v1;
    }

    friend vec2 operator / (const vec2 v1, const T v2) {
        return (1 / v2) * v1;
    }
};
