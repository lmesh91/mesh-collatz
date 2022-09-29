#include <iostream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <thread>

//libs just from uint128 lib
#include <iomanip>
#include <limits>
#include <cinttypes>
#include <cuda.h>
#include <cmath>
#include <vector>
#include <iterator>
#include <cuda_runtime.h>

//This makes the code more readable
#define POSITIVE true
#define NEGATIVE false

//This class is HEAVILY inspired from https://github.com/curtisseizert/CUDA-uint128/blob/master/cuda_uint128.h.
class int129 {
    public:
    unsigned long long lo, hi;
    bool sign;

    __host__ __device__ int129(){};

    //Initialization with smaller ints
    template <typename T>
    __host__ __device__ int129(const T& n) {
        this->hi = 0;
        this->lo = n >= 0 ? n : -n;
        this->sign = n >= 0 ? POSITIVE : NEGATIVE;
    }

    // In terms of operators - I'm implementing all of them except bitwise ones. (Unless, for some reason, I need one.)
    // I'm also not including more quality-of-life things like being able to print these. *for now...

    // = for int129
    __host__ __device__ int129& operator=(const int129& n) {
        this->lo = n.lo;
        this->hi = n.hi;
        this->sign = n.sign;
        return *this;
    }

    // = for smaller ints
    template <typename T>
    __host__ __device__ int129& operator=(const T& n) {
        this->hi = 0;
        this->lo = n >= 0 ? n : -n;
        this->sign = n >= 0 ? POSITIVE : NEGATIVE;
    }

    // Comparison: <, <=, >, >=, ==, !=

    // Unsigned operators, firstly.
    __host__ __device__ static bool ul129(int129 a, int129 b) {
        if (a.hi < b.hi) return 1;
        if (a.hi > b.hi) return 0;
        if (a.lo < b.lo) return 1;
        else return 0;
    }
    
    __host__ __device__ static bool ule129(int129 a, int129 b) {
        if (a.hi < b.hi) return 1;
        if (a.hi > b.hi) return 0;
        if (a.lo <= b.lo) return 1;
        else return 0;
    }
    
    __host__ __device__ static bool ug129(int129 a, int129 b) {
        if (a.hi < b.hi) return 0;
        if (a.hi > b.hi) return 1;
        if (a.lo <= b.lo) return 0;
        else return 1;
    }
    
    __host__ __device__ static bool uge129(int129 a, int129 b) {
        if (a.hi < b.hi) return 0;
        if (a.hi > b.hi) return 1;
        if (a.lo < b.lo) return 0;
        else return 1;
    }

    //And now the signed operators.
    __host__ __device__ static bool lt129(int129 a, int129 b) {
        if (a.sign < b.sign) return 1;
        if (a.sign > b.sign) return 0;
        if (a.sign == POSITIVE) return ul129(a, b);
        else return ug129(a, b);
    }
    __host__ __device__ friend bool operator<(int129 a, int129 b) { return lt129(a, b); }
    
    __host__ __device__ static bool leq129(int129 a, int129 b) {
        if (a.sign < b.sign) return 1;
        if (a.sign > b.sign) return 0;
        if (a.sign == POSITIVE) return ule129(a, b);
        else return uge129(a, b);
    }
    __host__ __device__ friend bool operator<=(int129 a, int129 b) { return leq129(a, b); }
    
    __host__ __device__ static bool gt129(int129 a, int129 b) {
        if (a.sign < b.sign) return 0;
        if (a.sign > b.sign) return 1;
        if (a.sign == POSITIVE) return ug129(a, b);
        else return ul129(a, b);
    }
    __host__ __device__ friend bool operator>(int129 a, int129 b) { return gt129(a, b); }
    
    __host__ __device__ static bool geq129(int129 a, int129 b) {
        if (a.sign < b.sign) return 0;
        if (a.sign > b.sign) return 1;
        if (a.sign == POSITIVE) return ug129(a, b);
        else return ul129(a, b);
    }
    __host__ __device__ friend bool operator>=(int129 a, int129 b) { return geq129(a, b); }

    //Note: No special cases for 0 and -0. I'll assume that I implement 0 such that it's always positive.
    __host__ __device__ static bool eq129(int129 a, int129 b) {
        if (a.lo == b.lo && a.hi == b.hi && a.sign == b.sign) return 1;
        else return 0;
    }
    __host__ __device__ friend bool operator==(int129 a, int129 b) { return eq129(a, b); }
    
    __host__ __device__ static bool neq129(int129 a, int129 b) {
        if (a.lo != b.lo || a.hi != b.hi || a.sign != b.sign) return 1;
        else return 0;
    }
    __host__ __device__ friend bool operator!=(int129 a, int129 b) { return neq129(a, b); }

    //Onto the arithmetic operations! +, -, *, and %.

    //Again, we do unsigned operators first.

    //ASM version (broken)
    /*__host__ __device__ static inline int129 uadd129(int129 x, int129 y) {
    int129 res;
    asm("add.cc.u64    %0, %2, %4;\n\t"
        "addc.u64      %1, %3, %5;\n\t"
        : "=l" (res.lo)  "=l" (res.hi)
        : "l" (x.lo), "l" (x.hi), "l" (y.lo), "l" (y.hi));
    return res;
    }*/

    __host__ __device__ static inline int129 uadd129(int129 x, int129 y) {
        int129 res;
        res.hi = x.hi + y.hi;
        res.lo = x.lo + y.lo;
        if (res.lo < x.lo) ++res.hi;
        return res;
    }

    //ASM version (broken)
    /*__host__ __device__ static inline int129 uadd129(int129 x, unsigned long long y) {
        int129 res;
        asm(  "add.cc.u64    %0, %2, %4;\n\t"
              "addc.u64      %1, %3, 0;\n\t"
              : "=l" (res.lo), "=l" (res.hi)
              : "l" (x.lo), "l" (x.hi),
                "l" (y));
        return res;
    }*/
    
    __host__ __device__ static inline int129 uadd129(int129 x, unsigned long long y) {
        int129 res;
        res.hi = x.hi;
        res.lo = x.lo + y;
        if (res.lo < x.lo) ++res.hi;
        return res;
    }

    __host__ __device__ static inline int129 usub129(int129 x, int129 y) {
        int129 res;
        res.lo = x.lo - y.lo;
        res.hi = x.hi - y.hi;
        if (x.lo < y.lo) res.hi--;
        return res;
    }
    
    __host__ __device__ static inline int129 usub129(int129 x, unsigned long long y) {
        int129 res;
        res.lo = x.lo - y;
        res.hi = x.hi;
        if (x.lo < y) res.hi--;
        return res;
    }

    // And now, the signed ones.
    __host__ __device__ static inline int129 add129(int129 x, int129 y) {
        int129 res;
        if (x.sign == y.sign) {
            res = uadd129(x, y); //sign is preserved
            res.sign = x.sign;
        }
        if (x.sign > y.sign) {
            if (uge129(x, y)) { // x - y is positive
                res = usub129(x, y);
                res.sign = x.sign;
            } else {
                res = usub129(y, x);
                res.sign = y.sign;
            }
        } else {
            if (uge129(y, x)) { // -x + y is positive
                res = usub129(y, x);
                res.sign = y.sign;
            } else {
                res = usub129(x, y);
                res.sign = x.sign;
            }
        }
    }

    template <typename T>
    __host__ __device__ static inline int129 add129(int129 x, T y) {
        int129 res;
        bool ysign = y >= 0 ? POSITIVE : NEGATIVE;
        if (x.sign == ysign) {
            res = uadd129(x, y); //sign is preserved
            res.sign = x.sign;
        }
        if (x.sign > ysign) {
            if (uge129(x, y)) { // x - y is positive
                res = usub129(x, y);
                res.sign = x.sign;
            } else {
                res = usub129(y, x);
                res.sign = ysign;
            }
        } else {
            if (uge129(y, x)) { // -x + y is positive
                res = usub129(y, x);
                res.sign = ysign;
            } else {
                res = usub129(x, y);
                res.sign = x.sign;
            }
        }
    }

    //Now subtraction is comparatively easy...
    __host__ __device__ static inline int129 sub129(int129 x, int129 y) {
        y.sign = !y.sign;
        return add129(x, y);
    }

    template <typename T>
    __host__ __device__ static inline int129 sub129(int129 x, T y) {
        return add129(x, -y);
    }
};