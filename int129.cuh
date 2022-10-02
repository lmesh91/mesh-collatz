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

//This class is taken from https://github.com/curtisseizert/CUDA-uint128/blob/master/cuda_uint128.h, with some slight modifications for signed integers.
//There's probably some bugs in here that just aren't used by the program, so be careful if you decide to use this yourself.
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
        if (a.sign == POSITIVE) return uge129(a, b);
        else return ule129(a, b);
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

    //Again, we do unsigned operators first. And addition and subtraction.

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
        } else if (x.sign > y.sign) {
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
        return res;
    }

    template <typename T>
    __host__ __device__ static inline int129 add129(int129 x, T y) {
        int129 res;
        bool ysign = y >= 0 ? POSITIVE : NEGATIVE;
        y = y >= 0 ? y : -y;
        if (x.sign == ysign) {
            res = uadd129(x, y); //sign is preserved
            res.sign = x.sign;
        } else if (x.sign > ysign) {
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
        return res;
    }

    template <typename T>
    __host__ __device__ friend int129 operator+(int129 a, const T& b) { return add129(a, b); }
    
    template <typename T>
    __host__ __device__ inline int129 operator+=(const T& b) { return add129(*this, b); }
    
    __host__ __device__ inline int129 operator++() { return *this += 1; }
    

    //Now subtraction is comparatively easy...
    __host__ __device__ static inline int129 sub129(int129 x, int129 y) {
        y.sign = !y.sign;
        return add129(x, y);
    }

    template <typename T>
    __host__ __device__ static inline int129 sub129(int129 x, T y) {
        return add129(x, -y);
    }
    
    template <typename T>
    __host__ __device__ friend int129 operator-(int129 a, const T& b) { return sub129(a, b); }
    
    template <typename T>
    __host__ __device__ inline int129 operator-=(const T& b) { return sub129(*this, b); }
    
    __host__ __device__ inline int129 operator--() { return *this += 1; }

    //Alright, onto multiplication and division!
    
    //First, unsigned operations.
    __device__ static inline int129 umul129(int129 x, int129 y) {
        int129 res;
        res.lo = x.lo * y.lo;
        res.hi = __umul64hi(x.lo, y.lo);
        res.hi += x.hi * y.lo + x.lo * y.hi;
        return res;
    }

    __device__ static inline int129 umul129(int129 x, unsigned long long y) {
        int129 res;
        res.lo = x.lo * y;
        res.hi = __umul64hi(x.lo, y);
        res.hi += x.hi * y;
        return res;
    }

    // taken from libdivide's adaptation of this implementation origininally in
    // Hacker's Delight: http://www.hackersdelight.org/hdcodetxt/divDouble.c.txt
    // License permits inclusion here per:
    // http://www.hackersdelight.org/permissions.htm
    __device__ static inline unsigned long long udiv129to64(int129 x, unsigned long long v, unsigned long long * r = NULL) // x / v
    {
        const unsigned long long b = 1ull << 32;
        unsigned long long  un1, un0,
                vn1, vn0,
                q1, q0,
                un64, un21, un10,
                rhat;
        int s;

        if(x.hi >= v){
        if( r != NULL) *r = (unsigned long long) -1;
        return  (unsigned long long) -1;
        }

        s = __clzll(v);

        if(s > 0){
        v = v << s;
        un64 = (x.hi << s) | ((x.lo >> (64 - s)) & (-s >> 31));
        un10 = x.lo << s;
        }else{
        un64 = x.lo | x.hi;
        un10 = x.lo;
        }

        vn1 = v >> 32;
        vn0 = v & 0xffffffff;

        un1 = un10 >> 32;
        un0 = un10 & 0xffffffff;

        q1 = un64/vn1;
        rhat = un64 - q1*vn1;

    again1:
        if (q1 >= b || q1*vn0 > b*rhat + un1){
        q1 -= 1;
        rhat = rhat + vn1;
        if(rhat < b) goto again1;
        }

        un21 = un64*b + un1 - q1*v;

        q0 = un21/vn1;
        rhat = un21 - q0*vn1;
    again2:
        if(q0 >= b || q0 * vn0 > b*rhat + un0){
        q0 = q0 - 1;
        rhat = rhat + vn1;
        if(rhat < b) goto again2;
        }

        if(r != NULL) *r = (un21*b + un0 - q0*v) >> s;
        return q1*b + q0;
    }

    __device__ static inline int129 udiv129(int129 x, unsigned long long v, unsigned long long *r = NULL) {
        int129 res;
        res.hi = x.hi/v;
        x.hi %= v;
        res.lo = udiv129to64(x, v, r);
        return res;
    }

    //Signed operators are much easier this time.
    __device__ static inline int129 mul129(int129 x, int129 y) {
        int129 res = umul129(x, y);
        if (x.sign == y.sign) res.sign = POSITIVE;
        else res.sign = NEGATIVE;
        return res;
    }
    
    __device__ static inline int129 mul129(int129 x, long long y) {
        int129 res = umul129(x, (y >= 0 ? y : -y));
        if (x.sign == (y >= 0 ? POSITIVE : NEGATIVE)) res.sign = POSITIVE;
        else res.sign = NEGATIVE;
        return res;
    }

    template <typename T>
    __device__ friend int129 operator*(int129 a, const T b) { return mul129(a, b); }
    template <typename T>
     __device__ inline int129& operator*=(const T* b) { return mul129(*this, b); }

     __device__ static inline int129 div129(int129 x, long long y) {
        int129 res = udiv129(x, y);
        if (x.sign == (y >= 0 ? POSITIVE : NEGATIVE)) res.sign = POSITIVE;
        else res.sign = NEGATIVE;
        return res;
    }

    template <typename T>
     __device__ friend int129 operator/(int129 a, const T* b) { return div129(a, *b); }
    template <typename T>
     __device__ inline int129& operator/=(const T* b) { int129 x = *this; return x / *b; }

    //modulo becomes really easy now, with how division is defined
    template <typename T>
     __device__ friend T operator%(int129 x, const T& v) {
        unsigned long long ures;
        udiv129to64(x, v, &ures);
        if ((v >= 0) ? (x.sign == POSITIVE) : (x.sign == NEGATIVE)) {
            return (T)ures;
        } else {
            return -1 * (T)ures;
        }
    }

    template <typename T>
     __device__ inline int129& operator%=(const T& v) {
        int129 x = *this;
        return x % v;
    }

    //>> is also used
    template <typename T>
    __host__ __device__ inline int129& operator>>=(const T& b)
    {
        if (b < 64) {
            lo = (lo >> b) | (hi << (64-b));
            hi >>= b;
        } else {
            lo = hi >> (b-64);
            hi = 0;
        }
        return *this;
    }
    template <typename T>
    __host__ __device__ friend inline int129 operator>>(int129 a, const T & b){a >>= b; return a;}
    
};