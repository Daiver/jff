#ifndef AD_H
#define AD_H

template<typename T>
class AD
{
public:
    AD(T val, T der): der(der), val(val) {}

    T val;
    T der;
};

template<typename T>
AD<T> operator* (T a, T b)
{
    return AD<T>(a.val * b.val, a.val*b.der + a.der*b.val);
}

template<typename T>
AD<T> operator+ (T a, T b)
{
    return AD<T>(a.val + b.val, b.der + a.der);
}

#endif
