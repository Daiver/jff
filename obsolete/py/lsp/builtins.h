#ifndef BUILTINS_H
#define BUILTINS_H

#include <iostream>
#include <vector>
#include <memory>
#include <cstdarg>

namespace builtins {

template<class T>
using LVector = std::shared_ptr<std::vector<T>>;

/*template<class T>*/
//LVector<T> createVector(int count, ...)
//{
    //auto res = std::make_shared<std::vector<T>>(count);
    //va_list list;
    //va_start(list, count);
    //for(int i = 0; i < count; ++i){
        //(*res)[i] = va_arg(list, T);
    //}
    //va_end(list);
/*}*/

template<class T>
float toFloat(const T &x)
{
    return (float)x;
}

template<class T>
double toDouble(const T &x)
{
    return (double)x;
}

template<class T>
int toInt(const T &x)
{
    return (int)x;
}

template<class T>
inline void print(const T &x)
{
    std::cout << x;
}

template<class T>
inline void println(const T &x)
{
    print(x);
    std::cout << std::endl;
}

template<>
inline void print<bool>(const bool &x)
{
    std::cout << (x ? "True" : "False");
}

template<class T>
inline void print(const LVector<T> &v)
{
    std::cout << "[";
    for(auto &x : v){
        print(x);
        std::cout << ", ";
    }
    std::cout << "]";
}

template<class T>
inline T opAdd(T a, T b)
{
    return a + b;
}

template<class T>
inline T opMul(T a, T b)
{
    return a * b;
}

template<class T>
inline T opSub(T a, T b)
{
    return a - b;
}

template<class T>
inline T opDiv(T a, T b)
{
    return a / b;
}

template<class T>
inline bool opGreaterThan(const T &a, const T &b)
{
    return a > b;
}

template<class T>
inline bool opLessThan(const T &a, const T &b)
{
    return a < b;
}

template<class T>
inline bool opEqual(const T &a, const T &b)
{
    return a == b;
}

template<class T>
inline bool opNotEqual(const T &a, const T &b)
{
    return a != b;
}


}

#endif
