#include <stdio.h>

template<int I>
class A
{
public:
    void foo() const {return I;}
};

template<int J>
class B
{
public:
    static const int value = J;
};

template<typename T>
A<T::value> func(T tmp)
{
    
}

int main()
{
    func(B<13>());
    return 0;
}
