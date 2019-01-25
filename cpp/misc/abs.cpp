#include <iostream>
#include <cmath>

class A
{
public:
    float i;
};

namespace std {
    A abs(const A &a){
        A b = a;
        b.i = std::abs(a.i);
        return b;
    }
}

int main()
{
    float f = -10.0;
    double d = -10.0;
    A a;
    a.i = -10;
    std::cout << std::abs(f) << " " << std::abs(d) << " " << std::abs(a).i << std::endl;
    return 0;
}
