#include <iostream>

#include "barycentric.h"

void foo()
{
    float l1, l2, l3;
    barycoords_from_2d_triangle<float>(
        0, 0,
        1, 0,
        0, 1,
        0.5, 0.5,
        l1, l2, l3);
    std::cout << "HI! " << l1 << " " << l2 << " " << l3 << std::endl;
}
