#ifndef RASTER_BARYCENTRIC_H
#define RASTER_BARYCENTRIC_H

#include <vector>

/*

def barycoords_from_2d_triangle(tri_points, p):
    p1 = tri_points[0]
    p2 = tri_points[1]
    p3 = tri_points[2]

    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    l1 = (p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])
    l2 = (p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])
    l1 /= denom
    l2 /= denom
    l3 = 1.0 - l1 - l2
    return l1, l2, l3

*/

namespace Barycentric {

template<typename Scalar>
void barycoords_from_2d_triangle(
    const Scalar x1, const Scalar y1,
    const Scalar x2, const Scalar y2,
    const Scalar x3, const Scalar y3,
    const Scalar x, const Scalar y,
    Scalar &out_l1, Scalar &out_l2, Scalar &out_l3)
{
    const Scalar denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    out_l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator;
    out_l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator;
    out_l3 = Scalar(1.0) - out_l1 - out_l2;
}

inline std::vector<float> barycoords_from_2d_trianglef(
    const float x1, const float y1,
    const float x2, const float y2,
    const float x3, const float y3,
    const float x, const float y)
{
    float out_l1;
    float out_l2;
    float out_l3;
    barycoords_from_2d_triangle(
        x1, y1, x2, y2, x3, y3,
        x, y, out_l1, out_l2, out_l3);
    return {out_l1, out_l2, out_l3};
}

}

#endif // RASTER_BARYCENTRIC_H
