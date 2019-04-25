#include <torch/extension.h>

#include <iostream>
#include <vector>

#include "barycentric.h"

class Vector3f
{
public:
    Vector3f(const float x, const float y, const float z):
    m_values({x, y, z}) {}

private:
    float m_values[3];
};

/*

def rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d):
    n_rows = z_buffer.shape[0]
    n_cols = z_buffer.shape[1]
    tri_coords_xy = tri_coords_3d[:, :2]
    tri_coords_xy_int = tri_coords_xy.round().int()
    x_start = tri_coords_xy_int[:, 0].min()
    x_finish = tri_coords_xy_int[:, 0].max()
    y_start = tri_coords_xy_int[:, 1].min()
    y_finish = tri_coords_xy_int[:, 1].max()
    for x in range(x_start, x_finish + 1):
        if not (0 <= x < n_cols):
            continue
        for y in range(y_start, y_finish + 1):
            if not (0 <= y < n_rows):
                continue
            l1, l2, l3 = barycoords_from_2d_triangle(tri_coords_xy, (float(x), float(y)))
            is_l1_ok = 0.0 - 1e-7 <= l1 <= 1.0 + 1e-7
            is_l2_ok = 0.0 - 1e-7 <= l2 <= 1.0 + 1e-7
            is_l3_ok = 0.0 - 1e-7 <= l3 <= 1.0 + 1e-7
            if not (is_l1_ok and is_l2_ok and is_l3_ok):
                continue
            z_val = tri_coords_3d[0, 2] * l1 + tri_coords_3d[1, 2] * l2 + tri_coords_3d[2, 2] * l3
            if z_buffer[y, x] > z_val:
                continue
            barycentrics_l1l2l3[y, x, 0] = l1
            barycentrics_l1l2l3[y, x, 1] = l2
            barycentrics_l1l2l3[y, x, 2] = l3
            barycentrics_triangle_indices[y, x] = tri_index
            z_buffer[y, x] = z_val
*/

void rasterize_triangle(
    const int tri_index,
//    tri_coords_3d,
    torch::Tensor barycentrics_l1l2l3,
    torch::Tensor barycentrics_triangle_indices,
    torch::Tensor z_buffer)
{

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &lltm_forward, "LLTM forward");
//  m.def("backward", &lltm_backward, "LLTM backward");
//  m.def("foo", &foo, "Foo?");
  m.def("barycoords_from_2d_trianglef", &Barycentric::barycoords_from_2d_trianglef, "Foo?");
  m.def("rasterize_triangle", &rasterize_triangle, "");
}
