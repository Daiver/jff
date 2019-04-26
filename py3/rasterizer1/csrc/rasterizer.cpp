#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
//#include <math.h>

#include "barycentric.h"

template<typename Scalar>
class Vector2
{
public:
    Vector2() = default;
    Vector2(const Scalar x, const Scalar y)
    {
        m_values[0] = x;
        m_values[1] = y;
    }

    const Scalar operator[](const int index) const { return m_values[index]; }
    Scalar &operator[](const int index) { return m_values[index]; }

    template<typename Scalar2>
    Vector2<Scalar2> cast() const
    {
        return Vector2<Scalar2>(Scalar2(m_values[0]), Scalar2(m_values[1]));
    }

    Vector2<Scalar> round() const
    {
        return Vector2<Scalar>(std::round(m_values[0]), std::round(m_values[1]));
    }

private:
    Scalar m_values[2];
};

template<typename Scalar>
class Vector3
{
public:
    Vector3() = default;
    Vector3(const Scalar x, const Scalar y, const Scalar z)
    {
        m_values[0] = x;
        m_values[1] = y;
        m_values[2] = z;
    }

    const Scalar operator[](const int index) const { return m_values[index]; }
    Scalar &operator[](const int index) { return m_values[index]; }

    template<typename Scalar2>
    Vector3<Scalar2> cast() const
    {
        return Vector3<Scalar2>(Scalar2(m_values[0]), Scalar2(m_values[1]), Scalar2(m_values[2]));
    }

    Vector3<Scalar> round() const
    {
        return Vector3<Scalar>(std::round(m_values[0]), std::round(m_values[1]), std::round(m_values[2]));
    }

    Vector2<Scalar> xy() const
    {
        return Vector2<Scalar>(m_values[0], m_values[1]);
    }

private:
    Scalar m_values[3];
};

using Vector3f = Vector3<float>;
using Vector3i = Vector3<int>;

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
    const int64_t tri_index,
    const Vector3f &v1,
    const Vector3f &v2,
    const Vector3f &v3,
    torch::Tensor barycentrics_l1l2l3,
    torch::Tensor barycentrics_triangle_indices,
    torch::Tensor z_buffer)
{
    assert(z_buffer.dim() == 2);
    assert(barycentrics_l1l2l3.dim() == 3);
    assert(barycentrics_triangle_indices.dim() == 2);
    const int64_t n_rows = z_buffer.size(0);
    const int64_t n_cols = z_buffer.size(1);

    const auto v1_xy = v1.xy();
    const auto v1_xy_int = v1_xy.round().cast<int>();
    const auto v2_xy = v2.xy();
    const auto v2_xy_int = v2_xy.round().cast<int>();
    const auto v3_xy = v3.xy();
    const auto v3_xy_int = v3_xy.round().cast<int>();

    const int64_t x_start  = std::min({v1_xy_int[0], v2_xy_int[0], v3_xy_int[0]});
    const int64_t x_finish = std::max({v1_xy_int[0], v2_xy_int[0], v3_xy_int[0]});

    const int64_t y_start  = std::min({v1_xy_int[1], v2_xy_int[1], v3_xy_int[1]});
    const int64_t y_finish = std::max({v1_xy_int[1], v2_xy_int[1], v3_xy_int[1]});

    // TODO: it's possible to walk through not all triangle
    for(int64_t x = x_start; x < x_finish + 1; ++x){
        if(x < 0 || x >= n_cols)
            continue;
        for(int64_t y = y_start; y < y_finish + 1; ++y){
            if(y < 0 || y >= n_rows)
                continue;
            float l1 = 0;
            float l2 = 0;
            float l3 = 0;
//            barycoords_from_2d_trianglef()
        }
    }

}

//def rasterize_barycentrics_and_z_buffer_by_triangles(
//        triangle_vertex_indices, vertices,
//        barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer):
//    for tri_index, face in enumerate(triangle_vertex_indices):
//
//        tri_coords_3d = torch.stack((
//            vertices[face[0]],
//            vertices[face[1]],
//            vertices[face[2]],
//        ))
//        rasterize_triangle(barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer, tri_index, tri_coords_3d)

void rasterize_triangles(
    torch::Tensor triangle_vertex_indices,
    torch::Tensor vertices,
    torch::Tensor barycentrics_l1l2l3,
    torch::Tensor barycentrics_triangle_indices,
    torch::Tensor z_buffer)
{
    assert(vertices.dim() == 2);
    assert(triangle_vertex_indices.dim() == 2);
    const int64_t n_triangles = triangle_vertex_indices.size(0);

    const auto triangle_vertex_indices_acc = triangle_vertex_indices.accessor<int, 2>();
    const auto vertices_acc = vertices.accessor<float, 2>();
    for(int64_t tri_index = 0; tri_index <n_triangles; ++tri_index){
        const int i1 = triangle_vertex_indices_acc[tri_index][0];
        const int i2 = triangle_vertex_indices_acc[tri_index][1];
        const int i3 = triangle_vertex_indices_acc[tri_index][2];
        const Vector3f v1(vertices_acc[i1][0], vertices_acc[i1][1], vertices_acc[i1][2]);
        const Vector3f v2(vertices_acc[i2][0], vertices_acc[i2][1], vertices_acc[i2][2]);
        const Vector3f v3(vertices_acc[i3][0], vertices_acc[i3][1], vertices_acc[i3][2]);
        rasterize_triangle(tri_index, v1, v2, v3, barycentrics_l1l2l3, barycentrics_triangle_indices, z_buffer);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &lltm_forward, "LLTM forward");
//  m.def("backward", &lltm_backward, "LLTM backward");
//  m.def("foo", &foo, "Foo?");
  m.def("barycoords_from_2d_trianglef", &Barycentric::barycoords_from_2d_trianglef, "Foo?");
  m.def("rasterize_triangles", &rasterize_triangles, "");
}
