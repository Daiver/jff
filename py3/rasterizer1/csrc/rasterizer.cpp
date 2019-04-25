#include <torch/extension.h>

#include <iostream>
#include <vector>

#include "barycentric.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &lltm_forward, "LLTM forward");
//  m.def("backward", &lltm_backward, "LLTM backward");
//  m.def("foo", &foo, "Foo?");
  m.def("barycoords_from_2d_trianglef", &Barycentric::barycoords_from_2d_trianglef, "Foo?");
}
