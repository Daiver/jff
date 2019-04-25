import rasterizer_cpp


def main():
    print(rasterizer_cpp)
    print(rasterizer_cpp.barycoords_from_2d_trianglef(
        0, 0,
        1, 0,
        0, 1,
        0.5, 0.5))
    print(rasterizer_cpp.barycoords_from_2d_trianglef(
        0, 0,
        1, 0,
        0, 1,
        0.0, 0.0))


if __name__ == '__main__':
    main()
