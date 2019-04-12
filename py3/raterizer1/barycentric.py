
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
