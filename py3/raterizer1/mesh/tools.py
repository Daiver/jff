def triangulate_polygons(polygon_vertex_indices):
    res = []
    for polygon in polygon_vertex_indices:
        n_vertices = len(polygon)
        assert n_vertices >= 3
        for i in range(1, n_vertices - 1)
            res.append([polygon[i - 1], polygon[i], polygon[i + 1]])

    return res
