from .mesh import Mesh


# Currently normals are not supported
# TODO: not effective, use file streams or something like this
def from_obj_file(file_name):
    with open(file_name) as f:
        string = f.read()
        return from_obj_string(string)


# TODO: just ugly, split it to several functions, add assertions, etc
def from_obj_string(string):
    lines = string.split("\n")
    vertices = []
    texture_vertices = []
    polygon_vertex_indices = []
    texture_polygon_vertex_indices = []

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        tokens = line.split(" ")

        if tokens[0] == "v":
            assert len(tokens) == 4
            vertices.append([float(x) for x in tokens])
        elif tokens[0] == "vt":
            assert len(tokens) == 3
            texture_vertices.append([float(x) for x in tokens])
        elif tokens[0] == "f":
            assert len(tokens) >= 3
            vertex_face = []
            texture_face = []
            for token in tokens:
                is_token_contains_double_slash = "//" in token
                face_tokens = token.split("/")
                vertex_face.append(int(face_tokens[0]))
                if len(face_tokens) > 1 and not is_token_contains_double_slash:
                    texture_face.append(int(face_tokens[1]))
            polygon_vertex_indices.append(vertex_face)
            if len(texture_face) > 0:
                texture_polygon_vertex_indices.append(texture_face)

    normals = None
    triangle_vertex_indices = None
    triangle_texture_vertex_indices = None

    return Mesh(
        vertices=vertices,
        polygon_vertex_indices=polygon_vertex_indices,
        texture_vertices=texture_vertices,
        texture_polygon_vertex_indices=texture_polygon_vertex_indices,
        normals=normals,
        triangle_vertex_indices=triangle_vertex_indices,
        triangle_texture_vertex_indices=triangle_texture_vertex_indices)
