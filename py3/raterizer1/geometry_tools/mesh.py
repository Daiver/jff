class Mesh:
    def __init__(self,
                 vertices,
                 polygon_vertex_indices,
                 texture_vertices=None,
                 texture_polygon_vertex_indices=None,
                 normals=None,
                 triangle_vertex_indices=None,
                 triangle_texture_vertex_indices=None):
        self.vertices = vertices
        self.polygon_vertex_indices = polygon_vertex_indices
        self.texture_vertices = texture_vertices
        self.texture_polygon_vertex_indices = texture_polygon_vertex_indices
        self.normals = normals
        self.triangle_vertex_indices = triangle_vertex_indices
        self.triangle_texture_vertex_indices = triangle_texture_vertex_indices

    def __eq__(self, other):
        assert isinstance(other, Mesh)
        
