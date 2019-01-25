class Vertex:
    def __init__(self, vertexes, faces, hedges, point, _edge):
        self.vertexes = vertexes
        self.faces = faces
        self.hedges = hedges
        self.point = point
        self._edge = _edge
    
    def edge(self): return self.hedges[self._edge]

    def __repr__(self):
        return 'Vertex %d(%s, %d)' % (self.index, str(self.point), str(self._edge))


