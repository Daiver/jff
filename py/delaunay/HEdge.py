class HEdge:
    def __init__(self, vertexes, faces, hedges, _head, _opposite, _next, _leftFace=None):
        self.vertexes = vertexes
        self.faces = faces
        self.hedges = hedges
        self._head = _head
        self._opposite = _opposite
        self._next = _next
        self._leftFace = _leftFace

    def head(self):     return self.vertexes[self._head]
    def opposite(self): 
        if self._opposite == None: return None
        return self.hedges[self._opposite]
    def next(self):     return self.hedges[self._next]
    def leftFace(self): return self.faces[self._leftFace]

    def vertexesPoints(self):
        return self.head().point, self.opposite().head().point

    def __repr__(self): 
        return 'HEdge %d(head %d opposite %d next %d leftFace %s)' % (
                self.index, self._head, self._opposite, self._next, str(self._leftFace))

