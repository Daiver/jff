import numpy as np
import cv2

def drawPoints(positions, adj, width, height):
    img = np.ones((width, height, 3), dtype=np.uint8)
    img *= 255
    for i, a in enumerate(adj):
        p1 = positions[i]
        for j in a:
            p2 = positions[j]
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0))

    for p in positions:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 0, 0))

    return img

def printVec(v):
    return ', '.join('[' + ('%.5f ' % x[0]) + ('%.5f' % x[1]) + ']' for x in v)

def readMSMMFromFile(fname):
    with open(fname, 'rt') as f:
        nVerts = int(f.readline())
        positions = []
        for i in xrange(nVerts):
            s = map(float, f.readline().split())
            positions.append(np.array(s))
        adj = []
        for i in xrange(nVerts):
            s = map(int,f.readline().split())
            adj.append(s)

    print adj[0]
    print adj[1]
    return MSMMesh(adj, positions)

class MSMMesh:
    def __init__(self, adj, positions):
        self.adj = adj
        self.positions = positions

    def moveVertex(self, index, pos):
        dt = 1.01
        k  = 0.1
        mass = 1
        newPositions = self.positions[:]
        newPositions[index] = pos
        velocities = [np.zeros(2) for x in newPositions]
        positions = self.positions

        cv2.imshow('', drawPoints(positions, self.adj, 800, 800))
        cv2.waitKey(2000)

        for iter in xrange(5000):
            #newPositions[index] = newPositions[index] + 0.0001 * (pos - newPositions[index])
            for i, x in enumerate(self.adj):
                if i == index: continue
                if i in [0, 1, 2, 3]:
                    continue
                imp = np.zeros(2)
                s   = 0
                p = newPositions[i]

                for a in (x):
                    f = -k*(np.linalg.norm(positions[a] - positions[i]) - np.linalg.norm(newPositions[a] - newPositions[i]))
                    if np.linalg.norm(newPositions[a] - p) == 0:
                        direction = 0
                    else:
                       direction = (newPositions[a] - p)/np.linalg.norm(newPositions[a] - p)
                    imp += direction*f
                    s += f
                ''''if s == 0:
                    imp = np.zeros(2)
                else:
                    imp /= s'''
                #print i, imp
                velocities[i] = imp/mass * dt
                #velocities[i] *= 0.999
            
            for i, p in enumerate(newPositions):
                if i == index: continue
                newPositions[i] = p + velocities[i]*dt
           # print 'V'
            #print printVec(velocities)
            #print 'Pos'
            #print printVec(newPositions)
            if  iter % 30 == 0:
                en = sum(np.linalg.norm(x) for x in velocities)
                print 'iter', iter, en
                cv2.imshow('', drawPoints(newPositions, self.adj, 800, 800))
                cv2.waitKey(10)
        cv2.imwrite('res.png', drawPoints(newPositions, self.adj, 800, 800))

if __name__ == '__main__':
    mesh = readMSMMFromFile('./simplemesh.txt')
#    exit()
    #adj = [
            #[1, 3, 4],
            #[0, 4, 2, 3, 5],
            #[1, 5, 4],
            #[0, 4, 6, 7, 1],
            #[1, 3, 5, 7, 0, 2, 6, 8],
            #[2, 4, 8, 1, 7],
            #[3, 7, 4],
            #[6, 4, 8, 3, 5],
            #[7, 5, 4]
            #]
    #positions = [
            #np.array([0, 0]), #0
            #np.array([1, 0]), #1
            #np.array([2, 0]), #2
            #np.array([0, 1]), #3
            #np.array([1, 1]), #4
            #np.array([2, 1]), #5
            #np.array([0, 2]), #6
            #np.array([1, 2]), #7
            #np.array([2, 2])  #8
            #]
    #mesh = MSMMesh(adj, positions)
    mesh.moveVertex(45, np.array([430.0, 530]))
