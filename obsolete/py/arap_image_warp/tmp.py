import numpy as np
from arap_image_warp1 import *

if __name__ == '__main__':
    #    pos << QVector2D(0, 0) << QVector2D(1, 1) << QVector2D(2, 0) << QVector2D(1, -1) << QVector2D(1, 0);
    '''values << 1 << 4 << 3;      //0
    values << 0 << 4 << 2;      //1
    values << 1 << 3 << 4;      //2
    values << 4 << 2 << 0;      //3
    values << 0 << 1 << 2 << 3; //4'''
    pos = np.array([
            [0,0],
            [1,1],
            [2,0],
            [1,-1],
            [1,0]
            ])
    adj = [
            [1, 4, 3],
            [0, 4, 2],
            [1, 3, 4],
            [4, 2, 0],
            [0, 1, 2, 3]
            ]
    freezeIndices = [4, 0]
    freezePos = np.array([[5, 6], [2, 3]])
    rotPos, gs, cells = computeARAPImageWarpStage1(
                        pos, adj, freezeIndices, freezePos, 1000)
    newPos = computeARAPImageWarpStage2(pos, rotPos, adj, gs, cells, 
                                        freezeIndices, freezePos, 1000)
    print newPos

def main1():
    g = np.array([
           0,  10,
           10,   0,
           20,  30,
           30, -20,
           40,  50,
           50, -40,
           5,   6,
           8,  -7
        ], dtype=np.float32).reshape((8, 2))
    g2 = np.dot(g.transpose(), g)
    g3 = np.linalg.pinv(g2) 
    g4 = np.dot(g3, g.transpose())
    print g4
