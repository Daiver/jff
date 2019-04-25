import numpy as np
#import autograd.numpy as np
#from autograd import grad, hessian
import copy
import cv2
import lm

def composeErrorFunction(initialPos, moves, distObservations):
    nMoves = len(moves)
    nObservations = len(distObservations)
    assert nObservations > 0
    assert nObservations == nMoves + 1
    nLandmarks = len(distObservations[0])

    posesVarsOffset = 0
    landmarksVarsOffset = nMoves * 2 + 2

    def costFunc(vars):
        res = []
        res.append((vars[0] - initialPos[0]) * 100.0)
        res.append((vars[1] - initialPos[1]) * 100.0)
        for moveInd in xrange(0, nMoves):
            curPosX  = vars[2*(moveInd + 1) + 0]
            curPosY  = vars[2*(moveInd + 1) + 1]
            prevPosX = vars[2*(moveInd + 0) + 0]
            prevPosY = vars[2*(moveInd + 0) + 1]
            deltaX = curPosX - prevPosX
            deltaY = curPosY - prevPosY
            res.append(deltaX - moves[moveInd][0])
            res.append(deltaY - moves[moveInd][1])
        for landmarkInd in xrange(0, nLandmarks):
            landmarkPosX = vars[landmarksVarsOffset + 2 * landmarkInd + 0]
            landmarkPosY = vars[landmarksVarsOffset + 2 * landmarkInd + 1]

            for obsInd in xrange(0, nObservations):
                curDist = distObservations[obsInd][landmarkInd]
                if curDist == None:
                    continue
                curPosX = vars[2*(obsInd) + 0]
                curPosY = vars[2*(obsInd) + 1]
                res.append(((curPosX - landmarkPosX)**2 + (curPosY - landmarkPosY)**2) - curDist)

        res = np.array(res)
        return res
        return res.dot(res)
    return costFunc

def simpleTest():
    initialPos = [0, 0]
    moves = [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            #[1, 0]
            ]
    distObservations = [
            [1, 4, 1],
            [2.0, 1, 2.0],
            [5.0, 0, 5.0],
            [8.0, 1, None],
            [None, 4, 5],
            #[25.0, 4, 9]
            ]
    vars = np.zeros(2 + 2*len(moves) + 2*len(distObservations[0]))
    func = composeErrorFunction(initialPos, moves, distObservations)
    jac = lm.numJac(func)
    res = lm.gaussNewton(func, jac, vars, 40, True)
    #res = minimize(func, vars, jac=grad(func))
    print res

    ansPoses = [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0]
    ansLandmarks = [-1.0, 0.0, 0.0, 2.0, 1.0, 0.0]
    resLandmarks = res[2 + 2 * len(moves):]
    print ansLandmarks
    print resLandmarks
    print 'Landmark diff:', np.linalg.norm(ansLandmarks - resLandmarks)

def drawRobot(img, pos, maxVisibleDist, visibleLandmarksPoses):
    for landmarkPos in visibleLandmarksPoses:
        cv2.line(img, 
                (int(pos[0]), int(pos[1])),
                (int(landmarkPos[0]), int(landmarkPos[1])),
                (0, 255, 0), 1)
    ppos1 = (int(pos[0]) - 4, int(pos[1] - 4))
    ppos2 = (int(pos[0]) + 4, int(pos[1] + 4))
    cv2.rectangle(img, ppos1, ppos2, (0, 0, 255), 5)
    cv2.circle(img, (int(pos[0]), int(pos[1])), maxVisibleDist, (127, 127, 127))

def drawLandmarks(img, landmarkPoses):
    for landmarkPos in landmarkPoses:
        cv2.circle(img, 
                (int(landmarkPos[0]), int(landmarkPos[1])),
                3,
                (80, 200, 0), 5)

def createCanvas(canvasSize):
    return np.zeros((canvasSize[0], canvasSize[1], 3))

def moveFromKey(key):
    stepLen = 10.0
    steps = {82: np.array([0, -stepLen]),
                84: np.array([0,  stepLen]),
                81: np.array([-stepLen, 0]),
                83: np.array([ stepLen, 0])}
    if key not in steps:
        return None
    return steps[key]

def euclDist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def getVisibleLandmarks(landmarkPoses, robotPos, maxVisibleDist):
    res = []
    for landmarkPos in landmarkPoses:
        if euclDist(landmarkPos, robotPos) <= maxVisibleDist:
            res.append(landmarkPos)
    return res
    
def distanceForVisibleLandmarks(landmarkPoses, robotPos, maxVisibleDist):
    res = []
    for landmarkPos in landmarkPoses:
        dist = euclDist(landmarkPos, robotPos)
        if dist <= maxVisibleDist:
            res.append(dist **2)
        else:
            res.append(None)
    return res
    
predictedPoses = None
predictedLandmarks = None
def predict(initialPos, moves, distObservationsRaw):
    global predictedPoses
    global predictedLandmarks
    moves = np.copy(np.array(moves)/100.0)
    distObservations = copy.deepcopy(distObservationsRaw)
    for i in xrange(len(distObservations)):
        for j in xrange(len(distObservationsRaw[0])):
            if distObservationsRaw[i][j] != None:
                distObservations[i][j] = distObservationsRaw[i][j]/10000.0
    #distObservations = np.copy(np.array(distObservations)/10000.0)
    initialPos = np.copy(initialPos/100.0)
    vars = np.random.rand(2 + 2*len(moves) + 2*len(distObservations[0]))
    #vars = np.zeros(2 + 2*len(moves) + 2*len(distObservations[0]), dtype=np.float64)
    if predictedPoses != None:
        vars[len(vars) - len(predictedLandmarks):] = predictedLandmarks
        vars[2: 2 + len(predictedPoses)] = predictedPoses
    vars[0] = initialPos[0]
    vars[1] = initialPos[1]
    func = composeErrorFunction(initialPos, moves, distObservations)
    jac = lm.numJac(func)
    res = lm.levmar(func, jac, vars, 1.0, 100)
    #res = lm.gaussNewton(func, jac, vars, 100, True)
    vars = res[0]
    landmarkPoses = vars[2 + 2 * len(moves):].reshape((-1, 2))*100
    robotPos      = vars[2 * len(moves): 2 * len(moves) + 2].reshape(2)*100
    predictedLandmarks = landmarkPoses.reshape(-1)/100
    predictedPoses = vars[2:2+len(moves)]
    print 'predicted landmark poses', landmarkPoses
    print 'predicted robot pose', robotPos

    return res[0], landmarkPoses, robotPos

if __name__ == '__main__':
    #simpleTest()
    #exit(0)
    canvasSize = [700, 1000]
    landmarkPoses = [
            [20, 45],
            [156, 30],
            [100, 60],
            [120, 140], 
            [100, 200], 
            [150, 199], 
            [270, 266],
            [90, 250],
            [110, 240],
            [321, 123]
            ]
    #landmarkPoses = [[-1.0, 0.0], [0, 2.0], [1, 0]]
    #initialPos = np.array([0.0, 0.0])
    initialPos = np.array([250.0, 250.0])
    maxVisibleDist = 120

    key = None
    robotPos = np.copy(initialPos)
    moves = []
    observations = []
    obs = distanceForVisibleLandmarks(landmarkPoses, robotPos, maxVisibleDist)
    observations.append(obs)
    iter = 0
    while key != 27:
        canvas = createCanvas(canvasSize)
        visibleLandmarksPoses = getVisibleLandmarks(landmarkPoses, robotPos, maxVisibleDist)
        drawRobot(canvas, robotPos, maxVisibleDist, visibleLandmarksPoses)
        drawLandmarks(canvas, landmarkPoses)
        cv2.imshow('', canvas)
        key = cv2.waitKey() % 0x100
        #print key
        move = moveFromKey(key)
        if move == None:
            key = cv2.waitKey() % 0x100
            continue

        moves.append(move)
        robotPos += move
        obs = distanceForVisibleLandmarks(landmarkPoses, robotPos, maxVisibleDist)
        observations.append(obs)
        print move, robotPos, obs
        if iter % 5 == 0 and iter > 0:
            _, predLanmarks, predPos = predict(initialPos, moves, observations)
            canvas2 = createCanvas(canvasSize)
            drawRobot(canvas2, predPos, maxVisibleDist, [])
            drawLandmarks(canvas2, predLanmarks)
            cv2.imshow('1', canvas2)
        iter += 1
