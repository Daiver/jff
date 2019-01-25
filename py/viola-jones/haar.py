import cv2
import numpy as np
import common

def haarHorizLine(x, y, width, height):
    return ([
                [
                    (int(y), int(x)),
                    (int(y), int(x + width)),
                    (int(y + 0.5 * height), int(x)),
                    (int(y + 0.5 * height), int(x + width))
                ]
            ],
            [
                [
                    (int(y + 0.5 * height), int(x)),
                    (int(y + 0.5 * height), int(x + width)),
                    (int(y + 1.0 * height), int(x)),
                    (int(y + 1.0 * height), int(x + width))
                ]
            ])

def haarVertLine(x, y, width, height):
    return ([
                [
                    (int(y), int(x)),
                    (int(y), int(x + 0.5 * width)),
                    (int(y + 1.0 * height), int(x)),
                    (int(y + 1.0 * height), int(x + 0.5 * width))
                ]
            ],
            [
                [
                    (int(y), int(x + 0.5 * width)),
                    (int(y), int(x + 1.0 * width)),
                    (int(y + 1.0 * height), int(x + 0.5 * width)),
                    (int(y + 1.0 * height), int(x + 1.0 * width))
                ]
            ])

def haarChess(x, y, width, height):
    return ([
                [
                    (int(y), int(x + 0.5 * width)),
                    (int(y), int(x + 1.00 * width)),
                    (int(y + 0.5 * height), int(x + 0.5 * width)),
                    (int(y + 0.5 * height), int(x + 1.0 * width))
                ],
                [
                    (int(y + 0.5 * height), int(x)),
                    (int(y + 0.5 * height), int(x + 0.5 * width)),
                    (int(y + 1.0 * height), int(x)),
                    (int(y + 1.0 * height), int(x + 0.5 * width))
                ]

            ],
            [
                [
                    (int(y), int(x)),
                    (int(y), int(x + 0.5 * width)),
                    (int(y + 0.5 * height), int(x)),
                    (int(y + 0.5 * height), int(x + 0.5 * width))
                ],
                [
                    (int(y + 0.5 * height), int(x + 0.5 * width)),
                    (int(y + 0.5 * height), int(x + 1.00 * width)),
                    (int(y + 1.00 * height), int(x + 0.5 * width)),
                    (int(y + 1.00 * height), int(x + 1.00 * width))
                ]
            ])


def haarBWB(x, y, width, height):
    return (
            [
                [
                    (int(y), int(x)),
                    (int(y), int(x + 1.0/3 * width)),
                    (int(y + 1.0 * height), int(x)),
                    (int(y + 1.0 * height), int(x + 1.0/3 * width))
                ],
                [
                    (int(y), int(x + 2.0/3 * width)),
                    (int(y), int(x + width)),
                    (int(y + 1.0 * height), int(x + 2.0/3 * width)),
                    (int(y + 1.0 * height), int(x + width))
                ]

            ],
            [
                [
                    (int(y), int(x + 1.0/3 * width)),
                    (int(y), int(x + 2.0/3 * width)),
                    (int(y + 1.0 * height), int(x + 1.0/3 * width)),
                    (int(y + 1.0 * height), int(x + 2.0/3 * width))
                ]
            ]
            
            )



def computeHaarFeature(integralImg, negRectsScaled, posRectsScaled):
    sm = 0.0
    for negRect in negRectsScaled:
        sm -= ( integralImg[negRect[0][0], negRect[0][1]]   
              + integralImg[negRect[3][0], negRect[3][1]]
              - integralImg[negRect[1][0], negRect[1][1]]
              - integralImg[negRect[2][0], negRect[2][1]]
                )

    for posRect in posRectsScaled:
        sm += ( integralImg[posRect[0][0], posRect[0][1]]   
              + integralImg[posRect[3][0], posRect[3][1]]
              - integralImg[posRect[1][0], posRect[1][1]]
              - integralImg[posRect[2][0], posRect[2][1]]
                )
    return sm

def drawHaarFeature(img, negRectsScaled, posRectsScaled):
    for negRect in negRectsScaled:
        #cv2.rectangle(img, negRect[0], negRect[3], 0)
        cv2.rectangle(img, (negRect[0][1], negRect[0][0]), (negRect[3][1], negRect[3][0]), 0)
        for p in negRect:
            cv2.circle(img, (p[1], p[0]), 2, 0)
    for posRect in posRectsScaled:
        cv2.rectangle(img, (posRect[0][1], posRect[0][0]), (posRect[3][1], posRect[3][0]), 255)
        for p in posRect:
            cv2.circle(img, (p[1], p[0]), 3, 255)

haarFeaturesFuncs = {
        'hor'   : haarHorizLine,
        'vert'  : haarVertLine,
        'bwb'   : haarBWB,
        'chess' : haarChess
        }

if __name__ == '__main__':
    img = cv2.imread('./testImgs/s10_10.pgm.png', 0)
    negR, posR = haarChess(10, 10, 50, 90)
    #negR, posR = haarBWB(10, 10, 80, 90)
    #negR, posR = haarHorizLine(10, 10, 50, 90)
    #negR, posR = haarVertLine(10, 10, 50, 90)
    drawHaarFeature(img, negR, posR)
    cv2.imshow('', img)
    cv2.waitKey()
