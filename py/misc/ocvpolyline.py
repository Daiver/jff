import cv2
import numpy as np

def draw_line(line):
    width, height = 640, 480  # picture's size
    img = np.zeros((height, width, 3), np.uint8) + 255 # make the background white
    line_width = 5
    color = (0,0,0) # change color or make a color generator for your self
    pts = np.array(line).reshape((1, -1, 2))
    print pts
    #cv2.polylines(img, pts, False, color, thickness=line_width)
    cv2.polylines(img, [pts], False, color, thickness=line_width, lineType=cv2.LINE_AA)
    cv2.imshow("Art", img)
    cv2.waitKey(0)    # miliseconds, 0 means wait forever


point_lists = [(200, 200), (200, 100), (400, 200)]
draw_line(point_lists)
