
def cutPatch(img, rect):
    x1, y1 = rect[0:2]
    x2, y2 = x1 + rect[2], y1 + rect[3]
    return img[y1:y2, x1:x2]

