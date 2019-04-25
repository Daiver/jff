import cv2

def cutImage(img, (x, y, w, h)):
    return img[x : x + w, y : y + h]

if __name__ == '__main__':
    pass
