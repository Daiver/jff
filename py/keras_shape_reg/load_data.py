import os
import cv2
import numpy as np

from input_shape import input_shape

def loadDataLabelsByTargetFile(path2Images, preprocessFunc = None):
    imgs = []
    targets = []
    names = []
    with open(os.path.join(path2Images, 'targets.txt')) as f:
        for s in f:
            if len(s) < 2:
                continue
            tokens = s.split(" ")
            name = (tokens[0])
            names.append(name)
            img = cv2.imread(os.path.join(path2Images, name))
            if preprocessFunc == None:
                imgs.append(img)
            else:
                imgs.append(preprocessFunc(img))
            coords = [float(x) for x in tokens[1:] if x != '\n']
            targets.append(coords)
    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    imgs /= 255.0
    targets = np.array(targets, dtype=np.float32)

    return imgs, targets, names

def loadDataMultiViewByTargetFile(path2Images, preprocessFunc = None):
    imgs = []
    targets = []
    names = []
    with open(os.path.join(path2Images, 'targets.txt')) as f:
        lines = f.readlines()
    assert(len(lines) % 2 == 0)
    for i in xrange(0, len(lines), 2):        
        tokens = lines[i].split(" ")
        name = (tokens[0])
        names.append(name)
        imgNames = [x.strip() for x in tokens[1:]]
        localImgs = []
        for imgName in imgNames:
            img = cv2.imread(os.path.join(path2Images, imgName))
            if preprocessFunc == None:
                localImgs.append(img)
            else:
                localImgs.append(preprocessFunc(img))

        img = np.concatenate(localImgs, -1)
        imgs.append(img)
        coordsStr = lines[i + 1].strip()
        coords = [float(x) for x in coordsStr.split(" ") if x != '\n']
        targets.append(coords)
        #if len(targets) > 3048:
        #    print 'Temporary break!'
        #    break
    imgShape = imgs[0].shape
    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs.reshape(-1, imgShape[0], imgShape[1], imgShape[2])
    imgs /= 255.0
    targets = np.array(targets, dtype=np.float32)

    return imgs, targets, names

if __name__ == '__main__':
    import cv2
    img_dir = '/home/daiver/R3DS/Data/Render2ShapeRegression/blendshape_data/'
    train_dir = img_dir + "train/"
    imgs_train, y_train, names_train = loadDataMultiViewByTargetFile(train_dir, None)
    print imgs_train.shape
    for img in imgs_train:
        img1 = img[:,:, :3]
        img2 = img[:,:, 3:]
        cv2.imshow('1', img1)
        cv2.imshow('2', img2)
        cv2.waitKey()



