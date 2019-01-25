import numpy as np
import cv2
import sklearn.decomposition
import os

dirName = '/home/daiver/Downloads/at_t_faces/'

def loadFaces(dirName):
    subdirs = os.listdir(dirName)
    paths = []
    for sd in subdirs:
        paths.extend([os.path.join(dirName, sd, x) for x in os.listdir(os.path.join(dirName, sd))])
    return [cv2.imread(x, 0) for x in paths]

if __name__ == '__main__':
    imgs = loadFaces(dirName)
    #imgs = imgs[:400]
    print len(imgs)
    X = np.zeros((len(imgs), imgs[0].shape[0] * imgs[0].shape[1]), dtype=np.float32)
    for i, img in enumerate(imgs):
        X[i] = img.reshape((-1))
    pca = sklearn.decomposition.PCA(n_components=250)
    pca.fit(X)

    for i, img in enumerate(imgs):
        sample = img.reshape((-1))
        trans = pca.transform(sample)
        res = pca.inverse_transform(trans)
        #print res
        cv2.imshow('orig', img)
        cv2.imshow('', res.reshape(img.shape).astype(np.uint8))
        print np.linalg.norm(img.reshape(-1) - res)
        cv2.waitKey()
