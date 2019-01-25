import read
import cv2
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def descFromLandmarks(desc_obj, img, bbox, landmarks):
    key_points = [cv2.KeyPoint(p[0], p[1], 32) for p in landmarks]
    return desc_obj.compute(img, key_points)[1]

def normalizeLandmarks(bbox, landmarks):
    landmarks = landmarks - bbox[:2]
    w, h = bbox[2:]
    landmarks[:, 0] /= w
    landmarks[:, 1] /= h
    return landmarks

def unnormalizeLandmarks(bbox, landmarks):
    w, h = bbox[2:]
    landmarks = np.copy(landmarks)
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    landmarks += bbox[:2]
    return landmarks

def generateTrainingData(names, images, bboxes, annotations, n_samples_per_image, desc_obj):
    res_names        = []
    res_target_shape = []
    res_descs        = []
    res_deltas       = []

    for name in names:
        bbox       = bboxes[name]
        landmarks  = annotations[name]
        normalized = normalizeLandmarks(bbox, landmarks)
        img        = images[name]
        for i in xrange(n_samples_per_image):
            res_names.append(name)
            res_target_shape.append(landmarks)
            name1       = random.choice(names)
            bbox1       = bboxes[name1]
            landmarks1  = annotations[name1]
            normalized1 = normalizeLandmarks(bbox1, landmarks1)
            transfered1 = unnormalizeLandmarks(bbox, normalized1)
            descs       = descFromLandmarks(orb, img, bbox, transfered1).reshape(-1)
            res_deltas.append(normalized - normalized1)
            res_deltas.append(descs)

    return res_names, res_target_shape, res_deltas, res_descs

if __name__ == '__main__':
    path2photos      = '/home/daiver/coding/data/helen/all_photos/'
    path2annotations = '/home/daiver/coding/data/helen/annotation/'

    n_photos2read = 20
    images = read.readPhotos(path2photos, n_photos2read)
    annotations = read.readAnnotations(path2annotations)
    bbox_offset = 60
    bboxes = {name: read.bboxFromLandmarks(annotations[name], bbox_offset) for name in images}

    orb = cv2.ORB()
    descsForAll = {
                name: descFromLandmarks(orb, images[name], bboxes[name], annotations[name]).reshape(-1)
                for name in images.keys()
            }

    tmp = generateTrainingData(images.keys(), images, bboxes, annotations, 10, orb)

    # name = images.keys()[0]
    # print annotations[name]
    # print normalizeLandmarks(bboxes[name], annotations[name])
    # print unnormalizeLandmarks(bboxes[name], normalizeLandmarks(bboxes[name], annotations[name]))

    

