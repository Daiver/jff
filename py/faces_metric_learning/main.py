import os
import sys
import numpy as np
import scipy
import cv2
import sklearn.decomposition

#more oneliners for god of oneliners!
def readDataset(dir_name):
    return {
            subdir: [
                cv2.imread(os.path.join(dir_name, subdir, fname), 0)
                for fname in os.listdir(os.path.join(dir_name, subdir))
            ]
            for subdir in os.listdir(dir_name) if subdir[0] == 's'
           }

def flatten(l): return [item for sublist in l for item in sublist]

def mkDescMat(diff_mat, cov_mat):
    cov_sqrt_mat  = scipy.linalg.sqrtm(cov_mat)
    cov_sqrt_inv  = np.linalg.inv(cov_sqrt_mat)
    #i don't know how sqrtm works, so check can be usefull
    assert np.abs(sum(sum(cov_sqrt_inv.T - cov_sqrt_inv))) < 0.00001 
    weighted_diff = cov_sqrt_inv.dot(diff_mat).dot(cov_sqrt_inv)
    #maybe we cannot use eigh?
    assert np.abs(sum(sum(weighted_diff.T - weighted_diff))) < 0.00001
    evals, evecs  = np.linalg.eigh(weighted_diff)
    neginds       = [i for i, x in enumerate(evals) if x < 0]
    assert len(neginds) > 0
    print 'n neg eigs', len(neginds)
    negvecs       = evecs[:, neginds]
    res_transform = negvecs.T.dot(cov_sqrt_mat)

    return res_transform

def mkCovMatOfDiffs(pairs, descs):
    n_feats   = descs.shape[1]
    res = np.zeros((n_feats, n_feats))

    for i1, i2 in pairs:
        d1 = descs[i1]
        d2 = descs[i2]
        e  = d1 - d2
        res += e.dot(e.T)

    return res / len(pairs)

#TODO: Dirty/Slow! May produce bad pairs
def genPairs(n_persons, n_photos_per_person):
    pos_pairs = []
    #provides too much pos pairs but i have no choice 
    for i in xrange(n_persons):
        for j1 in xrange(n_photos_per_person):
            for j2 in xrange(n_photos_per_person):
                if j1 == j2:
                    continue
                pos_pairs.append((i*n_photos_per_person + j1, i*n_photos_per_person + j2))
    neg_pairs = []
    for i1 in xrange(n_persons * n_photos_per_person):
        for i2 in xrange(n_persons * n_photos_per_person):
            if abs(i1 - i2) < n_photos_per_person:
                continue
            neg_pairs.append((i1, i2))
    print len(neg_pairs), len(pos_pairs)
    return pos_pairs, neg_pairs

def train(all_data_transformed, n_persons, n_photos_per_person):
    pos_pairs, neg_pairs = genPairs(n_persons, n_photos_per_person)
    neg_weight = 0.5
    pos_weight = 1.0 - neg_weight

    pos_cov_mat  = mkCovMatOfDiffs(pos_pairs, all_data_transformed)
    neg_cov_mat  = mkCovMatOfDiffs(neg_pairs, all_data_transformed)
    diffCovMat = pos_weight * pos_cov_mat - neg_weight * neg_cov_mat

    cov_mat = np.cov(all_data_transformed.T)
    desc_mat = mkDescMat(diffCovMat, cov_mat)
    return desc_mat

if __name__ == '__main__':
    path2data = '/home/daiver/Downloads/atnt_faces/'
    raw_images = readDataset(path2data)
    print raw_images.keys()
    all_flatten_data = np.array([x.reshape(-1) for x in flatten(raw_images.values())])
    
    pca = sklearn.decomposition.PCA(300)
    pca.fit(all_flatten_data)

    all_data_transformed = pca.transform(all_flatten_data)
    print len(all_data_transformed)

    split_thres_person = 33
    n_photos_per_person = 10
    split_thres_flat = n_photos_per_person * split_thres_person
    train_set = all_data_transformed[:split_thres_flat]
    test_set  = all_data_transformed[split_thres_flat:]
    desc_mat = train(train_set, split_thres_person, n_photos_per_person)
    #descs = all_data_transformed.dot(desc_mat.T)
    descs = test_set.dot(desc_mat.T)
    print desc_mat.shape, descs.shape
    
    test_images = flatten([raw_images.values()[i] for i in xrange(split_thres_person, len(raw_images.keys()))])
    for i, img in enumerate(test_images):
        nearest_inds = np.argsort(np.linalg.norm(descs - descs[i], axis=1))[1:4]

        img = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 2, img.shape[0] - 2), (255, 0, 0), 3)
        cv2.imshow('query', img)
        for j, ind in enumerate(nearest_inds):
            cv2.imshow('nearest %d' % j, test_images[ind])
        cv2.waitKey()

    '''
    d0 = descs[0]
    d1 = descs[1]
    d2 = descs[2]
    d20 = descs[10]
    d22 = descs[12]

    print 'd0/d1', np.linalg.norm(d0 - d1)
    print 'd0/d2', np.linalg.norm(d0 - d2)
    print 'd1/d2', np.linalg.norm(d1 - d2)
    print 'd0/d20', np.linalg.norm(d0 - d20)
    print 'd1/d20', np.linalg.norm(d1 - d20)
    print 'd2/d20', np.linalg.norm(d2 - d20)
    print 'd0/d22', np.linalg.norm(d2 - d22)
    print 'd22/d20', np.linalg.norm(d22 - d20)
    '''

