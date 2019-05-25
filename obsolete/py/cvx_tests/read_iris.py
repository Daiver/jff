import numpy as np

def readBinaryClasses(fname):
    data = []
    labels = []
    with open(fname) as f:
        for s in f:
            if s == '':
                continue
            tokens = s.split(',')

            name = tokens[-1]
            sample = map(float, tokens[:-1])
            if len(sample) == 0:
                continue
            data.append(sample)
            if name == 'Iris-setosa\n':
                labels.append(-1)
            else:
                labels.append(1)
    return np.array(data), np.array(labels, dtype=np.int8)
