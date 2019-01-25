import numpy as np


def read_obj_vertices(path):
    res = []
    with open(path) as f:
        for string in f:
            if string[0:2] != 'v ':
                continue
            positions = list(map(float, string.split(" ")[1:]))
            res.append(positions)

    return np.array(res, dtype=np.float32)
