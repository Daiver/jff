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


def replace_obj_vertices(src_obj_path, vertices, dst_path):
    v_ind = 0
    with open(src_obj_path) as src_file:
        with open(dst_path, 'w') as dst_file:
            for string in src_file:
                if string[0:2] != 'v ':
                    dst_file.write(string)
                else:
                    dst_file.write(f"v {vertices[v_ind, 0]} {vertices[v_ind, 1]} {vertices[v_ind, 2]}\n")
                    v_ind += 1
    assert v_ind == len(vertices)
