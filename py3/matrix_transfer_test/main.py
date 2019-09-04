import os
import json
import numpy as np
from scipy.spatial.transform import Rotation


def from_dcm_mat_to_euler(transformation_matrix):
    assert transformation_matrix.shape == (4, 4)
    rot_dcm = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]
    rot_euler = Rotation.from_dcm(rot_dcm).as_euler("xyz", degrees=True)
    return rot_euler, translation


def process_file(input_name, output_name):
    with open(input_name) as f:
        data = json.load(f)
    mat = np.array(data)
    rot, trans = from_dcm_mat_to_euler(mat)
    with open(output_name, "w") as f:
        res = {
            "rotationEuler": {
                "x": rot[0],
                "y": rot[1],
                "z": rot[2],
            },
            "scale": 1.0,
            "translation": {
                "x": trans[0],
                "y": trans[1],
                "z": trans[2],
            }
        }
        json.dump(res, f)


def process_dir(input_dir_name, output_dir_name):
    names = os.listdir(input_dir_name)
    names.sort()
    for name in names:
        if not name.endswith(".txt"):
            continue
        input_path = os.path.join(input_dir_name, name)
        output_path = os.path.join(output_dir_name, name)
        print(input_path, "->", output_path)
        process_file(input_path, output_path)


# process_file("/home/daiver/FromStabilizedScanSpaceToScan0000.txt", "/home/daiver/tmp.txt")
root = "/home/daiver/transforms/"
root_res = "/home/daiver/transforms_fixed/"
takes_names = os.listdir(root)
takes_names.sort()
for name in takes_names:
    input_dir = os.path.join(root, name, "Transforms")
    if not os.path.exists(input_dir):
        continue
    output_dir = os.path.join(root_res, name, "Transforms")
    os.makedirs(output_dir, exist_ok=True)
    process_dir(input_dir, output_dir)
