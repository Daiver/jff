import numpy as np

def exportMatTxt(fname, mat):
    with open(fname, 'w') as f:
        f.write("%d %d\n" % (mat.shape[0], mat.shape[1]))
        for row in mat:
            for val in row:
                f.write("%f " % val)
            f.write("\n")
