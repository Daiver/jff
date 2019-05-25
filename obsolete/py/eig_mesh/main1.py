import numpy as np

if __name__ == '__main__':
    n_vertices = 5
    mat = np.zeros((n_vertices, n_vertices))
    
    mat[0, 0] = -3
    mat[0, 1] =  1
    mat[0, 3] =  1
    mat[0, 4] =  1

    mat[1, 0] =  1
    mat[1, 1] = -3
    mat[1, 2] =  1
    mat[1, 3] =  1

    mat[2, 1] =  1
    mat[2, 2] = -2
    mat[2, 4] =  1

    mat[3, 0] =  1
    mat[3, 1] =  1
    mat[3, 3] = -2
    #mat[3, 4] =  1

    mat[4, 0] =  1
    mat[4, 2] =  1
    #mat[4, 3] =  1
    mat[4, 4] = -2

    w, v = np.linalg.eig(-mat)
    print w
    #print v

    for i in xrange(5):
        print w[i], '/', (v[:, i]) * 5

    
