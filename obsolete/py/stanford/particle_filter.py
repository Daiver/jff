import random

def particleFilter(test, mutate, init, num_of_iterations):
    p = init
    N = len(p)
    for t in range(num_of_iterations):
        p = mutate(p)
        w = [test(x) for x in p]
        p2 = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            p2.append(p[index])
        p = p2
    return p

