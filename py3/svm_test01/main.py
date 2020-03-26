import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC, NuSVC, LinearSVC


def main1():
    xs = np.array([
        [0, 0],
        [-1, 0],
        [1, 0]
    ], dtype=np.float32)
    ys = np.array([
        1, 0, 0
    ], dtype=np.float32)
    # reg = 1e-10
    reg = 3
    # reg = 5
    # reg = 500
    clf = SVC(C=reg, kernel="rbf", gamma="auto", degree=2, probability=True)
    # clf = SVC(C=reg, kernel="poly", gamma="scale", probability=True)
    clf.fit(xs, ys)
    print(clf._get_coef())
    print(clf.predict(xs))
    print(clf.predict_proba(xs))

    pts = [(x, y) for x in np.linspace(-2, 2, 51) for y in np.linspace(-2, 2, 51)]
    pts = np.array(pts)
    predicted = clf.predict(pts)
    pts_n = pts[predicted == 0]
    pts_p = pts[predicted == 1]

    tr_n = xs[ys == 0]
    tr_p = xs[ys == 1]

    plt.plot(pts_n[:, 0], pts_n[:, 1], ".r")
    plt.plot(pts_p[:, 0], pts_p[:, 1], ".g")
    plt.plot(tr_n[:, 0], tr_n[:, 1], "Dr")
    plt.plot(tr_p[:, 0], tr_p[:, 1], "Dg")
    plt.show()


def main():
    import cvxpy

    xs = np.array([
        [0, 0],
        [-1, 1],
        [1, 1]
    ], dtype=np.float32)
    ys = np.array([
        1, -1, -1
    ], dtype=np.float32)

    # Define and solve the CVXPY problem.
    slope = cvxpy.Variable(2)
    bias = cvxpy.Variable()

    predict = lambda xs: xs @ slope + bias

    # loss = cvxpy.sum(cvxpy.pos(1 - cvxpy.multiply(ys, predict(xs))))
    loss = cvxpy.sum_squares(cvxpy.pos(1 - cvxpy.multiply(ys, predict(xs))))
    reg = cvxpy.multiply(0.999999, cvxpy.norm(slope, 2))
    # reg = cvxpy.multiply(2, cvxpy.norm(slope, 2))
    # cost = cvxpy.sum(cvxpy.pos(bias))
    prob = cvxpy.Problem(cvxpy.Minimize(loss + reg))
    prob.solve(verbose=True)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(slope.value, bias.value)

    pts = [(x, y) for x in np.linspace(-2, 2, 51) for y in np.linspace(-2, 2, 51)]
    pts = np.array(pts)
    predicted = predict(pts).value
    print(predicted)
    pts_n = pts[predicted < 0]
    pts_p = pts[predicted >= 0]

    tr_n = xs[ys < 0]
    tr_p = xs[ys >= 0]

    plt.plot(pts_n[:, 0], pts_n[:, 1], ".r")
    plt.plot(pts_p[:, 0], pts_p[:, 1], ".g")
    plt.plot(tr_n[:, 0], tr_n[:, 1], "Dr")
    plt.plot(tr_p[:, 0], tr_p[:, 1], "Dg")
    plt.show()


if __name__ == '__main__':
    main()
