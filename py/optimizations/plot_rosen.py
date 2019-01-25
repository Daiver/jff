from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import rosen

def plotRosenbrock(markers):
    fig = plt.figure()
    ax = Axes3D(fig, azim = -128, elev = 43)
    s = .05
    X = np.arange(-2, 2.+s, s)
    Y = np.arange(-1, 3.+s, s)
    X, Y = np.meshgrid(X, Y)
    Z = (1.-X)**2 + 100.*(Y-X*X)**2
    # ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet)
    # Without using `` linewidth=0, edgecolor='none' '', the code may produce a graph with wide black edges, which 
    # will make the surface look much darker than the one illustrated in the figure above.
    ax.plot_wireframe(X, Y, Z, rstride = 1, cstride = 1)
    #ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet, linewidth=0, edgecolor='none')
    #ax.scatter(markers[0], markers[1], markers[2], c='r',s=100)
    #print [x for x in dir(ax) if x.find('plot') > -1]
    ax.plot3D(markers[0], markers[1], markers[2], c='r')
    #ax.scatter(markers[0][:1], markers[1][:1], markers[2][:1], c='r', marker='o', s=100)
    ax.scatter(markers[0], markers[1], markers[2], c='r', marker='o', s=100)
    ax.scatter(markers[0][-1:], markers[1][-1:], markers[2][-1:], c='b', marker='o', s=100)
    # Set the axis limits so that they are the same as in the figure above.
    ax.set_xlim([-2, 2.0])                                                       
    ax.set_ylim([-1, 3.0])                                                       
    ax.set_zlim([0, 2500]) 

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
#plt.savefig("Rosenbrock function.svg")

if __name__ == '__main__':
    plotRosenbrock(([1.0], [2.0], [5]))
