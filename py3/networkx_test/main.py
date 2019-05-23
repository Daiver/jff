import networkx as nx
import matplotlib.pyplot as plt


def main():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(3, 1)
    g.add_edge(4, 2)

    print("Start search")
    bfs_res = nx.bfs_predecessors(g, 0)
    for x in bfs_res:
        print(x)
    print("Start search")
    bfs_res = nx.bfs_successors(g, 0)
    for x in bfs_res:
        print(x)
    print("Start search")
    bfs_res = nx.bfs_tree(g, 0, depth_limit=1)
    for x in bfs_res:
        print(x)

    nx.draw(g)
    plt.show()


if __name__ == '__main__':
    main()
