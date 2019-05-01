import networkx as nx
import matplotlib.pyplot as plt


def main():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(1, 4)
    g.add_edge(2, 4)
    g.add_edge(3, 4)

    print(g.nodes())
    print(g.edges())
    print(list(g.adj[1]))
    print(g[1])

    print("Start search")
    bfs_res = nx.bfs_predecessors(g, 0)
    for x in bfs_res:
        print(x)
    print("Start search")
    bfs_res = nx.bfs_successors(g, 0)
    for x in bfs_res:
        print(x)
    print("Start search")
    bfs_res = nx.bfs_tree(g, 0)
    for x in bfs_res:
        print(x)

    # nx.draw(g)
    # plt.show()


if __name__ == '__main__':
    main()
