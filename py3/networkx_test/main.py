import networkx as nx


def main():
    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    g.add_edge(2, 1)

    print(g.nodes())
    print(g.edges())
    print(list(g.adj[1]))
    print(g[1])

    print("Start search")
    bfs_edges = nx.bfs_edges(g, 0)
    for x in bfs_edges:
        print(x)


if __name__ == '__main__':
    main()
