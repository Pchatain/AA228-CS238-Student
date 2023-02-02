import sys
import os

import networkx as nx


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    """
    Read a csv file and write a graph file to outfile.

    Args:
        infile (str): path to input csv file
        outfile (str): path to output graph file
    """
    with open(infile, 'r') as f:
        lines = f.readlines()
        print(f"liens is {lines}")

    G = nx.Graph()
    print(f"G is {G}")
    G.add_node(1)
    print(f"G is {G}")
    # write_gph(networkx.DiGraph(), {}, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    data_dir = "data"
    graph_dir = "graphs"
    inputfilename = os.path.join(data_dir, sys.argv[1])
    outputfilename = os.path.join(graph_dir, sys.argv[2])
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
