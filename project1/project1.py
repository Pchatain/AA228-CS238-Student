import sys
import os

import networkx as nx

DEBUG = True

def write_gph(dag, idx2names, filename):
    """
    Write a graph to a file.

    Args:
        dag (networkx.DiGraph): a directed acyclic graph
        idx2names (dict): a mapping from node indices to node names
        filename (str): path to output file
    
    """
    with open(filename, 'w') as f:
        for edge in dag.edges():
            if DEBUG: print(f"edge is {edge}")
            f.write("{},{}\n".format(edge[0], edge[1]))


def compute(infile, outfile):
    """
    Read a csv file and write a graph file to outfile.

    Args:
        infile (str): path to input csv file
        outfile (str): path to output graph file
    """
    G = nx.Graph()
    idx2names = {}
    with open(infile, 'r') as f:
        lines = f.readlines()
        last_name = None
        for i, name in enumerate(lines[0].split(",")):
            name = name.strip("\n").strip("\"")
            if DEBUG: print(f"name is {name}")
            G.add_node(name)
            if i > 0:
                G.add_edge(last_name, name)
            last_name = name
            
        # print(f"liens is {lines}")

    
    # for i in range(3):
    #     idx = i*2
    #     G.add_node(idx)
    #     G.add_node(idx+1)
    #     G.add_edge(idx, idx+1)
    #     idx2names[idx] = f"Parent{i}"
    #     idx2names[idx+1] = f"Child{i}"
    if DEBUG: 
        print(f"G is {G}")
        print(f"The nodes of G are {G.nodes()}")
        print(f"The edges of G are {G.edges()}")
        print("The adjacency matrix of G is")
        print(nx.adjacency_matrix(G).todense())
    
    write_gph(G, idx2names, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfolder>")
    data_dir = "data"
    graph_dir = sys.argv[2] + "_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    inputfilename = os.path.join(data_dir, sys.argv[1])
    graph_type = inputfilename.split(os.path.sep)[-1].split(".")[0]
    outputfilename = os.path.join(graph_dir, graph_type + ".gph")
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
