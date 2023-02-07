import sys
import os

import networkx as nx
import scipy
import numpy as np
import pgmpy
import pandas as pd
import matplotlib.pyplot as plt
import time

DEBUG       = False
DEBUG_BAYES = False
VERBOSE     = False
K2_LOG      = True
K2_DEBUG    = False

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
            f.write("{},{}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def sub2ind(siz, x):
    k = np.concatenate(([1], np.cumprod(siz[:-1])))
    return np.dot(k, x - 1) + 1

def calculate_q(vars, G):
    """
    Calculate the q vector for a DAG.
    
    Args:
        vars (list): a list of variables
        G (networkx.DiGraph): a directed acyclic graph
        
    Returns:
        q (list): a list of integers
    """
    n = len(vars)
    q = [np.max((1, np.prod([vars[j].r for j in G.predecessors(i)], dtype=int))) for i in range(n)]
    return q

def statistics(vars, G, D, idx2names):
    """
    Compute the statistics of a DAG, given a pandas datagrame and a graph.
    
    Args:
        vars (list): a list of variables
        G (networkx.DiGraph): a directed acyclic graph
        D (pandas.DataFrame): a dataframe, nxm where n is the number of nodes and m is the number of samples
    """
    n = D.shape[1]
    assert n == len(vars)
    if VERBOSE: print(f"n is {n}")

    # create a list of lists of lists of zeros
    q = calculate_q(vars, G)
    M = [np.zeros((q[i], vars[i].r), dtype=np.float32) for i in range(n)]
    saved_shapes_M = [M[i].shape for i in range(n)]
    # Use pandas groupby function to fill in M
    for i in range(n):
        if VERBOSE: print("-----------STARTING NEW NODE------------")
        if VERBOSE: print(f"i is {i} and idx2names[i] is {idx2names[i]}")
        parents = [idx2names[j] for j in G.predecessors(i)]
        if VERBOSE: print(f"parents is {parents} for node {i}")
        grouped = D.groupby(parents + [idx2names[i]]).size().reset_index(name='counts')
        if VERBOSE: print(f"grouped is \n {grouped}")
        if VERBOSE: print("--------")
        if len(parents) > 0:
            if VERBOSE: print("has parents")
            grouped = grouped.pivot(index=idx2names[i], columns=parents, values='counts').fillna(0).to_numpy().T
        else:
            if VERBOSE: print("no parents")
            grouped = grouped['counts'].fillna(0).to_numpy()
            if VERBOSE: print(f"grouped has shape {grouped.shape}")
            grouped = grouped.reshape(1,-1)
            if VERBOSE: print(f"new shape is {grouped.shape}")
        if VERBOSE: print(f"tjhe reformated data is \n {grouped}")
        if VERBOSE: print(f"with shape {grouped.shape}")
        if VERBOSE: print("--------END NODE------------")
        
        if grouped.astype(np.float32).shape != saved_shapes_M[i]:
            if K2_DEBUG: print(f"Shape of M[{i}] is {M[i].shape} but should be {saved_shapes_M[i]}")
            grouped = np.pad(grouped, ((0, M[i].shape[0] - grouped.shape[0]), (0, M[i].shape[1] - grouped.shape[1])))
            if K2_DEBUG: print(f"grouped is now shape {grouped.shape} \n {grouped}")
        M[i] = grouped.astype(np.float32)
        assert M[i].shape == saved_shapes_M[i], f"Shape of M[{i}] is {M[i].shape} but should be {saved_shapes_M[i]}"
    if VERBOSE: print(f"M is after all statistics {M}")
    return M


def prior(vars, G):
    """
    Compute the uniform prior of a DAG.
    
    Args:
        vars (list): a list of variables
        G (networkx.DiGraph): a directed acyclic graph
    """
    n = len(vars)
    q = calculate_q(vars, G)
    alpha = [np.ones((q[i], vars[i].r)) for i in range(n)]
    return alpha


def bayesian_score(vars, G, D, idx2names):
    """
    Compute the Bayesian score of a DAG.
    """
    n = len(vars)
    M = statistics(vars, G, D, idx2names)
    if DEBUG_BAYES:
        for i in range(n):
            print(f"M[{i}] has shape {M[i].shape}")
    if DEBUG_BAYES: print(M)
    alpha = prior(vars, G) # TODO Optimization: This recomputes q as in statistics
    if VERBOSE: print(alpha)
    if DEBUG_BAYES: print(f"alpha has shape {alpha[0].shape}")
    if K2_DEBUG:
        for i in range(n):
            print(f"alpha[{i}] has shape {alpha[i].shape}")
            print(f"M[{i}] has shape {M[i].shape}")
    components = [bayesian_score_component(M[i], alpha[i], i) for i in range(n)]
    return sum(components)


def bayesian_score_component(M, alpha, i):
    """
    Compute the Bayesian score component of a matrix M.
    
    Args:
        M (numpy.ndarray): a matrix with shape (n_nodes, n_parental_configurations, n_values (k))
        alpha (float): a parameter with shape (n_nodes, n_parental_configurations, n_values (k))
        
    Returns:
        float: the Bayesian score component
    """
    if DEBUG_BAYES: print(f"alpha is {alpha}")
    if DEBUG_BAYES: print(f"M is {M}")
    assert M.shape == alpha.shape, f"Component size mismatch. M has shape {M.shape} and alpha has shape {alpha.shape}"
    # print(scipy.special.loggamma(alpha + M))
    # print(scipy.special.loggamma(alpha))
    # print(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    # if DEBUG_BAYES: print(f"M has shape {M.shape}")
    p = sum(scipy.special.loggamma(alpha + M).reshape(-1)) # 2
    
    assert sum(scipy.special.loggamma(alpha).reshape(-1)) == 0
    p -= sum(scipy.special.loggamma(alpha).reshape(-1)) # 4

    # print(sum(scipy.special.loggamma(np.sum(alpha, axis=1)).reshape(-1)))
    p += sum(scipy.special.loggamma(np.sum(alpha, axis=1)).reshape(-1)) # 1

    # print(-sum(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)).reshape(-1)))
    p -= sum(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)).reshape(-1)) # 3

    if DEBUG: print(f"p is {p}")
    return p


class Variable:
    """
    A variable in a Bayesian network. For each edge in the graph G,
    there is a corresponding variable in the list vars for every value
    the parent node has.

    Args:
        name (str): the name of the variable
        r (int): the number of values the variable can take on
    """
    def __init__(self, name, r):
        self.name = name
        self.r = r

def k2_alg(vars, D, idx2names):
    """
    Run the K2 algorithm on a dataset D.

    Args:
        vars (list): a list of variables
        D (numpy.ndarray): a dataset with shape (n_samples, n_features)
        idx2names (dict): a mapping from indices to names
    """
    start_time = time.time()
    Graph_k2 = nx.DiGraph()
    Graph_k2.add_nodes_from(range(len(vars)))
    if K2_DEBUG: print(f"vars is {vars}")
    if K2_DEBUG: print(f"graph is {Graph_k2.nodes}")
    for i in range(1, len(vars)):
        y = bayesian_score(vars, Graph_k2, D, idx2names)
        while True:
            y_best, j_best = float("-inf"), 0
            for j in range(i):
                if not Graph_k2.has_edge(j, i):
                    Graph_k2.add_edge(j, i)
                    y_prime = bayesian_score(vars, Graph_k2, D, idx2names)
                    if y_prime > y_best:
                        y_best, j_best = y_prime, j
                    Graph_k2.remove_edge(j, i)
                    if K2_DEBUG: print(f"bad edge. y_best is {y_best}, y_prime is {y_prime}, j_best is {j_best}, j is {j}")
            if y_best > y:
                if K2_DEBUG: print(f"y_best is {y_best}, y is {y}. Graph_k2 has {Graph_k2.edges}")
                y = y_best
                Graph_k2.add_edge(j_best, i)
            else:
                break
        if K2_LOG: print(f"Finiished one while loop, y is {y}")
    if K2_LOG: print(f"Finished k2_alg, Graph_k2 is {Graph_k2.edges}")
    end_time = time.time()
    print(f"K2 algorithm took {end_time - start_time} seconds")
    return Graph_k2


def compute(infile, outfile, test=False, k2=False):
    """
    Read a csv file and write a graph file to outfile.

    Args:
        infile (str): path to input csv file
        outfile (str): path to output graph file

    Returns:
        D (np.ndarray): a data matrix of shape (n, m) 
        where n is the number of samples and m is the number of nodes. 
    """
    G = nx.DiGraph()
    idx2names = {}
    # read thet csv file into a dataframe
    D = pd.read_csv(infile)

    for i in range(D.shape[1]):
        idx2names[i] = D.columns[i]
    if "example" not in infile and not k2:
        # add an edge to the graph from column to column + 1 for each column in the dataframe
        for i in range(D.shape[1] - 1):
            G.add_edge(i, i+1)
        if DEBUG: 
            print(f"Datafframe is {D}")
            print(f"G is {G}")
            print(f"The nodes of G are {G.nodes()}")
            print(f"The edges of G are {G.edges()}")
            print("The adjacency matrix of G is")
            # print(nx.adjacency_matrix(G).todense())
    
    if test and "example" in infile:
        print(f"changing graph to be example graph")
        """
        parent1,child1
        parent2,child2
        parent3,child3
        """
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        G.add_edge(4, 5)
        idx2names = {0: 'parent1', 1: 'child1', 2: 'parent2', 3: 'child2', 4: 'parent3', 5: 'child3'}
        for edge in G.edges():
            print(f"edge is {idx2names[edge[0]]} -> {idx2names[edge[1]]}")
        print(f"G is {G}")
        print(f"The nodes of G are {G.nodes()}")
        print(f"The edges of G are {G.edges()}")

    # For each edge, there is a variable for each value the parent of that edge can have.
    # create a variable for each column in the dataframe
    vars = [Variable(col, 0) for col in D.columns]
    if DEBUG: print(f"vars is {vars}")
    # for each column in the dataframe, get the number of unique values in that column using pandas
    for i in range(D.shape[1]):
        vars[i].r = len(D[D.columns[i]].unique())
    if DEBUG_BAYES: 
        for i in range(D.shape[1]):
            print(f"Variable {vars[i].name}: {vars[i].r}")
    if k2:
        G = k2_alg(vars, D, idx2names)
    score = bayesian_score(vars, G, D, idx2names)
    print(f"Bayesian score is {score}")
    if not test:
        print(f"Writing graph to {outfile}")
        write_gph(G, idx2names, outfile)
    return score

def rand_graph_neighbor(G):
    n = len(G.nodes)
    i = np.random.randint(0, n-1)
    j = (i + np.random.randint(1, n-1)) % n
    G_new = G.copy()
    if (i, j) in G.edges:
        G_new.remove_edge(i, j)
    else:
        G_new.add_edge(i, j)
    return G_new

def hill_climb(csv_file, graph_file, outfile):
    """
    Perform hill climbing on top of an existing graph.
    Note: This code is not used in the final submission
    because it created cycles in the graphs for some reason.

    Args:
        csv_file (str): path to csv file
        graph_file (str): path to graph file
        outfile (str): path to output graph file
    """
    G, idx2names = read_graph(graph_file, csv_file)
    D = pd.read_csv(csv_file)
    vars = [Variable(col, 0) for col in D.columns]
    for i in range(D.shape[1]):
        vars[i].r = len(D[D.columns[i]].unique())
    
    old_score = bayesian_score(vars, G, D, idx2names)
    print(f"old score is {old_score}")
    first_score = old_score
    # perform the hill climbing algorithm
    for iteration in range(200):
        G_new = rand_graph_neighbor(G)
        # if G_new is cyclic, skip it
        if len(list(nx.simple_cycles(G))) > 0:
            # print(f"Bad, skipping")
            continue
        new_score = bayesian_score(vars, G_new, D, idx2names)
        if new_score > old_score:
            G = G_new
            old_score = new_score
            print(f"New score is {old_score}")
        if iteration % 10 == 0:
            print(f"iteration {iteration} score is {old_score}, saving checkpoint")
            write_gph(G, idx2names, outfile + ".checkpoint")

    score = bayesian_score(vars, G, D, idx2names)
    print(f"Bayesian score is {score}")
    if score > first_score:
        print(f"Writing graph to {outfile} because {score} > {first_score}")
        write_gph(G, idx2names, outfile)

def read_graph(graph_filename, csv_filename):
    """
    Read a graph file and return a networkx graph
    
    Args:
        filename (str): path to graph file
    Returns:
        G (nx.DiGraph): a networkx graph
    """
    G = nx.DiGraph()
    idx2names = {}
    names2idx = []
    with open(csv_filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                header = [field.strip('"\n') for field in line.split(",")]
                idx2names = {i: header[i] for i in range(len(header))}
                names2idx = {header[i]: i for i in range(len(header))}
                print(idx2names)
                break
    with open(graph_filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(",")
            if len(line) == 0:
                continue
            G.add_edge(int(names2idx[line[0]]), int(names2idx[line[1]]))
    return G, idx2names

def main():
    test = False
    data_dir = "data"
    if len(sys.argv) == 2:
        print("Running unit test beacuse no args were passed")
        test = True
        inputfilename = sys.argv[1]
        compute(inputfilename, "testing", test)
        return 0
    elif len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception("usage: python project1.py <infile>.csv <outfolder> optional:<graphfile>.gph")
    graph_dir = sys.argv[2] + "_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    inputfilename = sys.argv[1]
    graph_type = inputfilename.split(os.path.sep)[-1].split(".")[0]
    outputfilename = os.path.join(graph_dir, graph_type + ".gph")
    if len(sys.argv) == 3:
        compute(inputfilename, outputfilename, test, k2=True) # k2 starting point
    else:
        assert len(sys.argv) == 4, "usage: python project1.py <infile>.csv <outfolder> optional:<graphfile>.gph"
        hill_climb(inputfilename, graph_file=sys.argv[3], outfile=outputfilename)

if __name__ == '__main__':
    # main()
    G, idx2names = read_graph(sys.argv[1], sys.argv[2])
    print("graph has cycles list: ")
    print(list(nx.simple_cycles(G)))
    # write_gph(G, idx2names, sys.argv[3])

    # read the graph file
    """
    # code for saving the produced graphs as a pdf
    G, idx2names = read_graph(sys.argv[1], sys.argv[2])
    # then make a pdf of the graphs usting networkx
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.savefig(sys.argv[1] + ".png")"""

    """
    Implemented k2 and ran it. It produced:
        For small graph:
            Finiished one while loop, y is -4166.225858784904
            Finiished one while loop, y is -4166.225858784904
            Finiished one while loop, y is -4157.072323526103
            Finiished one while loop, y is -4073.192463465056
            Finiished one while loop, y is -4050.445393155355
            Finiished one while loop, y is -4015.6969794439788
            Finiished one while loop, y is -3835.6794252127916
        For medium graph:
            Finiished one while loop, y is -45367.62511363247
            Finiished one while loop, y is -45293.82258329491
            ...
            Finiished one while loop, y is -42698.695641014245
            Finiished one while loop, y is -42017.4490617635
        for large graph:
            Finiished one while loop, y is -486675.61757787236
            Finiished one while loop, y is -483519.6635786245
            ...
            Finiished one while loop, y is -404564.6021664627
            Finiished one while loop, y is -404192.31703113194
    """
