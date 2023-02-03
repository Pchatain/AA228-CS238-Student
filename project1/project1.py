import sys
import os

import networkx as nx
import scipy
import numpy as np
import pgmpy
import pandas as pd

DEBUG = True
DEBUG_BAYES = True

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
    q = [np.prod([vars[j].r for j in G.predecessors(i)], dtype=int) for i in range(n)]
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
    if DEBUG: print(f"n is {n}")

    # create a list of lists of lists of zeros
    q = calculate_q(vars, G)
    M = [np.zeros((vars[i].r, q[i])) for i in range(n)]

    # Use pandas groupby function to fill in M
    for i in range(n):
        if DEBUG: print("-----------STARTING NEW NODE------------")
        if DEBUG: print(f"i is {i} and idx2names[i] is {idx2names[i]}")
        parents = [idx2names[j] for j in G.predecessors(i)]
        if DEBUG: print(f"parents is {parents} for node {i}")
        grouped = D.groupby(parents + [idx2names[i]]).size().reset_index(name='counts')
        if DEBUG: print(f"grouped is \n {grouped}")
        if DEBUG: print("--------")
        if DEBUG: print(f"The shape of M[i] is {M[i].shape}")
        if len(parents) > 0:
            grouped = grouped.pivot(index=idx2names[i], columns=parents, values='counts').fillna(0).to_numpy()
        else:
            grouped = grouped.set_index(idx2names[i])['counts'].fillna(0).to_numpy()
        if DEBUG: print(f"tjhe reformated data is \n {grouped}")
        if DEBUG: print(f"with shape {grouped.shape}")
        if DEBUG: print("--------END NODE------------")
        M[i] = grouped
    if DEBUG: print(f"M is after all statistics {M}")
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
    alpha = [np.ones((vars[i].r, q[i])) for i in range(n)]
    return alpha


def bayesian_score(vars, G, D, idx2names):
    """
    Compute the Bayesian score of a DAG.
    """
    n = len(vars)
    M = statistics(vars, G, D, idx2names)
    if DEBUG_BAYES: print(f"M has shape {M[0].shape}")
    alpha = prior(vars, G) # TODO Optimization: This recomputes q as in statistics
    print(alpha)
    if DEBUG_BAYES: print(f"alpha has shape {alpha[0].shape}")
    components = [bayesian_score_component(M[i], alpha[i]) for i in range(1,n)]
    return sum(components)


def bayesian_score_component(M, alpha):
    """
    Compute the Bayesian score component of a matrix M.
    
    Args:
        M (numpy.ndarray): a matrix with shape (n_nodes, n_parental_configurations, n_values (k))
        alpha (float): a parameter with shape (n_nodes, n_parental_configurations, n_values (k))
        
    Returns:
        float: the Bayesian score component
    """
    assert M.shape == alpha.shape
    if len(M.shape) == 1:
        M = M.reshape(-1,1)
        print(f"M was bad shape and is now {M.shape}")
        print(f"Alpha is {alpha}")
        print(f"Np.sum(M, axis=1) is {np.sum(M, axis=1)}")
        print(scipy.special.loggamma(alpha + M))
        print(scipy.special.loggamma(alpha))
        print(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))

        p = sum(scipy.special.loggamma(alpha + M).reshape(-1)) # 2
        print(f"p is {p}")
        p -= sum(scipy.special.loggamma(alpha).reshape(-1)) # 4 zero potentially
        print(f"p is {p}")
        p += sum(scipy.special.loggamma(np.sum(alpha, axis=1)).reshape(-1)) # 1
        print(f"p is {p}")
        p -= sum(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)).reshape(-1)) # 3
        print(f"p is {p}")
        if DEBUG: print(f"m was bad shape and p is {p}")
        # TODO: This is almost certainly the problem?
        return p
    print(scipy.special.loggamma(alpha + M))
    print(scipy.special.loggamma(alpha))
    print(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    # if DEBUG_BAYES: print(f"M has shape {M.shape}")
    # if DEBUG_BAYES: print(f"alpha has shape {alpha.shape}")
    p =  sum(scipy.special.loggamma(alpha + M).reshape(-1))
    # if DEBUG_BAYES: print(f"p has shape {p.shape}")
    p -= sum(scipy.special.loggamma(alpha).reshape(-1))
    # if DEBUG_BAYES: print(f"p has shape {p.shape}")
    p += sum(scipy.special.loggamma(np.sum(alpha, axis=1)).reshape(-1))
    # if DEBUG_BAYES: print(f"p has shape {p.shape}")
    p -= sum(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)).reshape(-1))
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

def compute(infile, outfile, test=False):
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
    # add an edge to the graph from column to column + 1 for each column in the dataframe
    for i in range(D.shape[1] - 1):
        G.add_edge(i, i+1)
            
    if DEBUG: 
        print(f"Datafframe is {D}")
        print(f"G is {G}")
        print(f"The nodes of G are {G.nodes()}")
        print(f"The edges of G are {G.edges()}")
        print("The adjacency matrix of G is")
        print(nx.adjacency_matrix(G).todense())
    if not test:
        write_gph(G, idx2names, outfile)
    
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

    # For each edge, there is a variable for each value the parent of that edge can have.
    # create a variable for each column in the dataframe
    vars = [Variable(col, 0) for col in D.columns]
    if DEBUG: print(f"vars is {vars}")
    # for each column in the dataframe, get the number of unique values in that column using pandas
    for i in range(D.shape[1]):
        vars[i].r = len(D[D.columns[i]].unique())
    for i in range(D.shape[1]):
        if DEBUG_BAYES: print(f"Variable {vars[i].name}: {vars[i].r}")
    if test:
        M = statistics(vars, G, D, idx2names)
        if DEBUG: print(f"M is {M}")

        score = bayesian_score(vars, G, D, idx2names)
        if DEBUG_BAYES: print(f"Bayesian score is {score}")
        return score

def main():
    test = False
    data_dir = "data"
    if len(sys.argv) == 2:
        print("Running unit test beacuse no args were passed")
        test = True
        inputfilename = sys.argv[1]
        compute(inputfilename, "testing", test)
        return 0
    elif len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfolder>")
    graph_dir = sys.argv[2] + "_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    inputfilename = sys.argv[1]
    graph_type = inputfilename.split(os.path.sep)[-1].split(".")[0]
    outputfilename = os.path.join(graph_dir, graph_type + ".gph")
    compute(inputfilename, outputfilename, test)

if __name__ == '__main__':
    main()
