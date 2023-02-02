import sys
import os

import networkx as nx
import scipy
import numpy as np
import pgmpy
import pandas as pd

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
            f.write("{},{}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def sub2ind(siz, x):
    k = np.concatenate(([1], np.cumprod(siz[:-1])))
    return np.dot(k, x - 1) + 1


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
    q = [np.prod([vars[j].r for j in G.predecessors(i)], dtype=int) for i in range(n)]
    M = [np.zeros((vars[i].r, q[i])) for i in range(n)]

    # Use pandas groupby function to fill in M
    # print(f"Groupby {[name for name in D.columns]} produces the following matrix")
    # grouped = D.groupby(by=[name for name in D.columns]).size().reset_index(name='counts')
    # print(grouped)
    # print("--------")
    for i in range(n):
        print("-----------STARTING NEW NODE------------")
        print(f"i is {i} and idx2names[i] is {idx2names[i]}")
        parents = [idx2names[j] for j in G.predecessors(i)]
        print(f"parents is {parents} for node {i}")
        grouped = D.groupby(parents + [idx2names[i]]).size().reset_index(name='counts')
        print(f"grouped is \n {grouped}")
        print("--------")
        print(f"The shape of M[i] is {M[i].shape}")
        if len(parents) > 0:
            grouped = grouped.pivot(index=idx2names[i], columns=parents, values='counts').to_numpy()
        else:
            grouped = grouped.set_index('age')['counts'].to_frame()
            print(f"the reformated data has type {type(grouped)}")
            grouped = grouped.to_numpy()
        print(f"tjhe reformated data is \n {grouped}")
        print(f"with shape {grouped.shape}")
        print("--------END NODE------------")
        M[i] = grouped



    # iterate over the rows of the grouped dataframe and update m

        
    return M


def prior(vars, G):
    """
    Note that this code assumes that inneighbors(G,i) returns the indices
    of the neighbors of node i in graph G, and that np.prod is the function
    for computing the product of an array of numbers.
    """
    n = len(vars)
    r = [vars[i].r for i in range(n)]
    q = [np.prod([r[j] for j in G.parents(i)]) for i in range(n)]
    return q


def bayesian_score(vars, G, D):
    """
    Compute the Bayesian score of a DAG.
    """
    n = len(vars)
    M = statistics(vars, G, D)
    if DEBUG: print(f"M is {M}")
    alpha = prior(vars, G)
    if DEBUG: print(f"alpha is {alpha}")

    return np.sum(bayesian_score_component(M[i], alpha[i]) for i in range(1,n))


def bayesian_score_component(M, alpha):
    """
    Compute the Bayesian score component of a matrix M.
    
    Args:
        M (numpy.ndarray): a matrix
        alpha (float): a parameter
        
    Returns:
        float: the Bayesian score component
    """
    p = np.sum(scipy.special.loggama(alpha + sum(M, axis=1)))
    p -= np.sum(scipy.special.loggama(alpha))
    p += np.sum(scipy.special.loggama(np.sum(alpha, dims=2)))
    p -= np.sum(scipy.special.loggama(np.sum(alpha, dims=2) + np.sum(M, dims=2)))
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
    
    # For each edge, there is a variable for each value the parent of that edge can have.
    # create a variable for each column in the dataframe
    vars = [Variable(col, 0) for col in D.columns]
    if DEBUG: print(f"vars is {vars}")
    # for each column in the dataframe, get the number of unique values in that column using pandas
    for i in range(D.shape[1]):
        vars[i].r = len(D[D.columns[i]].unique())
    for i in range(D.shape[1]):
        if DEBUG: print(f"Variable {vars[i].name}: {vars[i].r}")
    
    M = statistics(vars, G, D, idx2names)
    if DEBUG: print(f"M is {M}")
    # make the vars list from the graph G
    # vars = [Variable(i, D.shape[1]) for i in range(D.shape[1])]
    # vars = [Var(i, D.shape[1]) for i in range(D.shape[1])]
    # score = bayesian_score(vars, G, D)
    # print(f"Bayesian score is {score}")
    # return score

def main():
    test = False
    data_dir = "data"
    if len(sys.argv) == 2:
        print("Running unit test beacuse no args were passed")
        test = True
        inputfilename = os.path.join(data_dir, sys.argv[1])
        compute(inputfilename, "testing", test)
        return 0
    elif len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfolder>")
    graph_dir = sys.argv[2] + "_graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    inputfilename = os.path.join(data_dir, sys.argv[1])
    graph_type = inputfilename.split(os.path.sep)[-1].split(".")[0]
    outputfilename = os.path.join(graph_dir, graph_type + ".gph")
    compute(inputfilename, outputfilename, test)

if __name__ == '__main__':
    main()
