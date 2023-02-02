import Pkg; Pkg.add("Graphs")
import Pkg; Pkg.add("Printf")

using Graphs
using Printf

"""
Functions taken from the textbook
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D::Matrix{Int})
    """
    Example usage:
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 3, 2)
        vars = [Variable(:A,2), Variable(:B,2), Variable(:C,2)] D = [1 2 2 1; 1 2 2 1; 2 2 2 2]
        M = statistics(vars, G, D)
    
    Returns:
        M is a matrix of the counts of each occurance of the 
    """
    n = size(D, 1)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j=1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
        end
        M[i][j,k] += 1.0
    end
end 

function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

function compute(infile, outfile)
    G = SimpleDiGraph(3)
    add_edge!(G, 1, 2)
    add_edge!(G, 3, 2)
    vars = [Variable(:A,2), Variable(:B,2), Variable(:C,2)]
    D = [1 2 2 1; 1 2 2 1; 2 2 2 2]
    M = statistics(vars, G, D)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
