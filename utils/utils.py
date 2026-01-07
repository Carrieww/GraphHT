"""
Device and seed setup utilities.
"""

import random

import numpy as np
import torch


def setup_device(args):
    """Set up device (CUDA/CPU)."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(args):
    """Set up random seeds."""
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))


def clean():
    """Clean CUDA cache."""
    torch.cuda.empty_cache()
    print("finished clean!")


def find_all_triangles_vf2(graph):
    """
    Find all unique triangles in the graph using VF2 isomorphism algorithm.

    Args:
        graph: The graph to search in

    Returns:
        List of unique triangles, each represented as a set of 3 node IDs
    """
    # Build a triangle pattern graph (3 nodes forming a triangle)
    import networkx as nx
    from networkx.algorithms import isomorphism

    if graph.is_directed():
        pattern_graph = nx.DiGraph()
    else:
        pattern_graph = nx.Graph()

    # Add 3 nodes forming a triangle
    pattern_graph.add_edge(0, 1)
    pattern_graph.add_edge(1, 2)
    pattern_graph.add_edge(2, 0)

    # Use VF2 matcher (no attribute matching, just structure)
    if graph.is_directed():
        matcher = isomorphism.DiGraphMatcher(graph, pattern_graph)
    else:
        matcher = isomorphism.GraphMatcher(graph, pattern_graph)

    # Find all subgraph isomorphisms
    matches = list(matcher.subgraph_isomorphisms_iter())

    # Extract unique triangles (as sets to avoid duplicates)
    unique_triangles = set()
    for match in matches:
        # match is a dict: {graph_node_id: pattern_node_id}
        # Keys are graph nodes, values are pattern nodes (0, 1, 2)
        triangle_nodes = set(match.keys())  # Get graph node IDs
        if len(triangle_nodes) == 3:
            # Use frozenset for hashing
            unique_triangles.add(frozenset(triangle_nodes))

    # Convert back to list of sets
    return [set(triangle) for triangle in unique_triangles]
