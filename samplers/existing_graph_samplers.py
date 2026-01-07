import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from littleballoffur import (
    CommonNeighborAwareRandomWalkSampler,
    CommunityStructureExpansionSampler,
    DegreeBasedSampler,
    ForestFireSampler,
    FrontierSampler,
    MetropolisHastingsRandomWalkSampler,
    NonBackTrackingRandomWalkSampler,
    PageRankBasedSampler,
    RandomEdgeSampler,
    RandomEdgeSamplerWithInduction,
    RandomNodeEdgeSampler,
    RandomNodeNeighborSampler,
    RandomNodeSampler,
    RandomWalkSampler,
    RandomWalkWithRestartSampler,
    ShortestPathSampler,
    SnowBallSampler,
)

from hypothesis_testing import hypothesis_testing
from instance_extraction import extract_attributes, extract_instances
from samplers.phase import PHASE
from samplers.phase_opt import Opt_PHASE
from samplers.subgraph_sampler import TriangleSubgraphSampler


class RandomNodeEdgeSampler_new(RandomNodeEdgeSampler):
    def sample(self, graph):
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        self._create_initial_edge_set(graph)
        new_graph = graph.edge_subgraph(list(self._sampled_edges))
        return new_graph


class RandomEdgeSampler_new(RandomEdgeSampler):
    def sample(self, graph):
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        self._create_initial_edge_set(graph)
        new_graph = graph.edge_subgraph(self._sampled_edges)
        return new_graph


class CommonNeighborAwareRandomWalkSampler_new(CommonNeighborAwareRandomWalkSampler):
    def _do_a_step(self, graph):
        self._get_node_scores(graph, self._current_node)
        self._current_node = np.random.choice(
            self._sampler[self._current_node]["neighbors"],
            1,
            replace=False,
            p=self._sampler[self._current_node]["scores"],
        )[0]
        self._sampled_nodes.add(self._current_node)


class FrontierSampler_new(FrontierSampler):
    def __init__(
        self, number_of_seeds: int = 10, number_of_nodes: int = 100, seed: int = 42
    ):
        self.number_of_seeds = number_of_seeds
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.index = None
        self._set_seed()

    def _reweight(self, graph):
        if self.index is not None:
            self._seed_weights[self.index] = self.backend.get_degree(
                graph, self._seeds[self.index]
            )
        else:
            self._seed_weights = [
                self.backend.get_degree(graph, seed) for seed in self._seeds
            ]

        weight_sum = np.sum(self._seed_weights)
        self._seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]

    def _do_update(self, graph):
        sample = int(
            np.random.choice(self._seeds, 1, replace=False, p=self._seed_weights)[0]
        )
        self.index = int(self._seeds.index(sample))
        new_seed = int(random.choice(self.backend.get_neighbors(graph, sample)))
        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[self.index] = new_seed

    def sample(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        self._nodes = set()
        self._edges = set()
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_seed_set(graph)
        while len(self._nodes) < self.number_of_nodes:
            self._reweight(graph)
            self._do_update(graph)
        new_graph = graph.edge_subgraph(list(self._edges))
        return new_graph


class RandomWalkWithRestartSampler_new(RandomWalkWithRestartSampler):
    def _create_initial_node_set(self, graph, start_node):
        if start_node is not None:
            if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
                self._current_node = start_node
                self._sampled_nodes = set([self._current_node])
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            self._current_node = random.sample(
                range(self.backend.get_number_of_nodes(graph)), 1
            )
            self._current_node = self._current_node[0]
            self._sampled_nodes = set([self._current_node])
        self._initial_node = self._current_node


class SnowBallSampler_new(SnowBallSampler):
    def sample(self, graph, start_node: int = None):
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_seed_set(graph, start_node)
        while len(self._nodes) < self.number_of_nodes:
            if self._queue.qsize() == 0:
                raise Exception(
                    f"Can only extract {len(self._nodes)} < {self.number_of_nodes} nodes."
                )
            source = self._queue.get()
            neighbors = self._get_neighbors(graph, source)
            for neighbor in neighbors:
                if neighbor not in self._nodes:
                    self._nodes.add(neighbor)
                    self._queue.put(neighbor)
                    if len(self._nodes) >= self.number_of_nodes:
                        break

        new_graph = self.backend.get_subgraph(graph, self._nodes)
        return new_graph


class CommunityStructureExpansionSampler_new(CommunityStructureExpansionSampler):
    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.known_expansion = {}
        self._set_seed()

    def _choose_new_node(self, graph):
        largest_expansion = 0
        for node in self._targets:
            if node in self.known_expansion.keys():
                expansion = self.known_expansion[node]
            else:
                expansion = len(
                    set(self.backend.get_neighbors(graph, node)).difference(
                        self._sampled_nodes
                    )
                )

            self.known_expansion[node] = expansion

            if expansion >= largest_expansion:
                new_node = node

        self._sampled_nodes.add(new_node)
        self.known_expansion[new_node] = 0

        for node in self._targets:
            if graph.has_edge(node, new_node) or graph.has_edge(new_node, node):
                self.known_expansion[node] -= 1
