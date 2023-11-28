import time
import random
import networkx as nx
import numpy as np

from new_graph_hypo_postprocess import new_graph_hypo_result
from littleballoffur.sampler import Sampler
from littleballoffur import (
    RandomNodeNeighborSampler,
    RandomWalkSampler,
    ForestFireSampler,
    ShortestPathSampler,
    MetropolisHastingsRandomWalkSampler,
    CommunityStructureExpansionSampler,
    NonBackTrackingRandomWalkSampler,
    SnowBallSampler,
    RandomWalkWithRestartSampler,
    FrontierSampler,
    CommonNeighborAwareRandomWalkSampler,
    RandomNodeSampler,
    DegreeBasedSampler,
    PageRankBasedSampler,
    RandomEdgeSampler,
    RandomNodeEdgeSampler,
    RandomEdgeSamplerWithInduction,
)


def time_sampling_extraction(
    args, new_graph, result_list, time_used_list, time_one_sample_start, num_sample
):
    args.logger.info(
        f"Time for sampling once {args.ratio}: {round(time.time() - time_one_sample_start, 2)}."
    )
    time_used_list["sampling"].append(round(time.time() - time_one_sample_start, 2))

    result_list = new_graph_hypo_result(args, new_graph, result_list, time_used_list)

    return result_list, time_used_list


def sample_graph(args, graph, result_list, time_used_list, sampler_type):
    sampler_mapping = {
        "RNNS": RandomNodeNeighborSampler,
        "SRW": RandomWalkSampler,
        "FFS": ForestFireSampler,
        "ShortestPathS": ShortestPathSampler,
        "MHRWS": MetropolisHastingsRandomWalkSampler,
        "CommunitySES": CommunityStructureExpansionSampler_new,
        "NBRW": NonBackTrackingRandomWalkSampler,
        "SBS": SnowBallSampler_new,
        "RW_Starter": RandomWalkWithRestartSampler_new,
        "FrontierS": FrontierSampler_new,
        "CNARW": CommonNeighborAwareRandomWalkSampler_new,
        "RNS": RandomNodeSampler,
        "DBS": DegreeBasedSampler,
        "PRBS": PageRankBasedSampler,
        "RES": RandomEdgeSampler_new,
        "RNES": RandomNodeEdgeSampler_new,
        "RES_Induction": RandomEdgeSamplerWithInduction,
        "ours": newSampler,
    }

    if sampler_type not in sampler_mapping:
        raise ValueError("Invalid sampler type.")

    sampler_class = sampler_mapping[sampler_type]

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        seed = int(args.seed) * num_sample

        if sampler_type == "FFS":
            model = sampler_class(number_of_nodes=args.ratio, seed=seed, p=0.4)
        elif sampler_type == "MHRWS":
            model = sampler_class(number_of_nodes=args.ratio, seed=seed, alpha=0.5)
        elif sampler_type == "SBS":
            model = sampler_class(number_of_nodes=args.ratio, seed=seed, k=200)
        elif sampler_type == "RW_Starter":
            model = sampler_class(number_of_nodes=args.ratio, seed=seed, p=0.01)
        elif sampler_type in ["RES", "RNES", "RES_Induction"]:
            model = sampler_class(number_of_edges=args.ratio, seed=seed)
        elif sampler_type == "ours":
            if args.hypo in [1, 3]:
                no_repeat = "edge"
            else:
                no_repeat = "node"
            model = newSampler(
                args.attribute[str(list(args.attribute.keys())[0])]["path"],
                number_of_nodes=args.ratio,
                no_repeat=no_repeat,
                seed=seed,
            )
        else:
            model = sampler_class(number_of_nodes=args.ratio, seed=seed)

        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()

        args.logger.info(
            f"The sampled graph has {num_nodes} nodes and {num_edges} edges."
        )
        print(f"The sampled graph has {num_nodes} nodes and {num_edges} edges.")

        result_list, time_used_list = time_sampling_extraction(
            args,
            new_graph,
            result_list,
            time_used_list,
            time_one_sample_start,
            num_sample,
        )

    return result_list, time_used_list


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
                    # if len(self._nodes) :
                    # print(len(self._nodes))
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
                # print("here")
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


import math


class newSampler(Sampler):
    r"""
    Args:
        number_of_seeds (int): Number of seed nodes. Default is 50.
        number_of_nodes (int): Number of nodes to sample. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(
        self,
        condition: list,
        no_repeat: str,
        number_of_seeds: int = 50,
        number_of_nodes: int = 100,
        seed: int = 42,
    ):
        self.number_of_seeds = number_of_seeds
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self.index = None
        self.path = condition
        self.good_node_map = {}
        self.edge_count = 0
        self.node_count = 0
        self.no_repeat = no_repeat

    def assign_weight(self, graph, node):
        for index, condition in enumerate(self.path):
            score = 0.1
            flag = True
            if graph.nodes[node]["label"] == condition["type"]:
                score += 1
                for attr, v in condition["attribute"].items():
                    if graph.nodes[node][attr] != v:
                        score = 0.1
                        flag = False
                        break
                    else:
                        score += 1
            else:
                flag = False
            if flag:
                self.good_node_map[node] = score
                return score
        self.good_node_map[node] = score
        return score

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        if self.index is not None:
            self._seed_weights[self.index] = self._check_map_weight(
                graph, self._seeds[self.index]
            )

        else:
            self._seed_weights = []
            for i in self._seeds:
                weight = self.assign_weight(graph, i)
                #### do nothing
                self._seed_weights.append(weight)

        weight_sum = np.sum(self._seed_weights)
        self._norm_seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]

    def _create_initial_seed_set(self, graph):
        """
        Choosing initial nodes.
        """
        nodes = self.backend.get_nodes(graph)
        ### do nothing
        self._seeds = random.sample(nodes, self.number_of_seeds)

    def _check_map_weight(self, graph, node):
        if node in self.good_node_map:
            weight = self.good_node_map[node]
        else:
            weight = self.assign_weight(graph, node)
        return weight

    def _choose_new_seed(self, graph, sample, neighbors):
        if self.no_repeat == "edge":
            return self._choose_seed_with_edge_check(graph, sample, neighbors)
        elif self.no_repeat == "node":
            return self._choose_seed_without_node_repeats(graph, sample, neighbors)

    def _choose_seed_with_edge_check(self, graph, sample, neighbors):
        neighbor_weight = [
            0.01 if (sample, i) in self._edges else self._check_map_weight(graph, i)
            for i in neighbors
        ]
        return self._calculate_new_seed(graph, neighbors, neighbor_weight)

    def _choose_seed_without_node_repeats(self, graph, sample, neighbors):
        unvisited_neighbors = set(neighbors) - self._nodes
        unvisited_neighbors_list = (
            list(unvisited_neighbors) if unvisited_neighbors else neighbors
        )
        neighbor_weight = [
            self._check_map_weight(graph, i) for i in unvisited_neighbors_list
        ]
        return self._calculate_new_seed(
            graph, unvisited_neighbors_list, neighbor_weight
        )

    def _calculate_new_seed(self, graph, neighbor_list, neighbor_weight):
        weight_sum = np.sum(neighbor_weight)
        norm_seed_weights = [float(weight) / weight_sum for weight in neighbor_weight]
        return int(
            np.random.choice(neighbor_list, 1, replace=False, p=norm_seed_weights)[0]
        )

    def _do_update(self, graph):
        """
        Choose new seed node.
        """
        sample = int(
            np.random.choice(self._seeds, 1, replace=False, p=self._norm_seed_weights)[
                0
            ]
        )
        self.index = int(self._seeds.index(sample))
        neighbors = self.backend.get_neighbors(graph, sample)

        new_seed = self._choose_new_seed(graph, sample, neighbors)

        if (sample, new_seed) in self._edges:
            self.edge_count += 1
        if new_seed in self._nodes:
            self.node_count += 1
        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[self.index] = new_seed

        # if self.no_repeat == "edge":
        #     neighbor_weight = []
        #     for i in neighbors:
        #         if (sample, i) in self._edges:
        #             weight = 0.01
        #         else:
        #             weight = self._check_map_weight(graph, i)
        #         neighbor_weight.append(weight)
        #     weight_sum = np.sum(neighbor_weight)
        #     norm_seed_weights = [
        #         float(weight) / weight_sum for weight in neighbor_weight
        #     ]
        #     new_seed = int(
        #         np.random.choice(neighbors, 1, replace=False, p=norm_seed_weights)[0]
        #     )
        #
        # elif self.no_repeat == "node":
        #     unvisited_neighbors = set(neighbors) - self._nodes
        #     unvisited_neighbors_list = list(unvisited_neighbors)
        #     neighbor_weight = []
        #     if len(unvisited_neighbors_list) == 0:
        #         unvisited_neighbors_list = neighbors
        #         pass
        #     for i in unvisited_neighbors_list:
        #         weight = self._check_map_weight(graph, i)
        #         neighbor_weight.append(weight)
        #
        #     weight_sum = np.sum(neighbor_weight)
        #     norm_seed_weights = [
        #         float(weight) / weight_sum for weight in neighbor_weight
        #     ]
        #     new_seed = int(
        #         np.random.choice(
        #             unvisited_neighbors_list, 1, replace=False, p=norm_seed_weights
        #         )[0]
        #     )
        #
        # if (sample, new_seed) in self._edges:
        #     self.edge_count += 1
        # if new_seed in self._nodes:
        #     self.node_count += 1
        # self._edges.add((sample, new_seed))
        # self._nodes.add(sample)
        # self._nodes.add(new_seed)
        # self._seeds[self.index] = new_seed

    def sample(self, graph):
        self._nodes = set()
        self._edges = set()
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_seed_set(graph)
        while len(self._nodes) < self.number_of_nodes:
            self._reweight(graph)
            self._do_update(graph)
        new_graph = graph.edge_subgraph(list(self._edges))
        print(f"No. of repeat nodes: {self.node_count}.")
        print(f"No. of repeat edges: {self.edge_count}.")
        return new_graph


# class newSampler_V2(Sampler):
#     r"""
#     Args:
#         number_of_seeds (int): Number of seed nodes. Default is 50.
#         number_of_nodes (int): Number of nodes to sample. Default is 100.
#         seed (int): Random seed. Default is 42.
#     """

#     def __init__(
#         self,
#         condition: list,
#         theta: int = 50,
#         number_of_seeds: int = 50,
#         number_of_nodes: int = 100,
#         seed: int = 42,
#     ):
#         self.number_of_seeds = number_of_seeds
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed()
#         self.index = None
#         self.path = condition
#         self.theta = theta

#     def assign_weight(self, graph, node):
#         length = len(self.path)
#         for index, condition in enumerate(self.path):
#             flag = True
#             if graph.nodes[node]["label"] == condition["type"]:
#                 for attr, v in condition["attribute"].items():
#                     if graph.nodes[node][attr] != v:
#                         flag = False
#                         break
#             else:
#                 flag = False
#             if flag:
#                 if (
#                     self.index is not None
#                     and len(self._nodes) < self.number_of_nodes / 2
#                 ):
#                     return length - index
#                 elif (
#                     self.index is not None
#                     and len(self._nodes) >= self.number_of_nodes / 2
#                 ):
#                     return index + 1
#         return 0.1  # self._seed_weights.append(1)

#     def _reweight(self, graph):
#         """
#         Create new seed weights.
#         """
#         if self.index is not None:
#             self._seed_weights[self.index] = self.assign_weight(
#                 graph, self._seeds[self.index]
#             )
#         # self._seed_weights = []

#         else:
#             self._seed_weights = []
#             for i in self._seeds:
#                 weight = self.assign_weight(graph, i)
#                 #### do nothing
#                 self._seed_weights.append(weight)

#         weight_sum = np.sum(self._seed_weights)
#         # print(self._seed_weights)
#         self._norm_seed_weights = [
#             float(weight) / weight_sum for weight in self._seed_weights
#         ]
#     def _create_initial_seed_set(self, graph):
#         """
#         Choosing initial nodes.
#         """
#         nodes = self.backend.get_nodes(graph)
#         ### do nothing
#         self._seeds = random.sample(nodes, self.number_of_seeds)

#     def _do_update(self, graph):
#         """
#         Choose new seed node.
#         """
#         # print(self._seeds)
#         sample = int(
#             np.random.choice(self._seeds, 1, replace=False, p=self._norm_seed_weights)[
#                 0
#             ]
#         )
#         self.index = int(self._seeds.index(sample))
#         ### just randomly pick one
#         new_seed = int(random.choice(self.backend.get_neighbors(graph, sample)))

#         self._edges.add((sample, new_seed))
#         self._nodes.add(sample)
#         self._nodes.add(new_seed)
#         self._seeds[self.index] = new_seed

#     def sample(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
#         """
#         Arg types:
#             * **graph** *(NetworkX graph)* - The graph to be sampled from.

#         Return types:
#             * **new_graph** *(NetworkX graph)* - The graph of sampled nodes.
#         """
#         self._nodes = set()
#         self._edges = set()
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         self._create_initial_seed_set(graph)
#         while len(self._nodes) < self.number_of_nodes:
#             self._reweight(graph)
#             self._do_update(graph)
#         new_graph = graph.edge_subgraph(list(self._edges))
#         return new_graph

#
# class newSampler_v1(Sampler):
#     r"""
#     Args:
#         number_of_seeds (int): Number of seed nodes. Default is 50.
#         number_of_nodes (int): Number of nodes to sample. Default is 100.
#         seed (int): Random seed. Default is 42.
#     """
#
#     def __init__(
#         self,
#         condition: list,
#         theta: int = 50,
#         number_of_seeds: int = 50,
#         number_of_nodes: int = 100,
#         seed: int = 42,
#     ):
#         self.number_of_seeds = theta
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed()
#         self.index = None
#         self.path = condition
#         self.theta = theta
#
#     def assign_weight(self, graph, node):
#         # index = len(self.path)
#         for condition in self.path:
#             flag = True
#             if graph.nodes[node]["label"] == condition["type"]:
#                 for attr, v in condition["attribute"].items():
#                     if graph.nodes[node][attr] != v:
#                         flag = False
#                         break
#             else:
#                 flag = False
#             if flag:
#                 return 2
#         return 0.1  # self._seed_weights.append(1)
#
#     def _reweight(self, graph):
#         """
#         Create new seed weights.
#         """
#         self._seed_weights = []
#         for i in self._seeds:
#             weight = self.assign_weight(graph, i)
#             #### do nothing
#             self._seed_weights.append(weight)
#
#         weight_sum = np.sum(self._seed_weights)
#         # print(self._seed_weights)
#         self._seed_weights = [
#             float(weight) / weight_sum for weight in self._seed_weights
#         ]
#
#     def _create_initial_seed_set(self, graph):
#         """
#         Choosing initial nodes.
#         """
#         nodes = self.backend.get_nodes(graph)
#         self._seeds = random.sample(nodes, self.number_of_seeds)
#
#     def _do_update(self, graph):
#         """
#         Choose new seed node.
#         """
#         # print(self._seeds)
#         sample = np.random.choice(self._seeds, 1, replace=False, p=self._seed_weights)[
#             0
#         ]
#         self.index = self._seeds.index(sample)
#         new_seed = random.choice(self.backend.get_neighbors(graph, sample))
#
#         self._edges.add((sample, new_seed))
#         self._nodes.add(sample)
#         self._nodes.add(new_seed)
#         self._seeds[self.index] = new_seed
#
#     def sample(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
#         """
#         Arg types:
#             * **graph** *(NetworkX graph)* - The graph to be sampled from.
#
#         Return types:
#             * **new_graph** *(NetworkX graph)* - The graph of sampled nodes.
#         """
#         self._nodes = set()
#         self._edges = set()
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         self._create_initial_seed_set(graph)
#         while len(self._nodes) < self.number_of_nodes:
#             self._reweight(graph)
#             self._do_update(graph)
#         new_graph = graph.edge_subgraph(list(self._edges))
#         return new_graph
