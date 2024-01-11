import math
from littleballoffur.sampler import Sampler
import numpy as np
import random


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
        no_repeat: int = 50,
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
        self.path_length = len(self.path)
        self.good_node_map = {}
        self.edge_count = 0
        self.node_count = 0
        self.no_repeat = no_repeat
        self.memory_map = {}

    def assign_node_weight(self, graph, node):
        target_node_condition = self.path[0]
        if graph.nodes[node]["label"] == target_node_condition["type"]:
            flag = self.check_condition(graph, target_node_condition, node)
            if flag:
                return 10
        return 0.1

    def check_condition(self, graph, conditions, node):
        flag = True
        for attr, value in conditions.items():
            if attr in graph.nodes[node] and graph.nodes[node][attr] != value:
                flag = False
                break
        return flag

    def assign_path_weight(self, graph, neighbor):
        # path hypo
        assert len(self.path) > 0, f"This sampler only support path conditions."

        # only check forward path pattern
        if self.memory_map[self.index][-1] == 10:
            consecutive_count = 0
            for i in range(len(self.memory_map[self.index]) - 1, -1, -1):
                if self.memory_map[self.index][i] == 10:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count == self.path_length:
                consecutive_count = 0

            target_node_condition = self.path[consecutive_count]

        else:
            target_node_condition = self.path[0]

        if graph.nodes[neighbor]["label"] == target_node_condition["type"]:
            flag = self.check_condition(graph, target_node_condition, neighbor)
            if flag:
                return 10
        return 0.1

    def _check_map_weight(self, graph, label=None, current_node=None, neighbor=None):
        if current_node in self.good_node_map:
            weight = self.good_node_map[current_node]
        else:
            if label == "node":
                weight = self.assign_node_weight(graph, current_node)
            elif label == "edge":
                assert (
                    neighbor is not None
                ), f"v must be provided for assigning edge weights."
                weight = self.assign_edge_weight(graph, current_node, neighbor)
            elif label == "path":
                assert (
                    neighbor is not None
                ), f"v must be provided for assigning path weights."
                weight = self.assign_path_weight(graph, neighbor)
        return weight

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        if self.index is not None:
            # new_weight = self._check_map_weight(graph, self._seeds[self.index], "node")
            # self._seed_weights[self.index] = new_weight
            # self.update_memory(self.new_weight)
            pass

        else:
            self._seed_weights = []
            for i, val in enumerate(self._seeds):
                weight = self.assign_node_weight(graph, val)
                self._seed_weights.append(weight)
                self.memory_map[i] = [weight]

        weight_sum = np.sum(self._seed_weights)
        self._norm_seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]

    def _create_initial_seed_set(self, graph):
        """
        Choosing initial nodes.
        """
        nodes = self.backend.get_nodes(graph)
        self._seeds = random.sample(nodes, self.number_of_seeds)

    def update_memory(self, weight):
        if len(self.memory_map[self.index]) == self.path_length:
            self.memory_map[self.index].append(weight)
            self.memory_map[self.index] = self.memory_map[self.index][1:]
            # assert len(self.memory_map[self.index]) == self.path_length
        elif len(self.memory_map[self.index]) < self.path_length:
            self.memory_map[self.index].append(weight)
        else:
            raise Exception(
                "memory_map can only memorize the previous len(path_length) weights."
            )

    def _do_update(self, graph):
        # randomly pick one seed
        sample = np.random.choice(
            self._seeds, 1, replace=False, p=self._norm_seed_weights
        )[0]
        self.index = self._seeds.index(sample)
        # randomly pick one neighbor
        neighbors = self.backend.get_neighbors(graph, sample)
        neighbor_weight = [
            self._check_map_weight(graph, "path", current_node=sample, neighbor=i)
            for i in neighbors
        ]

        neighbor_weight_sum = np.sum(neighbor_weight)
        self._norm_neighbor_weights = [
            float(weight) / neighbor_weight_sum for weight in neighbor_weight
        ]

        new_seed = np.random.choice(
            neighbors, 1, replace=False, p=self._norm_neighbor_weights
        )[0]

        self.new_weight = neighbor_weight[neighbors.index(new_seed)]
        self.update_memory(self.new_weight)

        if (sample, new_seed) in self._edges:
            self.edge_count += 1
        if new_seed in self._nodes:
            self.node_count += 1
        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[self.index] = new_seed

    def sample(self, graph, args):
        self._nodes = set()
        self._edges = set()
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_seed_set(graph)
        self._reweight(graph)
        while len(self._nodes) < self.number_of_nodes:
            self._do_update(graph)
        # new_graph = graph.edge_subgraph(list(self._edges))
        new_graph = self.backend.get_subgraph(graph, self._nodes)
        print(f"No. of repeat nodes: {self.node_count}.")
        print(f"No. of repeat edges: {self.edge_count}.")
        args.logger.info(f"No. of repeat nodes: {self.node_count}.")
        args.logger.info(f"No. of repeat edges: {self.edge_count}.")
        return new_graph


class newSampler_edge_node(Sampler):
    r"""
    Args:
        number_of_seeds (int): Number of seed nodes. Default is 50.
        number_of_nodes (int): Number of nodes to sample. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(
        self,
        condition: list,
        no_repeat: int = 50,
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

    def assign_node_weight(self, graph, node):
        for condition in self.path:
            flag = True
            if graph.nodes[node]["label"] == condition["type"]:
                for attr, v in condition["attribute"].items():
                    if attr in graph.nodes[node] and graph.nodes[node][attr] != v:
                        flag = False
                        break
            else:
                flag = False
            if flag:
                return 3
        return 0.1

    def check_condition(self, graph, conditions, node):
        flag = True
        for attr, value in conditions.items():
            if attr in graph.nodes[node] and graph.nodes[node][attr] != value:
                flag = False
                break
        return flag

    def assign_edge_weight(self, graph, u, v):
        # node or edge hypo
        if len(self.path) <= 2:
            u_node_type = self.path[0]["type"]

            # path length == 1 => node hypo
            if len(self.path) > 1:
                v_node_type = self.path[1]["type"]
            else:
                v_node_type = None
                if graph.nodes[u]["label"] == graph.nodes[v]["label"] == u_node_type:
                    flag = self.check_condition(graph, self.path[0]["attribute"], u)
                    if flag and self.check_condition(
                        graph, self.path[0]["attribute"], v
                    ):
                        return 30
                    else:
                        return 20
                elif graph.nodes[u]["label"] == u_node_type:
                    flag = self.check_condition(graph, self.path[0]["attribute"], u)
                    if flag:
                        return 20
                    else:
                        return 1
                elif graph.nodes[v]["label"] == u_node_type:
                    flag = self.check_condition(graph, self.path[0]["attribute"], v)
                    if flag:
                        return 20
                    else:
                        return 1
                else:
                    return 1

            # edge hypo
            if v_node_type is not None:
                if graph.nodes[u]["label"] == u_node_type:
                    flag = self.check_condition(graph, self.path[0]["attribute"], u)
                    if flag:
                        if graph.nodes[v][
                            "label"
                        ] == v_node_type and self.check_condition(
                            graph, self.path[1]["attribute"], v
                        ):
                            return 30
                        else:
                            return 20
                    else:
                        return 1

                elif graph.nodes[v]["label"] == u_node_type:
                    flag = self.check_condition(graph, self.path[0]["attribute"], v)
                    if flag:
                        if graph.nodes[u][
                            "label"
                        ] == v_node_type and self.check_condition(
                            graph, self.path[1]["attribute"], u
                        ):
                            return 30
                        else:
                            return 20
                    else:
                        return 1
                else:
                    return 1
        else:
            raise Exception("Sorry we don't support path hypothesis for now!")

    def _check_map_weight(self, graph, u, label, v=None):
        if u in self.good_node_map:
            weight = self.good_node_map[u]
        else:
            if label == "node":
                weight = self.assign_node_weight(graph, u)
            elif label == "edge":
                assert v is not None, f"v must be provided for assigning edge weights."
                weight = self.assign_edge_weight(graph, u, v)
        return weight

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        if self.index is not None:
            self._seed_weights[self.index] = self._check_map_weight(
                graph, self._seeds[self.index], "node"
            )

        else:
            self._seed_weights = []
            for i in self._seeds:
                weight = self.assign_node_weight(graph, i)
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
        self._seeds = random.sample(nodes, self.number_of_seeds)

    def _do_update(self, graph):
        # randomly pick one seed
        sample = np.random.choice(
            self._seeds, 1, replace=False, p=self._norm_seed_weights
        )[0]
        self.index = self._seeds.index(sample)
        # randomly pick one neighbor
        neighbors = self.backend.get_neighbors(graph, sample)
        neighbor_weight = [
            self._check_map_weight(graph, sample, "edge", i) for i in neighbors
        ]

        neighbor_weight_sum = np.sum(neighbor_weight)
        self._norm_neighbor_weights = [
            float(weight) / neighbor_weight_sum for weight in neighbor_weight
        ]

        new_seed = np.random.choice(
            neighbors, 1, replace=False, p=self._norm_neighbor_weights
        )[0]

        if (sample, new_seed) in self._edges:
            self.edge_count += 1
        if new_seed in self._nodes:
            self.node_count += 1
        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[self.index] = new_seed

    def sample(self, graph, args):
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
        args.logger.info(f"No. of repeat nodes: {self.node_count}.")
        args.logger.info(f"No. of repeat edges: {self.edge_count}.")
        return new_graph


# class newSampler_Jan9(Sampler):
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
#         no_repeat: str,
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
#         self.good_node_map = {}
#         self.edge_count = 0
#         self.node_count = 0
#         self.no_repeat = no_repeat
#
#     def assign_weight(self, graph, node):
#         for index, condition in enumerate(self.path):
#             score = 0.1
#             flag = True
#             if graph.nodes[node]["label"] == condition["type"]:
#                 score += 1
#                 for attr, v in condition["attribute"].items():
#                     if attr in graph.nodes[node] and graph.nodes[node][attr] != v:
#                         score = 0.1
#                         flag = False
#                         break
#                     else:
#                         score += 1
#             else:
#                 flag = False
#             if flag:
#                 self.good_node_map[node] = score
#                 return score
#         self.good_node_map[node] = score
#         return score
#
#     def _reweight(self, graph):
#         """
#         Create new seed weights.
#         """
#         if self.index is not None:
#             self._seed_weights[self.index] = self._check_map_weight(
#                 graph, self._seeds[self.index]
#             )
#
#         else:
#             self._seed_weights = []
#             for i in self._seeds:
#                 weight = self.assign_weight(graph, i)
#                 #### do nothing
#                 self._seed_weights.append(weight)
#
#         weight_sum = np.sum(self._seed_weights)
#         self._norm_seed_weights = [
#             float(weight) / weight_sum for weight in self._seed_weights
#         ]
#
#     def _create_initial_seed_set(self, graph):
#         """
#         Choosing initial nodes.
#         """
#         nodes = self.backend.get_nodes(graph)
#         ### do nothing
#         self._seeds = random.sample(nodes, self.number_of_seeds)
#
#     def _check_map_weight(self, graph, node):
#         if node in self.good_node_map:
#             weight = self.good_node_map[node]
#         else:
#             weight = self.assign_weight(graph, node)
#         return weight
#
#     def _choose_new_seed(self, graph, sample, neighbors):
#         if self.no_repeat == "edge":
#             return self._choose_seed_with_edge_check(graph, sample, neighbors)
#         elif self.no_repeat == "node":
#             return self._choose_seed_without_node_repeats(graph, sample, neighbors)
#
#     def _choose_seed_with_edge_check(self, graph, sample, neighbors):
#         neighbor_weight = [
#             0.01 if (sample, i) in self._edges else self._check_map_weight(graph, i)
#             for i in neighbors
#         ]
#         return self._calculate_new_seed(graph, neighbors, neighbor_weight)
#
#     def _choose_seed_without_node_repeats(self, graph, sample, neighbors):
#         unvisited_neighbors = set(neighbors) - self._nodes
#         unvisited_neighbors_list = (
#             list(unvisited_neighbors) if unvisited_neighbors else neighbors
#         )
#         neighbor_weight = [
#             self._check_map_weight(graph, i) for i in unvisited_neighbors_list
#         ]
#         return self._calculate_new_seed(
#             graph, unvisited_neighbors_list, neighbor_weight
#         )
#
#     def _calculate_new_seed(self, graph, neighbor_list, neighbor_weight):
#         weight_sum = np.sum(neighbor_weight)
#         norm_seed_weights = [float(weight) / weight_sum for weight in neighbor_weight]
#         return int(
#             np.random.choice(neighbor_list, 1, replace=False, p=norm_seed_weights)[0]
#         )
#
#     def _do_update(self, graph):
#         """
#         Choose new seed node.
#         """
#         sample = int(
#             np.random.choice(self._seeds, 1, replace=False, p=self._norm_seed_weights)[
#                 0
#             ]
#         )
#         self.index = int(self._seeds.index(sample))
#         neighbors = self.backend.get_neighbors(graph, sample)
#
#         new_seed = self._choose_new_seed(graph, sample, neighbors)
#
#         if (sample, new_seed) in self._edges:
#             self.edge_count += 1
#         if new_seed in self._nodes:
#             self.node_count += 1
#         self._edges.add((sample, new_seed))
#         self._nodes.add(sample)
#         self._nodes.add(new_seed)
#         self._seeds[self.index] = new_seed
#
#     def sample(self, graph, args):
#         self._nodes = set()
#         self._edges = set()
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         self._create_initial_seed_set(graph)
#         while len(self._nodes) < self.number_of_nodes:
#             self._reweight(graph)
#             self._do_update(graph)
#         new_graph = graph.edge_subgraph(list(self._edges))
#         print(f"No. of repeat nodes: {self.node_count}.")
#         print(f"No. of repeat edges: {self.edge_count}.")
#         args.logger.info(f"No. of repeat nodes: {self.node_count}.")
#         args.logger.info(f"No. of repeat edges: {self.edge_count}.")
#         return new_graph


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
