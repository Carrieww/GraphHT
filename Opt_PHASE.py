from littleballoffur.sampler import Sampler
import numpy as np
import random


class Opt_PHASE(Sampler):
    """
    Args:
        number_of_seeds (int): Number of seed nodes. Default is 50.
        number_of_nodes (int): Number of nodes to sample. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(
        self,
        condition: list,
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
        self.edge_count = 0
        self.node_count = 0
        self.memory_map = {}
        self.w1 = 10
        self.w2 = 0.1

    def _assign_node_weight(self, graph, node):
        target_node_condition = self.path[0]
        if graph.nodes[node]["label"] == target_node_condition["type"]:
            flag = self._check_condition(
                graph, target_node_condition["attribute"], node
            )
            if flag:
                return self.w1
        return self.w2

    def _check_condition(self, graph, conditions, node):
        """
        Checking whether the node satisfy the condition on path.
        """
        flag = True
        for attr, value in conditions.items():
            if attr not in graph.nodes[node]:
                flag = False
                break

            # vague match for citation 3-1-1
            if graph.nodes[node][attr] != value and (
                value not in graph.nodes[node][attr]
                if isinstance(graph.nodes[node][attr], str)
                else True
            ):
                flag = False
                break
        return flag

    def _assign_neighbor_weight(self, graph, neighbor=None):
        """
        assign weight to neighboring nodes.
        """
        assert neighbor is not None, f"v must be provided for assigning path weights."

        # path hypo
        assert len(self.path) > 0, f"Path condition must not be empty."

        # obtain the target node condition on the path
        if self.memory_map[self.index][-1] == self.w1:
            consecutive_count = 0
            for i in range(len(self.memory_map[self.index]) - 1, -1, -1):
                if self.memory_map[self.index][i] == self.w1:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count == self.path_length:
                consecutive_count = 0

            target_node_condition = self.path[consecutive_count]

        else:
            target_node_condition = self.path[0]

        # check whether the neighbor satisfy the target node condition
        if graph.nodes[neighbor]["label"] == target_node_condition["type"]:
            flag = self._check_condition(
                graph, target_node_condition["attribute"], neighbor
            )
            if flag:
                return self.w1
        return self.w2

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        if self.index is not None:
            pass

        else:
            self._seed_weights = []
            for i, val in enumerate(self._seeds):
                weight = self._assign_node_weight(graph, val)
                self._seed_weights.append(weight)
                self.memory_map[i] = [weight]

        weight_sum = np.sum(self._seed_weights)
        self._norm_seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]

    def _create_initial_seed_set(self, graph):
        """
        Choosing initial seeds randomly.
        """
        nodes = self.backend.get_nodes(graph)
        self._seeds = random.sample(nodes, self.number_of_seeds)

    def _update_memory(self, weight):
        """
        Updating the memory_map.
        """
        if len(self.memory_map[self.index]) == self.path_length:
            self.memory_map[self.index].append(weight)
            self.memory_map[self.index] = self.memory_map[self.index][1:]
        elif len(self.memory_map[self.index]) < self.path_length:
            self.memory_map[self.index].append(weight)
        else:
            raise Exception(
                "memory_map can only memorize the previous len(path_length) weights."
            )

    def _do_update(self, graph):
        # randomly pick one seed
        num_neighbor = 30
        sample = np.random.choice(
            self._seeds, 1, replace=False, p=self._norm_seed_weights
        )[0]
        self.index = self._seeds.index(sample)

        # remove visited nodes from neighbor nodes
        not_visited_nodes = set(graph.neighbors(sample)) - self._nodes
        neighbors = list(not_visited_nodes)

        # pick a neighboring node
        if len(neighbors) == 0:
            new_seed = random.choice(self.backend.get_neighbors(graph, sample))
            self.new_weight = self._assign_neighbor_weight(graph, neighbor=new_seed)
        else:
            neighbors = random.sample(neighbors, k=min(num_neighbor, len(neighbors)))
            neighbor_weight = [
                self._assign_neighbor_weight(graph, neighbor=i) for i in neighbors
            ]

            neighbor_weight_sum = np.sum(neighbor_weight)
            self._norm_neighbor_weights = [
                float(weight) / neighbor_weight_sum for weight in neighbor_weight
            ]

            new_seed = np.random.choice(
                neighbors, 1, replace=False, p=self._norm_neighbor_weights
            )[0]

            self.new_weight = neighbor_weight[neighbors.index(new_seed)]

        # memorize the weight of all necessary states (current and previous)
        self._update_memory(self.new_weight)

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
        new_graph = self.backend.get_subgraph(graph, self._nodes)
        self.backend.get_subgraph(graph, self._nodes)
        print(f"No. of repeat nodes: {self.node_count}.")
        print(f"No. of repeat edges: {self.edge_count}.")
        args.logger.info(f"No. of repeat nodes: {self.node_count}.")
        args.logger.info(f"No. of repeat edges: {self.edge_count}.")
        return new_graph
