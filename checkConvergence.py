# Step 1: Import necessary packages
import argparse
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from littleballoffur.sampler import Sampler

from extraction import getEdges, getNodes, getPaths
from main import getGroundTruth, prepare_dataset
from utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default="2022", help="random seed.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp",
        choices=["citation", "yelp", "movielens"],
        help="choose dataset from DBLP, yelp, or movielens.",
    )
    parser.add_argument(
        "--file_num",
        type=str,
        default="output",
        help="to name log and result files for multi runs.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="Opt_PHASE",
        help="sampling method.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="number of samples to draw from the input graph.",
    )

    ########## parameters for hypothesis ##########
    parser.add_argument(
        "--H0",
        type=str,
        default="The rating difference on path [business in LA - medium popularity user - business in AB] is greater than 0.5",
        help="The null hypothesis.",
    )
    parser.add_argument(
        "--HTtype",
        type=str,
        default="one-sample",
        choices=["one-sample"],
        help="We support one-sample hypothesis testing.",
    )
    parser.add_argument(
        "--hypothesis_pattern",
        type=dict,
        default={
            "3-1-1": {
                "type": "path",  # "edge" | "node" | "path" | "subgraph"
                "path": [
                    {"type": "business", "attribute": {"state": "LA"}},
                    {"type": "user", "attribute": {"popularity": "medium"}},
                    {"type": "business", "attribute": {"state": "AB"}},
                ],
                "target": {
                    "edge": "stars",  # or 'node': {'index': 2, 'attribute': 'age'}
                },
                "test": {
                    "comparison": ">",  # "!=" | "==" | ">" | "<"
                    "c": 0.5,  # constant value in hypothesis
                    "agg": "mean",  # aggregation function
                },
            }
        },
        help="the hypothesis pattern to test on. "
        "Format: {'name': {'type': 'edge'|'node'|'path'|'subgraph', 'structure': {...}, 'target': {'edge': 'attr'|'node': {...}}, 'test': {'comparison': '>', 'c': 0.5, 'agg': 'mean'}}}. "
        "For path: {'name': {'type': 'path', 'path': [...], 'target': {'edge': 'attr'}, 'test': {...}}}; "
        "For subgraph: {'name': {'type': 'subgraph', 'subgraph': {'nodes': [...], 'edges': [...]}, 'target': {'edge': 'attr'}, 'test': {...}}}",
    )

    ### our sampler hyper-parameter
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.95,
        help="significance level.",
    )

    args = parser.parse_args()

    # Extract hypothesis parameters from hypothesis_pattern if available
    if args.hypothesis_pattern and len(args.hypothesis_pattern) > 0:
        pattern_key = list(args.hypothesis_pattern.keys())[0]
        pattern = args.hypothesis_pattern[pattern_key]

        # Extract type (hypo)
        if "type" in pattern:
            type_mapping = {"edge": 1, "node": 2, "path": 3, "subgraph": 4}
            if pattern["type"] in type_mapping:
                args.hypo = type_mapping[pattern["type"]]
            else:
                raise ValueError(f"Unknown hypothesis type: {pattern['type']}")
        else:
            raise ValueError("hypothesis_pattern must contain a 'type' key.")

        # Extract test parameters
        if "test" in pattern:
            test_config = pattern["test"]
            if "comparison" in test_config:
                args.comparison = test_config["comparison"]
            else:
                raise ValueError(
                    "hypothesis_pattern['test'] must contain a 'comparison' key."
                )
            if "c" in test_config:
                args.c = test_config["c"]
            else:
                raise ValueError("hypothesis_pattern['test'] must contain a 'c' key.")
            if "agg" in test_config:
                args.agg = test_config["agg"]
            else:
                raise ValueError(
                    "hypothesis_pattern['test'] must contain an 'agg' key."
                )
        else:
            raise ValueError("hypothesis_pattern must contain a 'test' key.")

    return args


class DegreeBasedSampler(Sampler):
    r"""An implementation of degree based sampling. Nodes are sampled proportional
    to the degree centrality of nodes. `"For details about the algorithm see
    this paper." <https://arxiv.org/abs/cs/0103016>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self.sampled_nodes = set()
        self.degrees = []
        self.nodes = []

    def _create_initial_node_set(self, graph, checkpoint: int) -> List[int]:
        """
        Choosing initial nodes.
        """
        if not self.nodes:  # Initialize once
            self.nodes = list(range(self.backend.get_number_of_nodes(graph)))
            self.degrees = np.array(
                [float(self.backend.get_degree(graph, node)) for node in self.nodes]
            )

        current_sample_size = len(self.sampled_nodes)
        additional_nodes_needed = checkpoint - current_sample_size
        if additional_nodes_needed <= 0:
            return list(self.sampled_nodes)

        remaining_nodes = list(set(self.nodes) - self.sampled_nodes)
        remaining_degrees = self.degrees[remaining_nodes]
        degree_sum = sum(remaining_degrees)
        probabilities = remaining_degrees / degree_sum
        new_sampled_nodes = np.random.choice(
            remaining_nodes,
            size=additional_nodes_needed,
            replace=False,
            p=probabilities,
        )
        self.sampled_nodes.update(new_sampled_nodes)
        return list(self.sampled_nodes)

    def sample(self, graph, checkpoint: int):
        """
        Sampling nodes proportional to the degree.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        sampled_nodes = self._create_initial_node_set(graph, checkpoint)
        new_graph = self.backend.get_subgraph(graph, sampled_nodes)
        return new_graph


class RandomNodeSampler(Sampler):
    r"""An implementation of random node sampling. Nodes are sampled with uniform
    probability. `"For details about the algorithm see this paper." <https://www.pnas.org/content/102/12/4221>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self.sampled_nodes = set()

    def _create_initial_node_set(self, graph, checkpoint) -> List[int]:
        """
        Choosing initial nodes.
        """
        total_nodes = self.backend.get_nodes(graph)
        remaining_nodes = list(set(total_nodes) - self.sampled_nodes)
        nodes_needed = checkpoint - len(
            self.sampled_nodes
        )  # Calculate how many more nodes are needed

        if nodes_needed > 0:
            new_sampled_nodes = random.sample(remaining_nodes, nodes_needed)
            self.sampled_nodes.update(new_sampled_nodes)
        return list(self.sampled_nodes)

    def sample(self, graph, checkpoint: int):
        """
        Sampling nodes randomly.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        sampled_nodes = self._create_initial_node_set(graph, checkpoint)
        new_graph = self.backend.get_subgraph(graph, sampled_nodes)
        return new_graph


class RandomWalkSampler(Sampler):
    r"""An implementation of node sampling by random walks. A simple random walker
    which creates an induced subgraph by walking around. `"For details about the
    algorithm see this paper." <https://ieeexplore.ieee.org/document/5462078>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self._sampled_nodes = set()
        self._current_node = None

    def _create_initial_node_set(self, graph, start_node):
        """
        Choosing an initial node.
        """
        if self._current_node is None:
            if (
                start_node is not None
                and 0 <= start_node < self.backend.get_number_of_nodes(graph)
            ):
                self._current_node = start_node
            else:
                self._current_node = random.choice(
                    range(self.backend.get_number_of_nodes(graph))
                )

        self._sampled_nodes.add(self._current_node)

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        self._current_node = self.backend.get_random_neighbor(graph, self._current_node)
        self._sampled_nodes.add(self._current_node)

    def sample(self, graph, checkpoint: int, start_node: int = None):
        """
        Sampling nodes with a single random walk.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_node** *(int, optional)* - The start node.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        # Initialize only if we're starting a new walk
        if not self._sampled_nodes:
            self._create_initial_node_set(graph, start_node)

        while len(self._sampled_nodes) < checkpoint:
            self._do_a_step(graph)
            if len(self._sampled_nodes) >= checkpoint:
                break

        new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
        return new_graph


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
        super().__init__()
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

        self._nodes = set()
        self._edges = set()
        self._seeds = []

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
        assert (
            neighbor is not None
        ), f"neighbor must be provided for assigning path weights."

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
        """
        randomly select a seed node and pick a subset of neighbors for weight assigning process before selecting one
        """
        # randomly pick one seed
        sample = np.random.choice(
            self._seeds, 1, replace=False, p=self._norm_seed_weights
        )[0]
        self.index = self._seeds.index(sample)

        # remove visited nodes from neighbor nodes
        not_visited_nodes = set(graph.neighbors(sample)) - self._nodes
        neighbors = list(not_visited_nodes)

        # pick a neighboring node from a subset of neighbors
        num_neighbor = 30
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

    def sample(self, graph, args, checkpoint):
        self.number_of_nodes = checkpoint

        if not self._seeds:
            self._deploy_backend(graph)
            self._check_number_of_nodes(graph)
            self._create_initial_seed_set(graph)
            self._reweight(graph)

        # self._deploy_backend(graph)
        # self._check_number_of_nodes(graph)
        # self._create_initial_seed_set(graph)
        # self._reweight(graph)
        while len(self._nodes) < checkpoint:
            self._do_update(graph)
            if len(self._nodes) >= checkpoint:
                break

        new_graph = self.backend.get_subgraph(graph, self._nodes)
        self.backend.get_subgraph(graph, self._nodes)
        print(f"No. of repeat nodes: {self.node_count}.")
        print(f"No. of repeat edges: {self.edge_count}.")
        args.logger.info(f"No. of repeat nodes: {self.node_count}.")
        args.logger.info(f"No. of repeat edges: {self.edge_count}.")
        return new_graph


def sample_and_calculate_mean_citations(graph, sample_sizes, sampler_list):
    mean_citations = defaultdict(list)
    for sampler_name in sampler_list:
        args.logger.info(f">>> start sampling method {sampler_name}")
        print(f">>> start sampling method {sampler_name}")
        current_size = 0

        if sampler_name == "DBS":
            sampler = DegreeBasedSampler(max(sample_sizes), seed=int(args.seed))
        elif sampler_name == "Opt_PHASE":
            pattern_key = list(args.hypothesis_pattern.keys())[0]
            pattern = args.hypothesis_pattern[pattern_key]
            # Extract path from hypothesis_pattern (for path type)
            if pattern["type"] == "path" and "path" in pattern:
                path_condition = pattern["path"]
            else:
                raise ValueError(
                    f"Opt_PHASE requires path type hypothesis with 'path' key in hypothesis_pattern."
                )
            sampler = Opt_PHASE(
                path_condition,
                number_of_nodes=max(sample_sizes),
                seed=int(args.seed),
            )
        elif sampler_name == "SRW":
            sampler = RandomWalkSampler(max(sample_sizes), seed=int(args.seed))
        elif sampler_name == "RNS":
            sampler = RandomNodeSampler(max(sample_sizes), seed=int(args.seed))

        for size in sample_sizes:
            args.logger.info(f">>> start size {size}")
            print(f"start size {size}")

            if size > current_size:
                if sampler_name in ["DBS", "SRW", "RNS"]:
                    sampled_subgraph = sampler.sample(graph, size)
                elif sampler_name == "Opt_PHASE":
                    sampled_subgraph = sampler.sample(graph, args, size)
                current_size = size

            result_list = time_sampling_extraction(
                args,
                sampled_subgraph,
                30,
            )
            pattern_key = str(list(args.hypothesis_pattern.keys())[0])
            mean_citations[sampler_name].append(result_list[pattern_key])
            args.logger.info(f"mean for size {size} is {result_list[pattern_key]}")
            print(f"mean for size {size} is {result_list[pattern_key]}")

        with open(
            os.path.join(
                args.log_folderPath, f"convergence_{args.dataset}_Opt_PHASE_0429.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(mean_citations, f)

    return mean_citations


def time_sampling_extraction(args, new_graph, num_sample):
    result_list = new_graph_hypo_result(args, new_graph)

    return result_list


def new_graph_hypo_result(args, new_graph):
    # time_rating_extraction_start = time.time()

    dataset_functions = {
        "movielens": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "citation": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "yelp": {"edge": getEdges, "node": getNodes, "path": getPaths},
    }

    if args.dataset in dataset_functions:
        dataset_info = dataset_functions[args.dataset]

        if args.hypo == 1:
            total_result = dataset_info["edge"](args, new_graph)
        elif args.hypo == 2:
            # Extract dimension from hypothesis_pattern if needed
            pattern_key = list(args.hypothesis_pattern.keys())[0]
            pattern = args.hypothesis_pattern[pattern_key]
            if "dimension" in pattern:
                args.dimension = pattern["dimension"]
            elif hasattr(args, "dimension") and args.dimension is not None:
                pass  # Use existing dimension
            else:
                raise ValueError(
                    "dimension is required for node hypothesis but not found in hypothesis_pattern."
                )
            total_result = dataset_info["node"](
                args, new_graph, dimension=args.dimension
            )
        elif args.hypo == 3:
            args.total_valid = 0
            args.total_minus_reverse = 0
            total_result = dataset_info["path"](args, new_graph)

        else:
            raise Exception(
                f"Sorry, we don't support hypothesis {args.hypo} for {args.dataset}."
            )
    else:
        raise Exception(f"Sorry, we don't support the dataset {args.dataset}.")

    if str(list(args.hypothesis_pattern.keys())[0]) not in total_result:
        total_result[list(args.hypothesis_pattern.keys())[0]] = []

    result = {}

    for attribute, v in total_result.items():
        # args.valid_edges.append(len(v))
        if len(attribute.split("+")) == 1:
            if args.agg == "mean":
                if len(v) != 0:
                    print("not zero now!!!")
                    args.logger.info("not zero now!!!")
                    result[attribute] = sum(v) / len(v)
                #     hypo_result = HypothesisTesting(args, v, 1)
                #     result[attribute] = hypo_result
                else:
                    result[attribute] = -1

            else:
                raise Exception(f"Sorry, we don't support {args.agg}.")
        else:
            result[attribute] = v
    return result


def logger(args):
    args.log_folderPath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "convergence"
    )

    if not os.path.exists(args.log_folderPath):
        os.makedirs(args.log_folderPath)

    args.log_filepath = os.path.join(
        args.log_folderPath, f"convergence_{args.dataset}_Opt_PHASE_0429.log"
    )

    logging.basicConfig(
        filename=args.log_filepath, format="%(asctime)s %(message)s", filemode="w"
    )

    # Creating an object
    args.logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    args.logger.setLevel(logging.INFO)


args = parse_args()
setup_seed(args)
logger(args)
G = prepare_dataset(args)

# Calculate ground truth value for convergence plot
if args.dataset in ["movielens", "citation", "yelp"]:
    ground_truth_dict = getGroundTruth(args, G)
    pattern_key = str(list(args.hypothesis_pattern.keys())[0])
    if args.HTtype == "one-sample":
        args.ground_truth = ground_truth_dict[pattern_key]
    # ground_truth_value is set in getGroundTruth function
    if not hasattr(args, "ground_truth_value"):
        # Fallback: calculate mean if not set
        attribute_key = str(list(args.hypothesis_pattern.keys())[0])
        if args.hypo == 1:
            dict_result = getEdges(args, G)
        elif args.hypo == 2:
            args.dimension = args.hypothesis_pattern[attribute_key].get(
                "dimension", None
            )
            dict_result = getNodes(args, G, dimension=args.dimension)
        elif args.hypo == 3:
            args.total_valid = 0
            args.total_minus_reverse = 0
            dict_result = getPaths(args, G)
        if attribute_key in dict_result and len(dict_result[attribute_key]) > 0:
            args.ground_truth_value = round(
                sum(dict_result[attribute_key]) / len(dict_result[attribute_key]), 2
            )
        else:
            args.ground_truth_value = 0.0
else:
    args.ground_truth_value = 0.0

sampler_list = ["RNS", "Opt_PHASE", "DBS"]
sample_sizes = np.linspace(
    50000, 1623013, 100, dtype=int
)  # From 10 to 1000 nodes in 20 steps
sample_sizes = sample_sizes[:90]
print(sample_sizes)
args.logger.info(sample_sizes)
print("\n")
args.logger.info("\n")

mean_citations = sample_and_calculate_mean_citations(G, sample_sizes, sampler_list)


# Plotting the results
colors = ["#347B98", "#D63447", "#FDAC53", "#9DC209", "#285943", "#774936", "#4A5899"]
plt.figure(figsize=(10, 6))
x = np.around((sample_sizes / 1623013) * 100, decimals=2)
for i, sampler_name in enumerate(sampler_list):
    plt.plot(
        x, mean_citations[sampler_name], label=sampler_name, marker="o", color=colors[i]
    )

plt.axhline(
    y=args.ground_truth_value, color="r", linestyle="--", label="Avg Citation of Graph"
)
plt.xlabel("Percentage of sampled nodes (%)")
plt.ylabel("Aggregated value")
plt.title(f"Convergence of aggregated value - {args.dataset}")
plt.legend()

# plt.savefig(os.path.join(args.log_folderPath, f"convergence_{args.dataset}_Opt_PHASE_0429.png"))
plt.show()
