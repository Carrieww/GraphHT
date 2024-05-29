import time
import random
import networkx as nx
import numpy as np

from extraction import new_graph_hypo_result
from PHASE import PHASE
from Opt_PHASE import Opt_PHASE
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
        "PHASE": PHASE,
        "Opt_PHASE": Opt_PHASE,
    }

    if sampler_type not in sampler_mapping:
        raise ValueError("Invalid sampler type.")

    sampler_class = sampler_mapping[sampler_type]

    args.CI = {"lower": [], "upper": []}
    args.p_value = []
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
        elif sampler_type == "PHASE":
            model = PHASE(
                args.attribute[str(list(args.attribute.keys())[0])]["path"],
                number_of_nodes=args.ratio,
                seed=seed,
            )
        elif sampler_type == "Opt_PHASE":
            model = Opt_PHASE(
                args.attribute[str(list(args.attribute.keys())[0])]["path"],
                number_of_nodes=args.ratio,
                seed=seed,
            )
        else:
            model = sampler_class(number_of_nodes=args.ratio, seed=seed)

        if sampler_type == "PHASE" or sampler_type == "Opt_PHASE":
            new_graph = model.sample(graph, args)
        else:
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
        # overall_time_spent = round(time.time() - args.overall_time, 2)
        # if overall_time_spent > 18000:
        #     args.logger.error(
        #         f"The overall time spend for sampling and hypothsis testing is {overall_time_spent} > 18000 (5 hours). So we terminate it."
        #     )
        #     raise Exception(
        #         f"The overall time spend for sampling and hypothsis testing is {overall_time_spent} > 18000 (5 hours). So we terminate it."
        #     )

    if len(args.CI["lower"]) > 0:
        args.time_result[args.ratio].append(
            round(sum(args.CI["lower"]) / len(args.CI["lower"]), 2)
        )
    else:
        args.logger.info("no valid lower CI")
        args.time_result[args.ratio].append(-1)

    if len(args.CI["upper"]) > 0:
        args.time_result[args.ratio].append(
            round(sum(args.CI["upper"]) / len(args.CI["upper"]), 2)
        )
    else:
        args.logger.info("no valid upper CI")
        args.time_result[args.ratio].append(-1)

    if len(args.p_value) > 0:
        args.time_result[args.ratio].append(
            round(sum(args.p_value) / len(args.p_value), 2)
        )
    else:
        args.logger.info("no valid p-value")
        args.time_result[args.ratio].append(-1)

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
