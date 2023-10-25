import time

import networkx as nx
import numpy as np

from new_graph_hypo_postprocess import new_graph_hypo_result


def time_sampling_extraction(
    args, new_graph, result_list, time_used_list, time_one_sample_start, num_sample
):
    args.logger.info(
        f"Time for sampling once {args.ratio}: {round(time.time() - time_one_sample_start, 2)}."
    )
    time_used_list["sampling"].append(round(time.time() - time_one_sample_start, 2))

    result_list = new_graph_hypo_result(args, new_graph, result_list, time_used_list)

    return result_list, time_used_list


##############################
######## Exploration #########
##############################
def RNNS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomNodeNeighborSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomNodeNeighborSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def SRW(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomWalkSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomWalkSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def FFS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import ForestFireSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = ForestFireSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
            p=0.4,
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def ShortestPathS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import ShortestPathSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = ShortestPathSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        # if ((time.time() - time_one_sample_start) / 60) > 15:
        #     raise Exception(f"Sampling once takes more than 15 minites so we stop.")
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def MHRWS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import MetropolisHastingsRandomWalkSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = MetropolisHastingsRandomWalkSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
            alpha=0.5,
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def CommunitySES(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import CommunityStructureExpansionSampler

    class CommunityStructureExpansionSampler_new(CommunityStructureExpansionSampler):
        def __init__(self, number_of_nodes: int = 100, seed: int = 42):
            self.number_of_nodes = number_of_nodes
            self.seed = seed
            self.known_expansion = {}
            self._set_seed()

        def _choose_new_node(self, graph):
            """
            Choosing the node with the largest expansion.
            The randomization of the list breaks ties randomly.
            """
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

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = CommunityStructureExpansionSampler_new(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()

        if find_stop:
            return [num_nodes, num_edges], []
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


def NBRW(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import NonBackTrackingRandomWalkSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = NonBackTrackingRandomWalkSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def SBS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import SnowBallSampler

    class SnowBallSampler_new(SnowBallSampler):
        def sample(self, graph, start_node: int = None):
            """
            Sampling a graph with randomized snow ball sampling.

            Arg types:
                * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
                * **start_node** *(int, optional)* - The start node.

            Return types:
                * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
            """
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

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = SnowBallSampler_new(
            number_of_nodes=args.ratio, seed=(int(args.seed) * num_sample), k=200
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def RW_Starter(args, graph, result_list, time_used_list, find_stop=False):
    import random

    from littleballoffur import RandomWalkWithRestartSampler

    class RandomWalkWithRestartSampler_new(RandomWalkWithRestartSampler):
        def _create_initial_node_set(self, graph, start_node):
            """
            Choosing an initial node.
            """
            if start_node is not None:
                if start_node >= 0 and start_node < self.backend.get_number_of_nodes(
                    graph
                ):
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

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()

        model = RandomWalkWithRestartSampler_new(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
            p=0.01,
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def FrontierS(args, graph, result_list, time_used_list, find_stop=False):
    import random

    from littleballoffur import FrontierSampler

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
            """
            Create new seed weights.
            """
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
            """
            Choose new seed node.
            """
            sample = np.random.choice(
                self._seeds, 1, replace=False, p=self._seed_weights
            )[0]
            self.index = self._seeds.index(sample)
            new_seed = random.choice(self.backend.get_neighbors(graph, sample))
            self._edges.add((sample, new_seed))
            self._nodes.add(sample)
            self._nodes.add(new_seed)
            self._seeds[self.index] = new_seed

        def sample(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
            """
            Sampling nodes and edges with a frontier sampler.

            Arg types:
                * **graph** *(NetworkX graph)* - The graph to be sampled from.

            Return types:
                * **new_graph** *(NetworkX graph)* - The graph of sampled nodes.
            """
            self._nodes = set()
            self._edges = set()
            self._deploy_backend(graph)
            self._check_number_of_nodes(graph)
            self._create_initial_seed_set(graph)
            while len(self._nodes) < self.number_of_nodes:
                self._reweight(graph)
                self._do_update(graph)
            new_graph = graph.edge_subgraph(list(self._edges))
            # new_graph = self.backend.graph_from_edgelist(self._edges)
            # new_graph = self.backend.get_subgraph(graph, self._nodes)
            return new_graph

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = FrontierSampler_new(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def CNARW(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import CommonNeighborAwareRandomWalkSampler

    class CommonNeighborAwareRandomWalkSampler_new(
        CommonNeighborAwareRandomWalkSampler
    ):
        def _do_a_step(self, graph):
            """
            Doing a single random walk step.
            """
            self._get_node_scores(graph, self._current_node)
            self._current_node = np.random.choice(
                self._sampler[self._current_node]["neighbors"],
                1,
                replace=False,
                p=self._sampler[self._current_node]["scores"],
            )[0]
            self._sampled_nodes.add(self._current_node)

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = CommonNeighborAwareRandomWalkSampler_new(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


###############################
######## Node Sampler #########
###############################


def RNS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomNodeSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomNodeSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def DBS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import DegreeBasedSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = DegreeBasedSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def PRBS(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import PageRankBasedSampler

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = PageRankBasedSampler(
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


###############################
######## Edge Sampler #########
###############################


def RES(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomEdgeSampler

    class RandomEdgeSampler_new(RandomEdgeSampler):
        def sample(self, graph):
            """
            Sampling edges randomly.

            Arg types:
                * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

            Return types:
                * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
            """
            self._deploy_backend(graph)
            self._check_number_of_edges(graph)
            self._create_initial_edge_set(graph)
            new_graph = graph.edge_subgraph(self._sampled_edges)
            # new_graph = self.backend.graph_from_edgelist(self._sampled_edges)
            return new_graph

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomEdgeSampler_new(
            number_of_edges=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def RNES(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomNodeEdgeSampler

    class RandomNodeEdgeSampler_new(RandomNodeEdgeSampler):
        def sample(self, graph):
            """
            Sampling edges randomly from randomly sampled nodes.

            Arg types:
                * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

            Return types:
                * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
            """
            self._deploy_backend(graph)
            self._check_number_of_edges(graph)
            self._create_initial_edge_set(graph)
            new_graph = graph.edge_subgraph(list(self._sampled_edges))
            # new_graph = self.backend.graph_from_edgelist(self._sampled_edges)
            return new_graph

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomNodeEdgeSampler_new(
            number_of_edges=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


def RES_Induction(args, graph, result_list, time_used_list, find_stop=False):
    from littleballoffur import RandomEdgeSamplerWithInduction

    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = RandomEdgeSamplerWithInduction(
            number_of_edges=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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


from littleballoffur.sampler import Sampler
import random


class newSampler(Sampler):
    r"""
    Args:
        number_of_seeds (int): Number of seed nodes. Default is 10.
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

    def assign_weight(self, graph, node):
        index = len(self.path)
        for condition in self.path:
            flag = True
            if graph.nodes[node]["label"] == condition["type"]:
                for attr, v in condition["attribute"].items():
                    if graph.nodes[node][attr] != v:
                        flag = False
                        break
            else:
                flag = False
            if flag:
                return index
                # return index  # self._seed_weights.append(index)
            else:
                index -= 1
        if index == 0:
            return 1  # self._seed_weights.append(1)

    def check_condition(self, graph, node, condition_ind):
        condition = self.path[condition_ind]
        if graph.nodes[node]["label"] == condition["type"]:
            for attr, v in condition["attribute"].items():
                if graph.nodes[node][attr] != v:
                    return False
        return True

    def _reweight(self, graph):
        """
        Create new seed weights.
        """
        self._seed_weights = []
        for i in self._seeds:
            weight = self.assign_weight(graph, i)
            #### do nothing
            # self._seed_weights.append(weight)

            #### piece wise
            threshold = self.number_of_nodes / 2
            if weight != 0:
                if len(self._nodes) <= threshold:
                    pass
                else:
                    weight = len(self.path) - weight + 1
            else:
                weight = 1
            self._seed_weights.append(weight)

            #### first 1/4 select first node, 1/4 - 3/4 select user nodes, last 1/4 will uniform selection
            # if len(self._nodes) <= self.number_of_nodes / 4:
            #     if weight != len(self.path):
            #         weight = 1
            #     else:
            #         weight = 10
            # elif (
            #     self.number_of_nodes / 4
            #     < len(self._nodes)
            #     < self.number_of_nodes * 3 / 4
            # ):
            #     if weight != len(self.path) - 1:
            #         weight = 1
            #     else:
            #         weight = 10
            # else:
            #     weight = 1
            # self._seed_weights.append(weight)

        weight_sum = np.sum(self._seed_weights)
        # print(self._seed_weights)
        self._seed_weights = [
            float(weight) / weight_sum for weight in self._seed_weights
        ]

    def _create_initial_seed_set(self, graph):
        """
        Choosing initial nodes.
        """
        nodes = self.backend.get_nodes(graph)
        self._seeds = random.sample(nodes, self.number_of_seeds)

    def _do_update(self, graph):
        """
        Choose new seed node.
        """
        # print(self._seeds)
        sample = np.random.choice(self._seeds, 1, replace=False, p=self._seed_weights)[
            0
        ]
        self.index = self._seeds.index(sample)
        #### do nothing
        new_seed = random.choice(self.backend.get_neighbors(graph, sample))

        #### select according to the path
        # weight = self.assign_weight(graph, sample)
        # condition_ind = len(self.path) - weight + 1
        # if condition_ind > len(self.path) - 1:
        #     new_seed = random.choice(self.backend.get_neighbors(graph, sample))
        # else:
        #     found = False
        #     for neighbor in self.backend.get_neighbors(graph, sample):
        #         if self.check_condition(graph, neighbor, condition_ind):
        #             new_seed = neighbor
        #             found = True
        #             break
        #     if not found:
        #         new_seed = random.choice(self.backend.get_neighbors(graph, sample))

        self._edges.add((sample, new_seed))
        self._nodes.add(sample)
        self._nodes.add(new_seed)
        self._seeds[self.index] = new_seed

    def sample(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX graph)* - The graph of sampled nodes.
        """
        self._nodes = set()
        self._edges = set()
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_seed_set(graph)
        while len(self._nodes) < self.number_of_nodes:
            self._reweight(graph)
            self._do_update(graph)
        new_graph = graph.edge_subgraph(list(self._edges))
        # new_graph = self.backend.graph_from_edgelist(self._edges)
        # new_graph = self.backend.get_subgraph(new_graph, self._nodes)
        return new_graph


def OurSampler(args, graph, result_list, time_used_list, find_stop=False):
    for num_sample in range(args.num_samples):
        time_one_sample_start = time.time()
        model = newSampler(
            args.attribute[str(list(args.attribute.keys())[0])]["path"],
            number_of_nodes=args.ratio,
            seed=(int(args.seed) * num_sample),
        )
        new_graph = model.sample(graph)
        num_nodes = new_graph.number_of_nodes()
        num_edges = new_graph.number_of_edges()
        if find_stop:
            return [num_nodes, num_edges], []
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
