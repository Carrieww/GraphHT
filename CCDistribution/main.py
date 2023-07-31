import os
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from config import parse_args
from graphPreprocess import (
    getAuthorList,
    getMovieList,
    getPaperList,
    getRelationList,
    getRelationLists,
    getUserList,
    moviePreprocess,
)
from littleballoffur import (
    BreadthFirstSearchSampler,
    CirculatedNeighborsRandomWalkSampler,
    CommonNeighborAwareRandomWalkSampler,
    CommunityStructureExpansionSampler,
    DegreeBasedSampler,
    DepthFirstSearchSampler,
    DiffusionTreeSampler,
    ForestFireSampler,
    FrontierSampler,
    GraphReader,
    MetropolisHastingsRandomWalkSampler,
    NonBackTrackingRandomWalkSampler,
    PageRankBasedSampler,
    RandomEdgeSampler,
    RandomEdgeSamplerWithInduction,
    RandomNodeEdgeSampler,
    RandomNodeNeighborSampler,
    RandomNodeSampler,
    RandomWalkSampler,
    RandomWalkWithJumpSampler,
    RandomWalkWithRestartSampler,
    ShortestPathSampler,
    SnowBallSampler,
    SpikyBallSampler,
)
from scipy import stats
from utils import clean, logger, setup_device, setup_seed

# only undirected and unweighted graph


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    logger(args)
    args.checkpoint_name = args.property + args.checkpoint_name

    ## ratio_list
    if args.ratio_list is None:
        if args.ratio_num is None:
            raise Exception("One of ratio_list and ratio_num must be provided.")
        else:
            args.ratio_list = np.linspace(0.1, 0.9, args.ratio_num)
    else:
        pass

    # get the original graph
    if args.dataset == "facebook":
        reader = GraphReader(args.dataset)
        graph = reader.get_graph()
    elif args.dataset == "ca_GrQc":
        fh = open("/Users/wangyun/Documents/GitHub/GraphHT/ca-GrQc.txt", "rb")
        G = nx.read_edgelist(fh, nodetype=int, create_using=nx.Graph())
        fh.close()
        graph = nx.relabel.convert_node_labels_to_integers(
            G, first_label=0, ordering="default"
        )
    elif args.dataset == "lastfm_asia":
        G = nx.read_edgelist(
            "/Users/wangyun/Documents/GitHub/GraphHT/lastfm_asia_edges.csv",
            nodetype=int,
            create_using=nx.Graph(),
            delimiter=",",
        )
        graph = nx.relabel.convert_node_labels_to_integers(
            G, first_label=0, ordering="default"
        )
    elif args.dataset == "movielens":
        df_movies = pd.read_csv(
            "/Users/wangyun/Documents/GitHub/GraphHT/ml-latest-small/movies.csv"
        )
        df_ratings = pd.read_csv(
            "/Users/wangyun/Documents/GitHub/GraphHT/ml-latest-small/ratings.csv"
        )
        # df_tags = pd.read_csv(
        #     "/Users/wangyun/Documents/GitHub/GraphHT/ml-latest-small/tags.csv"
        # )
        df_movies = moviePreprocess(df_movies)
        movie_list = getMovieList(args, df_movies)
        graph = nx.Graph()
        graph.add_nodes_from(movie_list)
        user_list = getUserList(df_movies, df_ratings)
        graph.add_nodes_from(user_list)
        relation_list = getRelationList(args, graph, df_movies, df_ratings)

        # initiate graph
        # graph = nx.Graph()
        # graph.add_nodes_from(movie_list)
        # graph.add_nodes_from(user_list)
        graph.add_edges_from(relation_list)
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_cc)
        graph = nx.relabel.convert_node_labels_to_integers(
            largest_graph, first_label=0, ordering="default"
        )
    elif args.dataset == "citation":
        if not os.path.isfile(
            "/Users/wangyun/Documents/GitHub/GraphHT/datasets/citation/graph.pickle"
        ):
            df_paper_author = pd.read_csv(
                "/Users/wangyun/Documents/GitHub/GraphHT/datasets/citation/paper_author.csv"
            )
            df_paper_paper = pd.read_csv(
                "/Users/wangyun/Documents/GitHub/GraphHT/datasets/citation/paper_paper.csv"
            )
            paper_list = getPaperList(args, df_paper_author)
            graph = nx.Graph()
            graph.add_nodes_from(paper_list)
            author_list = getAuthorList(args, df_paper_author)
            graph.add_nodes_from(author_list)
            assert graph.number_of_nodes() == (
                len(df_paper_author.authorId.unique()) + len(paper_list)
            ), f"number of nodes != unique author + unique paper"

            author_paper_relation_list, paper_paper_relation_list = getRelationLists(
                args, graph, df_paper_author, df_paper_paper
            )

            graph.add_edges_from(author_paper_relation_list)
            graph.add_edges_from(paper_paper_relation_list)

            largest_cc = max(nx.connected_components(graph), key=len)
            largest_graph = graph.subgraph(largest_cc)
            graph = nx.relabel.convert_node_labels_to_integers(
                largest_graph, first_label=0, ordering="default"
            )
        else:
            print("loading dataset.")
            graph = pickle.load(
                open(
                    "/Users/wangyun/Documents/GitHub/GraphHT/datasets/citation/graph.pickle",
                    "rb",
                )
            )
    else:
        raise Exception(f"We dont support {args.dataset}")

    args.num_nodes = graph.number_of_nodes()
    args.num_edges = graph.number_of_edges()
    args.logger.info(f"{args.dataset} has {args.num_nodes} nodes.")
    args.logger.info(f"{args.dataset} has {args.num_edges} edges.")
    args.logger.info(
        f"The max degree of {args.dataset} has node index and degree pair: {sorted(graph.degree, key=lambda x: x[1], reverse=True)[0]}"
    )
    # print(graph.number_of_nodes(),graph.number_of_edges())

    sampling_init(args, graph)
    if hasattr(args, "smallest_size"):
        args.logger.info(
            f"{args.sampling_method} -> The smallest sample size to maintain {args.property} for {args.dataset} is {round(args.smallest_size,4)}"
        )
    else:
        args.logger.info(
            f"{args.sampling_method} -> Cannot find the smallest sample size to maintain {args.property} for {args.dataset} given ratio_list {args.ratio_list}"
        )


def sampling_init(args, graph):
    ori_result_list = cc_dist(args, graph)
    plt.plot(ori_result_list, label="original cc dist")

    args.result = defaultdict(list)
    degreeView = graph.degree()
    for ratio in args.ratio_list:
        args.logger.info(ratio)

        ## different sampling method

        ######### RW-based #########
        if args.sampling_method == "SRW":
            args.model = RandomWalkSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            if args.with_weight_func == True:
                ## why after re-weighting for an unbiased estimator, the distributions are different from the original one even when sample size increases?
                ## Do we need to consider the weight function?
                ## Yes! the weight function in SRW is very naive so that nodes with degree 0 is re-weighted to be very high proportional in the degree distribution
                ## note that the result degree distribution is an estimator, rather than the induced subgraph, whose deg dist. must converge to the proginal one when sample size becomes 1
                for d in range(1, args.n_deg + 1):
                    numerator_sum = 0
                    denominator_sum = 0
                    for i in new_graph.nodes():
                        weight = 1 / degreeView[i]
                        if degreeView[i] == d:
                            numerator_sum += weight
                        denominator_sum += weight
                    args.result[ratio].append(numerator_sum / denominator_sum)
            else:
                args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "DiffusionTreeS":
            args.model = DiffusionTreeSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "FFS":
            args.model = ForestFireSampler(
                number_of_nodes=int(args.num_nodes * ratio),
                seed=int(args.seed),
                p=0.4,
                max_visited_nodes_backlog=int(args.num_nodes * ratio),
            )
            # (number_of_nodes: int = 100, p: float = 0.4, seed: int = 42, max_visited_nodes_backlog: int = 100, restart_hop_size: int = 10)
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "SpikyBallS":
            args.model = SpikyBallSampler(
                number_of_nodes=int(args.num_nodes * ratio),
                seed=int(args.seed),
                initial_nodes_ratio=0.05,
                sampling_probability=0.4,
                max_visited_nodes_backlog=int(args.num_nodes * ratio),
                mode=args.sampling_mode,
            )
            # (number_of_nodes: int = 100, sampling_probability: float = 0.2, initial_nodes_ratio: float = 0.1, seed: int = 42, max_hops: int = 100000, mode: str = 'fireball', max_visited_nodes_backlog: int = 100, restart_hop_size: int = 10, distrib_coeff: float = 1.0)
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "NBRW":
            ## TODO: 是否需要reweight？
            args.model = NonBackTrackingRandomWalkSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "SBS":
            args.model = SnowBallSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "RW_Starter":
            args.model = RandomWalkWithRestartSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "BFS":
            args.model = BreadthFirstSearchSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "DFS":
            args.model = DepthFirstSearchSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "RW_Jump":
            args.model = RandomWalkWithJumpSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "FrontierS":
            args.model = FrontierSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "RNNS":
            args.model = RandomNodeNeighborSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "ShortestPathS":
            args.model = ShortestPathSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "CommunitySES":
            args.model = CommunityStructureExpansionSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "CNRWS":
            args.model = CirculatedNeighborsRandomWalkSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "MHRS":
            args.model = ModifiedMHRS(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = args.model.sample(graph)
            if args.with_weight_func == True:
                for d in range(1, args.n_deg + 1):
                    numerator_sum = 0
                    denominator_sum = 0
                    for i in new_graph.nodes():
                        weight = args.model._weight_dict[i]
                        if degreeView[i] == d:
                            numerator_sum += weight
                        denominator_sum += weight
                    args.result[ratio].append(numerator_sum / denominator_sum)
            else:
                args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "CNARW":
            model = CommonNeighborAwareRandomWalkSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            if args.with_weight_func == True:
                for d in range(1, args.n_deg + 1):
                    numerator_sum = 0
                    denominator_sum = 0
                    for i in new_graph.nodes():
                        model._get_node_scores(graph, i)
                        p_uu = 1 - sum(model._sampler[i]["scores"]) / len(
                            model._sampler[i]["neighbors"]
                        )
                        lambda_i = 1 / (1 - p_uu)
                        weight = lambda_i / len(model._sampler[i]["neighbors"])
                        # print(weight)
                        if degreeView[i] == d:
                            numerator_sum += weight
                        denominator_sum += weight
                    args.result[ratio].append(numerator_sum / denominator_sum)
            else:
                # print(ratio)
                args.result[ratio] = cc_dist(args, new_graph)

        ######### node sampling #########
        elif args.sampling_method == "RNS":
            model = RandomNodeSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "DBS":
            model = DegreeBasedSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "PRBS":
            model = PageRankBasedSampler(
                number_of_nodes=int(args.num_nodes * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        ######### edge sampling #########
        elif args.sampling_method == "RES":
            model = RandomEdgeSampler(
                number_of_edges=int(args.num_edges * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "RNES":
            model = RandomNodeEdgeSampler(
                number_of_edges=int(args.num_edges * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        elif args.sampling_method == "RES_Induction":
            model = RandomEdgeSamplerWithInduction(
                number_of_edges=int(args.num_edges * ratio), seed=int(args.seed)
            )
            new_graph = model.sample(graph)
            args.result[ratio] = cc_dist(args, new_graph)

        else:
            # print(args.sampling_method)
            print(f"{args.sampling_method} is an invalid sampling method")

        plt.plot(args.result[ratio], label=f"cc dist ({round(ratio,4)})")
        ## after getting resulted sampled graph for each ratio, then KS-test
        # print(ori_result_list)
        # print(args.result[ratio])
        _, pvalue = stats.ks_2samp(args.result[ratio], ori_result_list)
        if pvalue <= 0.05:
            args.logger.info(
                f"{ratio}: p-value is {round(pvalue,4)}, so we reject the null hypothesis. Samples are from different distributions."
            )
            # print(f"{i}: p-value is {round(pvalue,4)}, so we reject the null hypothesis. Samples are from different distributions.")
        else:
            args.logger.info(
                f"{ratio}: p-value is {round(pvalue,4)}, so we accept the null hypothesis. Samples are from the same distribution."
            )
            args.smallest_size = ratio
            break
            # print(f"{i}: p-value is {round(pvalue,4)}, so we accept the null hypothesis. Samples are from the same distribution.")

    plt.xlabel("cc")
    plt.ylabel("cdf")
    plt.legend()
    plt.title(f"cc distribution ({args.dataset}) - {args.sampling_method}")
    if args.sampling_method == "SpikyBallS":
        args.fig_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results",
            args.property
            + "_"
            + args.dataset
            + "_"
            + args.sampling_method
            + "_"
            + args.sampling_mode
            + "_fig.png",
        )
    else:
        args.fig_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results",
            args.property
            + "_"
            + args.dataset
            + "_"
            + args.sampling_method
            + "_fig.png",
        )
    plt.savefig(args.fig_path)
    # plt.show()


def cc_dist(args, new_graph):
    lcc = nx.clustering(new_graph)
    count, _ = np.histogram(list(lcc.values()), bins=args.bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf


class ModifiedMHRS(MetropolisHastingsRandomWalkSampler):
    def __init__(self, number_of_nodes: int = 100, seed: int = 42, alpha: float = 1.0):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.alpha = alpha
        self._set_seed()
        self._weight_dict = {}

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        score = random.uniform(0, 1)
        new_node = self.backend.get_random_neighbor(graph, self._current_node)
        ratio = float(self.backend.get_degree(graph, self._current_node)) / float(
            self.backend.get_degree(graph, new_node)
        )
        ratio = ratio**self.alpha
        if score < ratio:
            self._current_node = new_node
            self._weight_dict[self._current_node] = ratio
            self._sampled_nodes.add(self._current_node)


if __name__ == "__main__":
    main()