import os
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from config import parse_args
from dataprep.citation_prep import citation_prep
from dataprep.movielens_prep import movielens_prep, moviePreprocess
from new_graph_hypo_postprocess import new_graph_hypo_result
from scipy.stats import ttest_1samp
from utils import (
    clean,
    drawAllRatings,
    logger,
    print_hypo_log,
    setup_device,
    setup_seed,
)


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    logger(args)

    # global info of this run
    args.logger.info(f"Dataset: {args.dataset}")
    args.logger.info(f"Sampling Method: {args.sampling_method}")
    args.logger.info(f"Sampling Ratio: {args.sampling_ratio}")
    args.logger.info(f"Attribute: {args.attribute}")
    args.logger.info(f"Aggregation Method: {args.agg}")
    args.logger.info(f"=========== Start Running ===========")

    # get the graph
    time_dataset_prep = time.time()
    args.dataset_path = os.path.join(os.getcwd(), "datasets", args.dataset)
    if args.dataset == "movielens":
        assert (
            len(args.attribute) == 1
        ), f"Only one movie genre attribute is required for {args.dataset}."

        df_movies = pd.read_csv(os.path.join(args.dataset_path, "movies.csv"))
        df_ratings = pd.read_csv(os.path.join(args.dataset_path, "ratings.csv"))
        df_movies = moviePreprocess(df_movies)
        graph = movielens_prep(args, df_movies, df_ratings)

    elif args.dataset == "citation":
        assert (
            len(args.attribute) == 1
        ), f"Only one year attribute is required for {args.dataset}."
        args.attribute = args.attribute[0]

        df_paper_author = pd.read_csv(
            os.path.join(args.dataset_path, "paper_author.csv")
        )
        df_paper_paper = pd.read_csv(os.path.join(args.dataset_path, "paper_paper.csv"))
        graph = citation_prep(args, df_paper_author, df_paper_paper)

    else:
        args.logger.error(f"Sorry, we don't support {args.dataset}.")
        raise Exception(f"Sorry, we don't support {args.dataset}.")

    print(
        f">>> Total time for dataset {args.dataset} preperation is {round((time.time() - time_dataset_prep),2)}."
    )
    args.logger.info(
        f">>> Total time for dataset {args.dataset} preperation is {round((time.time() - time_dataset_prep),2)}."
    )

    # print graph characteristics
    args.num_nodes = graph.number_of_nodes()
    args.num_edges = graph.number_of_edges()
    args.logger.info(
        f"{args.dataset} has {args.num_nodes} nodes and {args.num_edges} edges."
    )
    args.logger.info(f"{args.dataset} is connected: {nx.is_connected(graph)}.")

    # prepare ground truths and population mean for one-side hypothesis testings
    if args.dataset == "movielens":
        args.ground_truth = getGroundTruth(
            args, df_movies=df_movies, df_ratings=df_ratings
        )
    elif args.dataset == "citation":
        args.ground_truth = getGroundTruth(args, df_paper_author=df_paper_author)
    args.CI = []
    args.popmean_lower = round(args.ground_truth - 0.05, 2)
    args.popmean_higher = round(args.ground_truth + 0.05, 2)

    # sample for each sampling ratio
    args.result = defaultdict(list)
    time_ratio_start = time.time()
    for ratio in args.sampling_ratio:
        args.ratio = ratio
        args.logger.info(" ")
        args.logger.info(f">>> sampling ratio: {args.ratio}")
        result_list = samplingGraph(args, graph)

        total_time = time.time() - time_ratio_start
        total_time_format = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        args.logger.info(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        args.result[ratio] = result_list

        testHypothesis(args, result_list)
        args.logger.info(
            f"The hypothesis testing for {args.ratio} sampling ratio is finished!"
        )

    drawAllRatings(args, args.result)
    print(
        f"All hypothesis testing for ratio list {args.sampling_ratio} and ploting is finished!"
    )


def getGroundTruth(args, **kwargs):
    if args.dataset == "movielens":
        assert len(kwargs) == 2, f"{args.dataset} requires df_movies and df_ratings."

        df_movies = kwargs["df_movies"]
        df_ratings = kwargs["df_ratings"]

        if (args.attribute is not None) and (len(args.attribute) == 1):
            movie = df_movies.loc[:, ["movieId"] + args.attribute]
            df_ratings = pd.merge(movie, df_ratings, on="movieId", how="inner")
            attribute = args.attribute[0]
            list_result = df_ratings[df_ratings[attribute] == 1].rating.values.tolist()
        else:
            args.logger.error(
                f"Sorry, args.attribute is None or len(args.attribute) != 1."
            )

            raise Exception(
                f"Sorry, args.attribute is None or len(args.attribute) != 1."
            )

    elif args.dataset == "citation":
        assert len(kwargs) == 1, f"{args.dataset} requires df_paper_author."

        df_paper_author = kwargs["df_paper_author"]

        if args.attribute is not None:
            df_paper_author = df_paper_author[df_paper_author.year == args.attribute]
            df = df_paper_author.groupby("paperTitle").count()
            list_result = df.author.values.tolist()
        else:
            args.logger.error(f"Sorry, args.attribute is None.")
            raise Exception(f"Sorry, args.attribute is None.")

    if len(list_result) == 0:
        args.logger.error(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )
        raise Exception(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )

    # check aggregation method
    if args.agg == "mean":
        ground_truth = sum(list_result) / len(list_result)
    elif args.agg == "max":
        ground_truth = max(list_result)
    elif args.agg == "min":
        ground_truth = min(list_result)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")

    args.logger.info(f"The ground truth is {round(ground_truth,2)}.")
    return ground_truth


def testHypothesis(args, result_list, verbose=1):
    time_start_HT = time.time()

    #################################
    # test H1: avg rating = popmean #
    #################################

    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.ground_truth, alternative="two-sided"
    )
    H0 = f"{args.agg} rating of {args.attribute} users"

    if verbose == 1:
        print_hypo_log(args, t_stat, p_value, H0, twoSides=True)

    CI = st.t.interval(
        confidence=0.95,
        df=len(result_list) - 1,
        loc=np.mean(result_list),
        scale=st.sem(result_list),
    )
    args.CI.append(CI)
    args.logger.info(CI)

    #######################################
    # test H1: avg rating > popmean_lower #
    #######################################

    H0 = f"{args.agg} rating of {args.attribute} users"
    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.popmean_lower, alternative="greater"
    )
    if verbose == 1:
        print_hypo_log(args, t_stat, p_value, H0, oneSide="lower")

    ########################################
    # test H1: avg rating < popmean_higher #
    ########################################

    H0 = f"{args.agg} rating of {args.attribute} users"
    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.popmean_higher, alternative="less"
    )
    if verbose == 1:
        print_hypo_log(args, t_stat, p_value, H0, oneSide="higher")

    print(f">>> Time for hypothesis testing is {round(time.time()-time_start_HT,5)}.")
    args.logger.info(
        f">>> Time for hypothesis testing is {round(time.time()-time_start_HT,5)}."
    )


def samplingGraph(args, graph):
    # time_sampling_graph_start = time.time()
    result_list = []
    time_used = defaultdict(list)
    if args.sampling_method == "RNNS":
        from littleballoffur import RandomNodeNeighborSampler

        for num_sample in range(args.num_samples):
            time_one_sample_start = time.time()
            model = RandomNodeNeighborSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            time_used["sampling"].append(round(time.time() - time_one_sample_start, 2))

            time_rating_extraction_start = time.time()
            result_list = new_graph_hypo_result(
                args, new_graph, result_list, num_sample
            )
            time_used["rating_extraction"].append(
                round(time.time() - time_rating_extraction_start, 2)
            )

    elif args.sampling_method == "SRW":
        from littleballoffur import RandomWalkSampler

        for num_sample in range(args.num_samples):
            time_one_sample_start = time.time()
            model = RandomWalkSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            time_used["sampling"].append(round(time.time() - time_one_sample_start, 2))

            time_rating_extraction_start = time.time()
            result_list = new_graph_hypo_result(
                args, new_graph, result_list, num_sample
            )
            time_used["rating_extraction"].append(
                round(time.time() - time_rating_extraction_start, 2)
            )

    elif args.sampling_method == "ShortestPathS":
        from littleballoffur import ShortestPathSampler

        for num_sample in range(args.num_samples):
            time_one_sample_start = time.time()
            model = ShortestPathSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            time_used["sampling"].append(round(time.time() - time_one_sample_start, 2))

            time_rating_extraction_start = time.time()
            result_list = new_graph_hypo_result(
                args, new_graph, result_list, num_sample
            )
            time_used["rating_extraction"].append(
                round(time.time() - time_rating_extraction_start, 2)
            )
    elif args.sampling_method == "MHRS":
        from littleballoffur import MetropolisHastingsRandomWalkSampler

        for num_sample in range(args.num_samples):
            time_one_sample_start = time.time()
            model = MetropolisHastingsRandomWalkSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            time_used["sampling"].append(round(time.time() - time_one_sample_start, 2))

            time_rating_extraction_start = time.time()
            result_list = new_graph_hypo_result(
                args, new_graph, result_list, num_sample
            )
            time_used["rating_extraction"].append(
                round(time.time() - time_rating_extraction_start, 2)
            )
    elif args.sampling_method == "CommunitySES":
        from littleballoffur import CommunityStructureExpansionSampler

        for num_sample in range(args.num_samples):
            time_one_sample_start = time.time()
            model = CommunityStructureExpansionSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            time_used["sampling"].append(round(time.time() - time_one_sample_start, 2))

            time_rating_extraction_start = time.time()
            result_list = new_graph_hypo_result(
                args, new_graph, result_list, num_sample
            )
            time_used["rating_extraction"].append(
                round(time.time() - time_rating_extraction_start, 2)
            )
    else:
        args.logger.error(f"Sorry, we don't support {args.sampling_method}.")
        raise Exception(f"Sorry, we don't support {args.sampling_method}.")

    time_one_sample = sum(time_used["sampling"]) / len(time_used["sampling"])
    print(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample,2)}."
    )
    args.logger.info(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample,2)}."
    )
    time_rating_extraction = sum(time_used["rating_extraction"]) / len(
        time_used["rating_extraction"]
    )
    print(
        f">>> Avg time for edge rating extraction at {args.ratio} sampling ratio one time is {round(time_rating_extraction,5)}."
    )
    args.logger.info(
        f">>> Avg time for  edge rating extraction at {args.ratio} sampling ratio one time is {round(time_rating_extraction,5)}."
    )
    return result_list


# def new_graph_hypo_result(args, new_graph, result_list, num_sample):
#     rating_dict = nx.get_edge_attributes(new_graph, name="rating")
#     if args.dataset == "movielens":
#         result = getRatings(args, new_graph, rating_dict)
#         args.logger.info(f"sample {num_sample}: {args.agg} rating is {result}.")
#     elif args.dataset == "citation":
#         result = getAuthors(args, new_graph, rating_dict)
#         args.logger.info(
#             f"sample {num_sample}: {args.agg} number of author is {result}."
#         )
#     result_list.append(result)
#     return result_list


# def getRatings(args, new_graph, rating_dict):
#     total_rating = []
#     for key, value in rating_dict.items():
#         from_node, to_node = key
#         # print(from_node, to_node)
#         if new_graph.nodes[from_node]["label"] == "movie":
#             # print("from")
#             if new_graph.nodes[from_node][args.attribute[0]] == 1:
#                 total_rating.append(value)
#         elif new_graph.nodes[to_node]["label"] == "movie":
#             # print("to")
#             if new_graph.nodes[to_node][args.attribute[0]] == 1:
#                 total_rating.append(value)

#     if len(total_rating) == 0:
#         total_rating.append(0)

#     if args.agg == "mean":
#         result = sum(total_rating) / len(total_rating)
#     elif args.agg == "max":
#         result = max(total_rating)
#     elif args.agg == "min":
#         result = min(total_rating)
#     else:
#         raise Exception(f"Sorry, we don't support {args.agg}.")
#     return result


if __name__ == "__main__":
    main()
