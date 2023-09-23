import os
import statistics
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from config import parse_args
from new_graph_hypo_postprocess import getEdges, getGenres
from sampling import (
    CNARW,
    DBS,
    FFS,
    MHRWS,
    NBRW,
    PRBS,
    RES,
    RNES,
    RNNS,
    RNS,
    SBS,
    SRW,
    CommunitySES,
    FrontierS,
    RES_Induction,
    RW_Starter,
    ShortestPathS,
)

# from new_graph_hypo_postprocess import new_graph_hypo_result
from scipy import stats
from utils import (
    clean,
    drawAllRatings,
    get_data,
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

    graph = get_data(args)

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

    # look for
    # if args.hypo == 2:
    #     # keep node size
    #     if args.sampling_method == "RES" or args.sampling_method == "RES_Induction":
    #         default_num_samples = args.num_samples
    #         default_sampling_ratio = args.sampling_ratio
    #         args.num_samples = 1
    #         if args.dataset == "movielens":
    #             sampling_ratio_start = 50
    #             sampling_ratio_end = 5000
    #             jump = 20
    #         elif args.dataset == "citation":
    #             raise Exception("not pre-set")
    #         elif args.dataset == "yelp":
    #             sampling_ratio_start = 30000
    #             sampling_ratio_end = 1500000
    #             jump = 5000
    #             threshold = 4000

    #         sampling_ratio = []
    #         while default_sampling_ratio != []:
    #             sampling_ratio_start = search(
    #                 args,
    #                 graph,
    #                 sampling_ratio_start,
    #                 sampling_ratio_end,
    #                 jump,
    #                 threshold,
    #                 default_sampling_ratio[0],
    #             )
    #             sampling_ratio_start
    #             sampling_ratio.append(sampling_ratio_start)
    #             print(sampling_ratio)
    #             default_sampling_ratio = default_sampling_ratio[1:]

    #         args.num_samples = default_num_samples
    #         args.sampling_ratio = sampling_ratio
    #         print(args.sampling_ratio)

    #         # sampling_ratio = []
    #         # for size in range(sampling_ratio_start, sampling_ratio_end, jump):
    #         #     args.sampling_ratio = [size]
    #         #     result_list = samplingGraph(args, graph, True)
    #         #     num_node = result_list[0]
    #         #     if abs(default_sampling_ratio[0] - num_node) < threshold:
    #         #         sampling_ratio.append(size)
    #         #         default_sampling_ratio = default_sampling_ratio[1:]
    #         #         if default_sampling_ratio == []:
    #         #             args.sampling_ratio = sampling_ratio
    #         #             args.num_samples = default_num_samples
    #         #             break

    # elif args.hypo == 3:
    #     # keep edge size
    #     if args.sampling_method != "RES":
    #         default_num_samples = args.num_samples
    #         default_sampling_ratio = args.sampling_ratio
    #         args.num_samples = 1
    #         if args.dataset == "movielens":
    #             sampling_ratio_start = 50
    #             sampling_ratio_end = 1000
    #             jump = 20
    #         elif args.dataset == "citation":
    #             raise Exception("not pre-set")
    #         elif args.dataset == "yelp":
    #             sampling_ratio_start = 5000
    #             sampling_ratio_end = 100000
    #             jump = 2000
    #             threshold = 2000
    # else:
    #     pass

    if args.dataset == "movielens":
        args.ground_truth = getGroundTruth(args, graph)
    elif args.dataset == "citation":
        args.ground_truth = getGroundTruth(args, graph)
    elif args.dataset == "yelp":
        args.ground_truth = getGroundTruth(args, graph)

    if args.HTtype == "one-sample":
        args.ground_truth = args.ground_truth[str(list(args.attribute.keys())[0])]
        args.CI = []
        args.popmean_lower = round(args.ground_truth - 0.05, 2)
        args.popmean_higher = round(args.ground_truth + 0.05, 2)

    # sample for each sampling ratio
    args.result = defaultdict(list)
    args.time_result = defaultdict(list)
    for ratio in args.sampling_ratio:
        # args.time_limit=time.time()
        args.valid_edges = []
        args.variance = []
        time_ratio_start = time.time()
        args.ratio = ratio
        args.logger.info(" ")
        args.logger.info(f">>> sampling ratio: {args.ratio}")
        result_list = samplingGraph(args, graph, False)

        print(
            f"average valid nodes/edges are {sum(args.valid_edges)/len(args.valid_edges)}"
        )
        print(f"average variance is {sum(args.variance)/len(args.variance)}")
        args.logger.info(
            f"average valid nodes/edges are {sum(args.valid_edges)/len(args.valid_edges)}"
        )
        args.logger.info(f"average variance is {sum(args.variance)/len(args.variance)}")

        # print total time used
        total_time = time.time() - time_ratio_start
        total_time_format = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        args.logger.info(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )

        ##############################
        ##### Hypothesis Testing #####
        ##############################
        if args.HTtype == "one-sample":
            # print(result_list)
            result_list = [i[str(list(args.attribute.keys())[0])] for i in result_list]
            result = sum(result_list) / len(result_list)
            # print percentage error w.r.t. the ground truth
            percent_error = 100 * abs((result) - args.ground_truth) / args.ground_truth
            print(
                f">>> Percentage error of sampling result {round(result,4)} w.r.t. the ground truth {round(args.ground_truth,4)} at {args.ratio} sampling ratio is {round(percent_error,2)}%."
            )
            args.logger.info(
                f">>> Percentage error of sampling result {round(result,4)} w.r.t. the ground truth {round(args.ground_truth,4)} at {args.ratio} sampling ratio is {round(percent_error,2)}%."
            )

            args.time_result[args.ratio].append(round(percent_error, 2))

            args.result[ratio] = result_list

            HypothesisTesting(args, result_list)
            args.logger.info(
                f"The hypothesis testing for {args.ratio} sampling ratio is finished!"
            )
        elif args.HTtype == "two-sample":
            result_list_new = defaultdict(list)

            value = []
            for attribute in args.attribute:
                result_attribute = [i[attribute] for i in result_list]
                result_list_new[attribute] = result_attribute
                value.append(sum(result_attribute) / len(result_attribute))

            percent_error_0 = (
                100
                * abs(value[0] - args.ground_truth[str(list(args.attribute.keys())[0])])
                / args.ground_truth[str(list(args.attribute.keys())[0])]
            )
            percent_error_1 = (
                100
                * abs(value[1] - args.ground_truth[args.attribute[1]])
                / args.ground_truth[args.attribute[1]]
            )

            percent_error = (percent_error_0 + percent_error_1) / 2
            print(
                f">>> {str(list(args.attribute.keys())[0])}: sampled result is {value[0]}."
            )
            print(f">>> {args.attribute[1]}: sampled result is {value[1]}.")
            args.logger.info(
                f">>> {str(list(args.attribute.keys())[0])}: sampled result is {value[0]}."
            )
            args.logger.info(f">>> {args.attribute[1]}: sampled result is {value[1]}.")
            print(
                f">>> Percentage error of {str(list(args.attribute.keys())[0])} at {args.ratio} sampling ratio is {round(percent_error_0,2)}%."
            )
            print(
                f">>> Percentage error of {args.attribute[1]} at {args.ratio} sampling ratio is {round(percent_error_1,2)}%."
            )
            args.logger.info(
                f">>> Percentage error of {str(list(args.attribute.keys())[0])} at {args.ratio} sampling ratio is {round(percent_error_0,2)}%."
            )
            args.logger.info(
                f">>> Percentage error of {args.attribute[1]} at {args.ratio} sampling ratio is {round(percent_error_1,2)}%."
            )

            args.time_result[args.ratio].append(round(percent_error, 2))

            # args.result[ratio] = result_list

            HypothesisTesting(args, result_list_new)
            args.logger.info(
                f"The hypothesis testing for {args.ratio} sampling ratio is finished!"
            )

        else:
            raise Exception(
                "Sorry we do not support hypothesis types other than one-sample and two-sample."
            )

    # drawAllRatings(args, args.result)

    headers = [
        "Sampling time",
        "Target extraction time",
        "HT time",
        "Percentage error",
    ]

    print(
        f"{headers[0].capitalize(): <25}{headers[1].capitalize(): <25}{headers[2].capitalize():<25}{headers[3].capitalize():<25}"
    )
    args.logger.info(
        f"{headers[0].capitalize(): <25}{headers[1].capitalize(): <25}{headers[2].capitalize():<25}{headers[3].capitalize():<25}"
    )

    for _, value in args.time_result.items():
        # print(value)
        print(f"{value[0]: <25}{value[1]: <25}{value[3]:<25}{value[2]:<25}")
        args.logger.info(f"{value[0]: <25}{value[1]: <25}{value[3]:<25}{value[2]:<25}")
    print(
        f"All hypothesis testing for ratio list {args.sampling_ratio} and ploting is finished!"
    )


def getGroundTruth(args, graph, **kwargs):
    time_get_ground_truth = time.time()
    dict_result = {}

    # chekc if the hypothesis is one-sample or two-sample by their length of attribute list
    if args.hypo < 10:
        args.HTtype = "one-sample"

        if (args.attribute is not None) and (len(args.attribute) == 1):
            # attribute = str(list(args.attribute.keys())[0])
            pass
        else:
            args.logger.error(
                f"Sorry, args.attribute is None or len(args.attribute) != 1."
            )

            raise Exception(
                f"Sorry, args.attribute is None or len(args.attribute) != 1."
            )
    else:
        args.HTtype = "two-sample"
        assert (
            args.comparison is not None
        ), f"{args.dataset} requires the args.comparison parameter."

        if (args.attribute is not None) and (len(args.attribute) == 2):
            pass
        else:
            args.logger.error(
                f"Sorry, args.attribute is None or len(args.attribute) != 2."
            )

            raise Exception(
                f"Sorry, args.attribute is None or len(args.attribute) != 2."
            )

    if args.dataset == "movielens":
        if args.hypo == 1:
            args.H0 = f"{args.agg} number of {args.attribute[0]} movies rated by users"
            df_movies = df_movies[df_movies[args.attribute[0]] == 1]
            df_ratings = pd.merge(df_ratings, df_movies, on="movieId", how="inner")
            df_ratings = df_ratings.groupby("userId").count()
            dict_result[args.attribute[0]] = df_ratings.movieId.to_list()

        elif args.hypo == 2:
            args.H0 = f"{args.agg} number of genres {str(list(args.attribute.keys())[0])} movies have"
            dict_result[str(list(args.attribute.keys())[0])] = getGenres(
                args, graph, dimension={"movie": "genre"}
            )[str(list(args.attribute.keys())[0])]

        elif args.hypo == 3:
            args.H0 = (
                f"{args.agg} rating of {str(list(args.attribute.keys())[0])} movies"
            )
            dict_result[str(list(args.attribute.keys())[0])] = getEdges(args, graph)[
                str(list(args.attribute.keys())[0])
            ]  # {attribute[0]:[1,2,3]}

        # two sample hypo
        elif args.hypo == 10:
            if args.comparison == "!=":
                args.H0 = f"{args.agg} rating of {args.attribute[0]} movies == that of {args.attribute[1]} movies."
            else:
                args.H0 = f"{args.agg} rating of {args.attribute[0]} movies {args.comparison} that of {args.attribute[1]} movies."
            movie_0 = df_movies.loc[:, ["movieId", args.attribute[0]]]
            df_ratings_0 = pd.merge(movie_0, df_ratings, on="movieId", how="inner")
            movie_1 = df_movies.loc[:, ["movieId", args.attribute[1]]]
            df_ratings_1 = pd.merge(movie_1, df_ratings, on="movieId", how="inner")

            dict_result[args.attribute[0]] = df_ratings_0[
                df_ratings_0[args.attribute[0]] == 1
            ].rating.values.tolist()
            dict_result[args.attribute[1]] = df_ratings_1[
                df_ratings_1[args.attribute[1]] == 1
            ].rating.values.tolist()

        else:
            args.logger.error(
                f"Sorry, {args.hypo} is not supported for {args.dataset}."
            )
            raise Exception(f"Sorry, {args.hypo} is not supported for {args.dataset}.")

    elif args.dataset == "citation":
        if args.hypo == 1:
            args.H0 = f"{args.agg} authors of papers in {args.attribute[0]}"
            if args.attribute is not None:
                df_paper_author = df_paper_author[
                    df_paper_author.year == int(args.attribute[0])
                ]
                df = df_paper_author.groupby("paperTitle").count()
                dict_result[args.attribute[0]] = df.author.values.tolist()
            else:
                args.logger.error(f"Sorry, args.attribute is None.")
                raise Exception(f"Sorry, args.attribute is None.")
        elif args.hypo == 2:
            args.H0 = f"{args.agg} citation of papers in {str(list(args.attribute.keys())[0])}"
            dict_result[str(list(args.attribute.keys())[0])] = getGenres(
                args, graph, dimension={"paper": "citation"}
            )[str(list(args.attribute.keys())[0])]

        elif args.hypo == 3:
            args.H0 = f"{args.agg} correlation score of papers in {str(list(args.attribute.keys())[0])} with its related papers"
            dict_result[str(list(args.attribute.keys())[0])] = getEdges(args, graph)[
                str(list(args.attribute.keys())[0])
            ]  # {attribute[0]:[1,2,3]}

        elif args.hypo == 4:
            args.H0 = f"{args.agg} number of triangles"
            dict_result[args.attribute[0]] = [sum(nx.triangles(graph).values()) / 3]

        # two sample hypo
        elif args.hypo == 10:
            if args.comparison == "!=":
                args.H0 = f"{args.agg} citations of {args.attribute[0]} papers == that of {args.attribute[1]} papers."
            else:
                args.H0 = f"{args.agg} citations of {args.attribute[0]} papers {args.comparison} that of {args.attribute[1]} papers."

            # args.H0 = f"{args.agg} citation of papers in {args.attribute[0]}"
            dict_result[args.attribute[0]] = []
            dict_result[args.attribute[1]] = []
            node_attribute = nx.get_node_attributes(graph, "year")
            for key, value in node_attribute.items():
                if value == int(args.attribute[0]):
                    dict_result[args.attribute[0]].append(graph.nodes[key]["citation"])
                elif value == int(args.attribute[1]):
                    dict_result[args.attribute[1]].append(graph.nodes[key]["citation"])

        else:
            args.logger.error(
                f"Sorry, {args.hypo} is not supported for {args.dataset}."
            )
            raise Exception(f"Sorry, {args.hypo} is not supported for {args.dataset}.")
    elif args.dataset == "yelp":
        if args.hypo == 3:
            args.H0 = f"{args.agg} stars  in {str(list(args.attribute.keys())[0])}"

            dict_result[str(list(args.attribute.keys())[0])] = getEdges(args, graph)[
                str(list(args.attribute.keys())[0])
            ]  # {attribute[0]:[1,2,3]}
            import collections

            frequency = collections.Counter(
                dict_result[str(list(args.attribute.keys())[0])]
            )
            print(dict(frequency))
        elif args.hypo == 2:
            args.H0 = f"{args.agg} number of stars {str(list(args.attribute.keys())[0])} restaurants have"
            dict_result[str(list(args.attribute.keys())[0])] = getGenres(
                args, graph, dimension={"business": "stars"}
            )[str(list(args.attribute.keys())[0])]

        else:
            args.logger.error(
                f"Sorry, {args.hypo} is not supported for {args.dataset}."
            )
            raise Exception(f"Sorry, {args.hypo} is not supported for {args.dataset}.")

    if len(dict_result[str(list(args.attribute.keys())[0])]) == 0:
        args.logger.error(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )
        raise Exception(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )

    print(
        f"total number of valid nodes/edges {len(dict_result[str(list(args.attribute.keys())[0])])}"
    )
    args.logger.info(
        f"total number of valid nodes/edges {len(dict_result[str(list(args.attribute.keys())[0])])}"
    )
    # check aggregation method
    ground_truth = {}
    for k, v in dict_result.items():
        if args.agg == "mean":
            ground_truth[k] = sum(v) / len(v)
        elif args.agg == "max":
            ground_truth[k] = max(v)
        elif args.agg == "min":
            ground_truth[k] = min(v)
        elif args.agg == "variance":
            ground_truth[k] = statistics.variance(v)
        elif args.agg == "number":
            ground_truth[k] = v[0]
        else:
            args.logger.error(f"Sorry, we don't support {args.agg}.")
            raise Exception(f"Sorry, we don't support {args.agg}.")

    time_now = time.time()
    for k, v in ground_truth.items():
        args.logger.info(
            f"{k}: The ground truth is {round(v,5)}, taking time {round(time_now-time_get_ground_truth,5)}."
        )
    return ground_truth


def HypothesisTesting(args, result_list, verbose=1):
    time_start_HT = time.time()

    #################################
    # test H1: avg rating = popmean #
    #################################
    if args.HTtype == "one-sample":
        # Two side HT
        # variance = statistics.variance(result_list)
        # print(f"The variance is {round(variance,3)}.")
        # args.logger.info(f"The variance is {round(variance,3)}.")
        t_stat, p_value = stats.ttest_1samp(
            result_list, popmean=args.ground_truth, alternative="two-sided"
        )
        if verbose == 1:
            print_hypo_log(args, t_stat, p_value, args.H0, twoSides=True)
    else:
        if args.comparison == "==" or args.comparison == "!=":
            alternative = "two-sided"
        elif args.comparison == "<":
            alternative = "greater"
        else:
            alternative = "less"

        # H0: equal variance
        t_stat, p_value = stats.levene(
            result_list[str(list(args.attribute.keys())[0])],
            result_list[args.attribute[1]],
            center="mean",
        )
        print(f"pvalue: {p_value} (< 0.05 means inequal variance).")
        args.logger.info(f"pvalue: {p_value} (< 0.05 means inequal variance).")
        # print("")

        t_stat, p_value = stats.ttest_ind(
            a=result_list[str(list(args.attribute.keys())[0])],
            b=result_list[args.attribute[1]],
            equal_var=True,
            alternative=alternative,
        )
        if verbose == 1:
            print_hypo_log(args, t_stat, p_value, args.H0)
    # CI = st.t.interval(
    #     confidence=0.95,
    #     df=len(result_list) - 1,
    #     loc=np.mean(result_list),
    #     scale=st.sem(result_list),
    # )
    # args.CI.append(CI)
    # args.logger.info(CI)

    #######################################
    # test H1: avg rating > popmean_lower #
    #######################################

    # H0 = f"{args.agg} rating of {args.attribute} users"
    # t_stat, p_value = ttest_1samp(
    #     result_list, popmean=args.popmean_lower, alternative="greater"
    # )
    # if verbose == 1:
    #     print_hypo_log(args, t_stat, p_value, args.H0, oneSide="lower")

    ########################################
    # test H1: avg rating < popmean_higher #
    ########################################

    # H0 = f"{args.agg} rating of {args.attribute} users"
    # t_stat, p_value = ttest_1samp(
    #     result_list, popmean=args.popmean_higher, alternative="less"
    # )
    # if verbose == 1:
    #     print_hypo_log(args, t_stat, p_value, args.H0, oneSide="higher")

    HTTime = round(time.time() - time_start_HT, 5)
    print(f">>> Time for hypothesis testing is {HTTime}.")
    print("")
    args.logger.info(f">>> Time for hypothesis testing is {HTTime}.")
    args.logger.info("")
    args.time_result[args.ratio].append(HTTime)


def samplingGraph(args, graph, find_stop=False):
    # time_sampling_graph_start = time.time()

    result_list = []
    time_used_list = defaultdict(list)
    ##############################
    ######## Exploration #########
    ##############################
    if args.sampling_method == "RNNS":
        result_list, time_used = RNNS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "SRW":
        result_list, time_used = SRW(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "ShortestPathS":
        result_list, time_used = ShortestPathS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "MHRWS":
        result_list, time_used = MHRWS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "CommunitySES":
        result_list, time_used = CommunitySES(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "CNARW":
        result_list, time_used = CNARW(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "FFS":
        result_list, time_used = FFS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "SBS":
        result_list, time_used = SBS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "FrontierS":
        result_list, time_used = FrontierS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "NBRW":
        result_list, time_used = NBRW(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "RW_Starter":
        result_list, time_used = RW_Starter(
            args, graph, result_list, time_used_list, find_stop
        )

    ###############################
    ######## Node Sampler #########
    ###############################
    elif args.sampling_method == "RNS":
        result_list, time_used = RNS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "DBS":
        result_list, time_used = DBS(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "PRBS":
        result_list, time_used = PRBS(
            args, graph, result_list, time_used_list, find_stop
        )

    ###############################
    ######## Edge Sampler #########
    ###############################
    elif args.sampling_method == "RES":
        result_list, time_used = RES(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "RNES":
        result_list, time_used = RNES(
            args, graph, result_list, time_used_list, find_stop
        )

    elif args.sampling_method == "RES_Induction":
        result_list, time_used = RES_Induction(
            args, graph, result_list, time_used_list, find_stop
        )

    else:
        args.logger.error(f"Sorry, we don't support {args.sampling_method}.")
        raise Exception(f"Sorry, we don't support {args.sampling_method}.")

    if find_stop:
        return result_list

    time_one_sample = sum(time_used["sampling"]) / len(time_used["sampling"])
    print(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample,2)}."
    )
    args.logger.info(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample,2)}."
    )
    args.time_result[args.ratio].append(round(time_one_sample, 2))
    time_rating_extraction = sum(time_used["sample_graph_by_condition"]) / len(
        time_used["sample_graph_by_condition"]
    )
    print(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_rating_extraction,5)}."
    )
    args.logger.info(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_rating_extraction,5)}."
    )
    args.time_result[args.ratio].append(round(time_rating_extraction, 5))

    return result_list


def search(
    args, graph, sampling_ratio_start, sampling_ratio_end, jump, threshold, target
):
    # sampling_ratio=[]
    for size in range(sampling_ratio_start, sampling_ratio_end, jump):
        print(size)
        args.ratio = size
        result_list = samplingGraph(args, graph, True)
        num_node = result_list[0]
        if abs(target - num_node) < threshold:
            return size
        if num_node > target:
            return size
            # sampling_ratio.append(size)
            # default_sampling_ratio = default_sampling_ratio[1:]
            # if default_sampling_ratio == []:
            #     args.sampling_ratio = sampling_ratio
            #     args.num_samples = default_num_samples
            #     break


if __name__ == "__main__":
    # import cProfile

    # cProfile.run(
    #     "main()",
    #     filename="HypothesisTesting/log_and_results_2008/result.out",
    #     sort="cumulative",
    # )
    main()
