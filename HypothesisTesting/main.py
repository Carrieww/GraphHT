import os
import statistics
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from config import parse_args
from new_graph_hypo_postprocess import getEdges, getNodes, getPaths
from sampling import sample_graph


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
    check_1_sample,
    log_global_info,
)


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    logger(args)

    log_global_info(args)

    graph = prepare_dataset(args)

    run_sampling_and_hypothesis_testing(args, graph)

    print_results(args)


def run_sampling_and_hypothesis_testing(args, graph):
    # sample for each sampling ratio
    args.result = defaultdict(list)
    args.time_result = defaultdict(list)
    args.coverage = defaultdict(list)

    for ratio in args.sampling_ratio:
        # sampling setup and execution
        args.valid_edges = []
        args.variance = []
        time_ratio_start = time.time()
        args.ratio = ratio
        args.logger.info(" ")
        args.logger.info(f">>> sampling ratio: {args.ratio}")
        result_list = samplingGraph(args, graph)

        valid_e_n = round(sum(args.valid_edges) / len(args.valid_edges), 2)
        print(f"average valid nodes/edges are {valid_e_n}")
        args.logger.info(f"average valid nodes/edges are {valid_e_n}")
        if hasattr(args, "variance"):
            if len(args.variance) != 0:
                print(f"average variance is {sum(args.variance)/len(args.variance)}")
                args.logger.info(
                    f"average variance is {sum(args.variance)/len(args.variance)}"
                )

        args.time_result[args.ratio].append(valid_e_n)

        # print total time used
        total_time = time.time() - time_ratio_start
        total_time_format = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        args.logger.info(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        hypo_testing(args, result_list, ratio)


def hypo_testing(args, result_list, ratio):
    if args.HTtype == "one-sample":
        # print(result_list)
        if args.hypo == 3:
            user_cov_list = [
                i[str(list(args.attribute.keys())[0]) + "+user_coverage"]
                for i in result_list
            ]
            movie_cov_list = [
                i[str(list(args.attribute.keys())[0]) + "+movie_coverage"]
                for i in result_list
            ]
            density_list = [i["density"] for i in result_list]
            diameter_list = [i["diameter"] for i in result_list]
            total_valid_path_list = [i["total_valid"] for i in result_list]
            total_valid_path_minus_reverse_list = [
                i["total_minus_reverse"] for i in result_list
            ]

            density = sum(density_list) / len(density_list)
            diameter = sum(diameter_list) / len(diameter_list)
            total_valid = sum(total_valid_path_list) / len(total_valid_path_list)
            total_valid_path_minus_reverse = sum(
                total_valid_path_minus_reverse_list
            ) / len(total_valid_path_minus_reverse_list)

            user_coverage_avg = round(sum(user_cov_list) / len(user_cov_list), 3)
            movie_coverage_avg = round(sum(movie_cov_list) / len(movie_cov_list), 3)
            args.coverage[args.ratio].extend(
                [
                    user_coverage_avg,
                    movie_coverage_avg,
                    round(total_valid, 3),
                    round(total_valid_path_minus_reverse, 3),
                    round(density, 3),
                    round(diameter, 3),
                ]
            )
            print(
                f">>> Diameter of sampling result at {args.ratio} sampling ratio is {round(diameter,3)}."
            )
            args.logger.info(
                f">>> Diameter of sampling result at {args.ratio} sampling ratio is {round(diameter,3)}."
            )

        result_list = [i[str(list(args.attribute.keys())[0])] for i in result_list]
        result = sum(result_list) / len(result_list)

        # print percentage error w.r.t. the ground truth
        percent_error = 100 * abs(result - args.ground_truth) / args.ground_truth
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

        percent_errors = [
            100
            * abs(value[i] - args.ground_truth[args.attribute[i]])
            / args.ground_truth[args.attribute[i]]
            for i in range(len(args.attribute))
        ]
        percent_error = sum(percent_errors) / len(percent_errors)

        for i, attribute in enumerate(args.attribute):
            print(f">>> {attribute}: sampled result is {value[i]}.")
            args.logger.info(f">>> {attribute}: sampled result is {value[i]}.")
            print(
                f">>> Percentage error of {attribute} at {args.ratio} sampling ratio is {round(percent_errors[i], 2)}%."
            )
            args.logger.info(
                f">>> Percentage error of {attribute} at {args.ratio} sampling ratio is {round(percent_errors[i], 2)}%."
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


def print_results(args):
    # drawAllRatings(args, args.result)

    # print the results
    headers = [
        "Sampling time",
        "Target extraction time",
        "HT time",
        "Total Time",
        "Percentage error",
        "node num",
        "Valid nodes/edges/paths",
    ]

    # Print headers
    header_format = " | ".join([header.capitalize().ljust(25) for header in headers])
    print(header_format)
    args.logger.info(header_format)

    list_valid = []
    for index, (ratio, value) in enumerate(args.time_result.items()):
        (
            sampling_time,
            target_extraction_time,
            valid_nodes_edges_paths,
            percent_error,
            ht_time,
        ) = value
        total_time = round(target_extraction_time + sampling_time + ht_time, 3)
        list_valid.append(valid_nodes_edges_paths)

        # Print the results
        result_format = (
            f"{sampling_time:.2f}".ljust(25)
            + f"{target_extraction_time:.2f}".ljust(25)
            + f"{ht_time:.2f}".ljust(25)
            + f"{total_time:.2f}".ljust(25)
            + f"{percent_error:.2f}".ljust(25)
            + f"{args.sampling_ratio[index]}".ljust(25)
            + f"{valid_nodes_edges_paths:.2f}".ljust(25)
        )

        print(result_format)
        args.logger.info(result_format)
    if args.hypo == 3:
        summary_statistics_headers = [
            "User Coverage",
            "Movie Coverage",
            "Total Valid Paths",
            "Reverse Paths",
            "Self-Loops",
            "Density",
            "Diameter",
        ]

        # Print hypothesis headers
        summary_statistics_header_format = " | ".join(
            [header.capitalize().ljust(25) for header in summary_statistics_headers]
        )
        print(summary_statistics_header_format)
        args.logger.info(summary_statistics_header_format)

        for index, (ratio, value) in enumerate(args.coverage.items()):
            (
                user_coverage,
                movie_coverage,
                total_valid_paths,
                total_without_reverse_paths,
                density,
                diameter,
            ) = value
            num_reverse_paths = round(
                total_valid_paths - total_without_reverse_paths, 3
            )
            num_self_loops = round(total_without_reverse_paths - list_valid[index], 3)

            # Print the hypothesis results
            hypothesis_result_format = (
                f"{user_coverage:.3f}".ljust(25)
                + f"{movie_coverage:.3f}".ljust(25)
                + f"{total_valid_paths:.3f}".ljust(25)
                + f"{num_reverse_paths:.3f}".ljust(25)
                + f"{num_self_loops:.3f}".ljust(25)
                + f"{density:.3f}".ljust(25)
                + f"{diameter:.3f}".ljust(25)
            )

            print(hypothesis_result_format)
            args.logger.info(hypothesis_result_format)

    print(
        f"All hypothesis testing for ratio list {args.sampling_ratio} and plotting is finished!"
    )


def prepare_dataset(args):
    # get the graph
    time_dataset_prep = time.time()
    args.dataset_path = os.path.join(os.getcwd(), "datasets", args.dataset)
    graph = get_data(args)

    print(
        f">>> Total time for dataset {args.dataset} preparation is {round((time.time() - time_dataset_prep),2)}."
    )
    args.logger.info(
        f">>> Total time for dataset {args.dataset} preparation is {round((time.time() - time_dataset_prep),2)}."
    )

    # graph characteristics, ground truth setup
    args.num_nodes = graph.number_of_nodes()
    args.num_edges = graph.number_of_edges()
    args.logger.info(
        f"{args.dataset} has {args.num_nodes} nodes and {args.num_edges} edges."
    )
    args.logger.info(f"{args.dataset} is connected: {nx.is_connected(graph)}.")

    if args.dataset in ["movielens", "citation", "yelp"]:
        args.ground_truth = getGroundTruth(args, graph)
    else:
        raise Exception(f"Sorry we do not support {args.dataset} dataset.")

    # check x-sample
    check_1_sample(args)

    if args.HTtype == "one-sample":
        args.ground_truth = args.ground_truth[str(list(args.attribute.keys())[0])]
        args.CI = []
        args.popmean_lower = round(args.ground_truth - 0.05, 2)
        args.popmean_higher = round(args.ground_truth + 0.05, 2)

    return graph


def getGroundTruth(args, graph):
    time_get_ground_truth = time.time()
    dict_result = {}

    # define hypothesis and data processing for each dataset
    # 1: edge hypo; 2: node hypo; 3: path hypo
    # Check if the dataset is supported
    if args.dataset not in {"movielens", "citation", "yelp"}:
        args.logger.error(f"Sorry, {args.dataset} is not supported.")
        raise Exception(f"Sorry, {args.dataset} is not supported.")

    attribute_key = str(list(args.attribute.keys())[0])
    args.H0 = f"{args.agg} {attribute_key} is "

    if args.hypo == 1:
        dict_result[attribute_key] = getEdges(args, graph)[attribute_key]
    elif args.hypo == 2:
        args.dimension = args.attribute[attribute_key]["dimension"]
        dict_result[attribute_key] = getNodes(args, graph, dimension=args.dimension)[
            attribute_key
        ]
    elif args.hypo == 3:
        args.total_valid = 0
        args.total_minus_reverse = 0
        dict_result[attribute_key] = getPaths(args, graph)[attribute_key]
    else:
        args.logger.error(f"Sorry, {args.hypo} is not supported for {args.dataset}.")
        raise Exception(f"Sorry, {args.hypo} is not supported for {args.dataset}.")

    # check if there are valid nodes/edges/paths
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

    # compute the ground truth based on the aggregation method
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

    # log the ground truth and time taken
    time_now = time.time()
    for k, v in ground_truth.items():
        args.logger.info(
            f"{k}: The ground truth is {round(v,5)}, taking time {round(time_now-time_get_ground_truth,5)}."
        )
    return ground_truth


def HypothesisTesting(args, result_list, verbose=1):
    time_start_HT = time.time()

    if args.HTtype == "one-sample":
        # Test H1: avg rating = popmean
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
        data1 = result_list[str(list(args.attribute.keys())[0])]
        data2 = result_list[args.attribute[1]]
        t_stat, p_value = stats.levene(data1, data2, center="mean")
        print(f"p-value: {p_value} (< 0.05 means unequal variance).")
        args.logger.info(f"p-value: {p_value} (< 0.05 means unequal variance).")

        # hypothesis testing
        t_stat, p_value = stats.ttest_ind(
            a=data1, b=data2, equal_var=True, alternative=alternative
        )
        if verbose == 1:
            print_hypo_log(args, t_stat, p_value, args.H0)

    ht_time = round(time.time() - time_start_HT, 5)
    print(f">>> Time for hypothesis testing is {ht_time}.\n")
    args.logger.info(f">>> Time for hypothesis testing is {ht_time}.\n")
    args.time_result[args.ratio].append(ht_time)


def samplingGraph(args, graph):
    # initialize an empty result list and a dictionary for time tracking
    result_list = []
    time_used_list = defaultdict(list)
    # list of supported sampling methods
    supported_methods = [
        "RNNS",
        "SRW",
        "ShortestPathS",
        "MHRWS",
        "CommunitySES",
        "CNARW",
        "FFS",
        "SBS",
        "FrontierS",
        "NBRW",
        "RW_Starter",
        "RNS",
        "DBS",
        "PRBS",
        "RES",
        "RNES",
        "RES_Induction",
        "ours",
    ]

    # if the sampling method is supported, call the selected function and update result and time tracking
    if args.sampling_method in supported_methods:
        # sampling_function = globals()[args.sampling_method]
        result_list, time_used = sample_graph(
            args, graph, result_list, time_used_list, args.sampling_method
        )
    else:
        # log an error and raise an exception
        args.logger.error(f"Sorry, we don't support {args.sampling_method}.")
        raise Exception(f"Sorry, we don't support {args.sampling_method}.")

    # calculate and log the avg time for the sampling method
    time_one_sample = sum(time_used["sampling"]) / len(time_used["sampling"])
    print(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample, 2)}."
    )
    args.logger.info(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample, 2)}."
    )
    args.time_result[args.ratio].append(round(time_one_sample, 2))

    # calculate and log the avg time for target node/edge/path extraction
    time_extraction = sum(time_used["sample_graph_by_condition"]) / len(
        time_used["sample_graph_by_condition"]
    )
    print(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_extraction, 5)}."
    )
    args.logger.info(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_extraction, 5)}."
    )
    args.time_result[args.ratio].append(round(time_extraction, 5))

    return result_list


if __name__ == "__main__":
    # import cProfile

    # cProfile.run(
    #     "main()",
    #     filename="HypothesisTesting/log_and_results_2008/result.out",
    #     sort="cumulative",
    # )
    main()
