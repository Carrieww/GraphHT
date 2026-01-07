import os
import time
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
import pandas as pd
from littleballoffur import RandomWalkSampler
from networkx.algorithms import isomorphism

from config import parse_args
from hypothesis_testing import hypothesis_testing
from instance_extraction import getEdges, getNodes, getPaths, getSubgraphs
from samplers.existing_graph_samplers import subgraph_hypothesis_testing
from utils import clean, compute_accuracy, setup_device, setup_seed
from utils.dataloader import DataLoader
from utils.logging import Logger

SUPPORTED_SAMPLING_METHODS = [
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
    "PHASE",
    "Opt_PHASE",
    "TriangleS",
]


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)

    # Initialize logger
    logger = Logger(args)
    logger.log_global_info()

    dataloader = DataLoader(args)
    graph = dataloader.load()
    accept = getGroundTruth(args, graph)

    # Run sampling and hypothesis testing
    assert (
        args.sampling_method in SUPPORTED_SAMPLING_METHODS
    ), f"Sorry, we don't support {args.sampling_method}."

    if args.time_accuracy:
        args.sampling_ratio = list(range(20, args.num_nodes, 20))
        print(f"the list of sampling size is list(range(20, args.num_nodes, 20))")
        args.logger.info(
            f"the list of sampling size is list(range(20, args.num_nodes, 20))"
        )
    else:
        args.sampling_ratio = [
            int(args.num_nodes * (percent / 100)) for percent in args.sampling_percent
        ]
        print(f"the list of sampling size is {args.sampling_ratio}")
        args.logger.info(f"the list of sampling size is {args.sampling_ratio}")

    # Sampling and hypothesis testing
    args.time_result = defaultdict(list)
    acc_count = 0
    for ratio in args.sampling_ratio:
        # Sampling setup and execution
        time_ratio_start = time.time()
        args.ratio = ratio
        args.logger.info(f">>> Sampling Ratio: {args.ratio}")
        (
            HT_result_list,
            average_CI_lower,
            average_CI_upper,
            average_p_value,
            time_dict,
        ) = subgraph_hypothesis_testing(args, graph)

        args.time_result[args.ratio].append(average_CI_lower)
        args.time_result[args.ratio].append(average_CI_upper)
        args.time_result[args.ratio].append(average_p_value)

        # Calculate total time, average time for sampling, extraction
        time_ratio = round(time.time() - time_ratio_start, 2)
        args.logger.info(f">>> Total time for sampling is {time_ratio}.")
        args.time_result[args.ratio].append(time_ratio)

        time_sampling = round(
            sum(time_dict["sampling"]) / len(time_dict["sampling"]), 2
        )
        args.logger.info(f">>> Average time for sampling is {time_sampling}.")
        args.time_result[args.ratio].append(time_sampling)

        time_extraction = round(
            sum(time_dict["extraction"]) / len(time_dict["extraction"]), 2
        )
        args.logger.info(f">>> Average time for extraction is {time_extraction}.")
        args.time_result[args.ratio].append(time_extraction)

        # Calculate accuracy
        accuracy = compute_accuracy(args.ground_truth, HT_result_list)
        args.time_result[args.ratio].append(accuracy)
        args.logger.info(f">>> Accuracy of sampling result is {round(accuracy,4)}.")

        if args.time_accuracy and time_sampling > args.time_accuracy_time:
            args.logger.info(
                f"time for one sampling {time_sampling} reaches the limit {args.time_accuracy_time}."
            )
            break

        if args.time_accuracy and accuracy >= 1 and acc_count >= 3:
            args.logger.info(f"accuracy reaches 1 for at least three times.")
            break
        elif args.time_accuracy and accuracy >= 1 and acc_count < 3:
            acc_count += 1

    logger.save_results()


def getGroundTruth(args, graph):
    time_get_ground_truth = time.time()
    subgraph_type = args.hypothesis_pattern[args.condition_name]["type"]

    # Extract instances
    if "edge" in subgraph_type:
        instances = getEdges(args, graph)
    elif "node" in subgraph_type:
        instances = getNodes(args, graph)
    elif "path" in subgraph_type:
        instances = getPaths(args, graph)
    elif "subgraph" in subgraph_type:
        instances = getSubgraphs(args, graph)

    test_statistics, accept, confidence_interval, p_value = hypothesis_testing(
        args, instances, graph
    )
    args.logger.info(
        f"Test statistics: {test_statistics}, Accept: {accept}, Confidence interval: {confidence_interval}, P-value: {p_value}"
    )
    args.logger.info(
        f"Time for getting ground truth: {round(time.time() - time_get_ground_truth, 2)}"
    )

    return accept


if __name__ == "__main__":
    # import cProfile

    # cProfile.run(
    #     "main()",
    #     filename="HypothesisTesting/log_and_results_2008/result.out",
    #     sort="cumulative",
    # )
    main()
