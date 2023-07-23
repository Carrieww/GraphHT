import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_device(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(args):
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))


def clean():
    torch.cuda.empty_cache()
    print("finished clean!")


def logger(args):
    # Create and configure logger
    if args.sampling_method == "SpikyBallS":
        if args.sampling_mode is None:
            raise Exception(
                "The sampling_mode must be provided for sampling_method SpikyBallS."
            )
        else:
            log_filepath = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "log_and_results_" + args.attribute[0],
                args.dataset
                + "_"
                + args.sampling_method
                + "_"
                + args.sampling_mode
                + "_"
                + args.agg
                + "_"
                + str(args.file_num)
                + "_log.log",
            )
    else:
        log_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results_" + args.attribute[0],
            args.dataset
            + "_"
            + args.sampling_method
            + "_"
            + args.agg
            + "_"
            + str(args.file_num)
            + "_log.log",
        )
    logging.basicConfig(
        filename=log_filepath, format="%(asctime)s %(message)s", filemode="w"
    )

    # Creating an object
    args.logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    args.logger.setLevel(logging.INFO)


def drawAllRatings(args, rating_summary):
    x = np.arange(0, args.num_samples, 1)
    plt.plot(x, [args.ground_truth] * args.num_samples, label=f"true {args.agg} rating")
    index = 1
    for ratio, rating in rating_summary.items():
        CI = list(args.CI[index - 1])
        plt.plot(x, rating, label=f"{args.agg} rating ({ratio})")
        plt.fill_between(x, CI[0], CI[1], alpha=index / 10)
        index += 1
    plt.xlabel("i-th sampled subgraph")
    plt.ylabel("average rating")
    plt.legend()
    plt.title(f"average rating ({args.dataset}) - {args.sampling_method}")
    if args.sampling_method == "SpikyBallS":
        args.fig_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results_" + args.attribute[0],
            args.dataset
            + "_"
            + args.sampling_method
            + "_"
            + args.sampling_mode
            + "_"
            + args.agg
            + "_"
            + str(args.file_num)
            + "_allResult.png",
        )
    else:
        args.fig_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results_" + args.attribute[0],
            args.dataset
            + "_"
            + args.sampling_method
            + "_"
            + args.agg
            + "_"
            + str(args.file_num)
            + "_allResult.png",
        )
    plt.savefig(args.fig_path)


# def drawAvgRating(args, avg_rating):
#     plt.plot([args.ground_truth_avg] * 50, label=f"true avg rating")
#     plt.plot(avg_rating, label=f"avg rating ({args.ratio})")
#     plt.xlabel("i-th sampled subgraph")
#     plt.ylabel("average rating")
#     plt.legend()
#     plt.title(f"average rating ({args.dataset}) - {args.sampling_method} {args.ratio}")
#     if args.sampling_method == "SpikyBallS":
#         args.fig_path = os.path.join(
#             os.path.dirname(os.path.realpath(__file__)),
#             "log_and_results_" + args.attribute[0],
#             args.dataset
#             + "_"
#             + args.sampling_method
#             + "_"
#             + args.sampling_mode
#             + "_"
#             + str(args.file_num)
#             + "_fig.png",
#         )
#     else:
#         args.fig_path = os.path.join(
#             os.path.dirname(os.path.realpath(__file__)),
#             "log_and_results_" + args.attribute[0],
#             args.dataset
#             + "_"
#             + args.sampling_method
#             + "_"
#             + str(args.file_num)
#             + "_fig.png",
#         )
#     plt.savefig(args.fig_path)
