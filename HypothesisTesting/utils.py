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
    if len(args.attribute) == 1:
        args.log_folderPath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "one_sample",
            "log_and_results_" + str(args.attribute[0]),
        )
    else:
        args.log_folderPath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "two_sample",
            "log_and_results_" + str(args.dataset),
        )
    if args.sampling_method == "SpikyBallS":
        if args.sampling_mode is None:
            args.logger.error(
                "The sampling_mode must be provided for sampling_method SpikyBallS."
            )
            raise Exception(
                "The sampling_mode must be provided for sampling_method SpikyBallS."
            )
        else:
            if len(args.attribute) == 1:
                string = str(args.attribute[0][:4])
            else:
                string = str(args.attribute[0][:4]) + str(args.attribute[1][:4])
            log_filepath = os.path.join(
                args.log_folderPath,
                args.dataset
                + "_hypo"
                + str(args.hypo)
                + "_"
                + string
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
        if len(args.attribute) == 1:
            string = str(args.attribute[0][:4])
        else:
            string = str(args.attribute[0][:4]) + str(args.attribute[1][:4])
        log_filepath = os.path.join(
            args.log_folderPath,
            args.dataset
            + "_hypo"
            + str(args.hypo)
            + "_"
            + string
            + "_"
            + args.sampling_method
            + "_"
            + args.agg
            + "_"
            + str(args.file_num)
            + "_log.log",
        )

    if not os.path.exists(args.log_folderPath):
        os.makedirs(args.log_folderPath)

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
        if np.isnan(CI[0]) or np.isnan(CI[1]):
            pass
        else:
            plt.fill_between(x, CI[0], CI[1], alpha=index / 10)
        index += 1
    plt.xlabel("i-th sampled subgraph")
    plt.ylabel("average rating")
    plt.legend()
    plt.title(f"average rating ({args.dataset}) - {args.sampling_method}")
    if args.sampling_method == "SpikyBallS":
        if len(args.attribute) == 1:
            string = str(args.attribute[0][:4])
        else:
            string = str(args.attribute[0][:4]) + str(args.attribute[1][:4])
        args.fig_path = os.path.join(
            args.log_folderPath,
            args.dataset
            + "_hypo"
            + str(args.hypo)
            + "_"
            + string
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
        if len(args.attribute) == 1:
            string = str(args.attribute[0][:4])
        else:
            string = str(args.attribute[0][:4]) + str(args.attribute[1][:4])
        args.fig_path = os.path.join(
            args.log_folderPath,
            args.dataset
            + "_hypo"
            + str(args.hypo)
            + "_"
            + string
            + "_"
            + args.sampling_method
            + "_"
            + args.agg
            + "_"
            + str(args.file_num)
            + "_allResult.png",
        )
    plt.savefig(args.fig_path)


def print_hypo_log(args, t_stat, p_value, H0, **kwargs):
    args.logger.info("")
    args.logger.info("[Hypothesis Testing Results]")

    if args.HTtype == "one-sample":
        assert len(kwargs) == 1, f"Only one kwargs is allowed! eg twoSide or oneSide"
        for key, value in kwargs.items():
            if key == "twoSides":
                args.logger.info(f"H0: {H0} == {args.ground_truth}.")
                args.logger.info(f"H1: {H0} != {args.ground_truth}.")
            elif key == "oneSide":
                if value == "lower":
                    args.logger.info(f"H0: {H0} = {args.popmean_lower}.")
                    args.logger.info(f"H1: {H0} > {args.popmean_lower}.")
                elif value == "higher":
                    args.logger.info(f"H0: {H0} = {args.popmean_higher}.")
                    args.logger.info(f"H1: {H0} < {args.popmean_higher}.")
                else:
                    args.logging.error(f"Sorry, we don't support {value} for {key}.")
                    raise Exception(f"Sorry, we don't support {value} for {key}.")
            else:
                args.logging.error(f"Sorry, we don't support {key}.")
                raise Exception(f"Sorry, we don't support {key}.")
    else:
        args.logger.info(f"H0: {H0}.")

    args.logger.info(f"T-statistic value: {t_stat}, P-value: {p_value}.")
    if p_value < 0.05:
        args.logger.info(
            f"The test is significant, we shall reject the null hypothesis."
        )
    else:
        args.logger.info(
            f"The test is NOT significant, we shall accept the null hypothesis."
        )

    return 0
