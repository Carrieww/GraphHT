import logging
import os
import random

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
    args.property = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    if args.sampling_method == "SpikyBallS":
        if args.sampling_mode is None:
            raise Exception(
                "The sampling_mode must be provided for sampling_method SpikyBallS."
            )
        else:
            log_filepath = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "log_and_results",
                args.property
                + "_"
                + args.dataset
                + "_"
                + args.sampling_method
                + "_"
                + args.sampling_mode
                + "_log.log",
            )
    else:
        log_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "log_and_results",
            args.property
            + "_"
            + args.dataset
            + "_"
            + args.sampling_method
            + "_log.log",
        )
    logging.basicConfig(
        filename=log_filepath, format="%(asctime)s %(message)s", filemode="w"
    )

    # Creating an object
    args.logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    args.logger.setLevel(logging.INFO)
