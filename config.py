# movielens 1-1-1
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default="2022", help="random seed.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        choices=["DBLP", "yelp", "movielens"],
        help="choose dataset from DBLP, yelp, or movielens.",
    )
    parser.add_argument(
        "--file_num",
        type=str,
        default="output",
        help="to name log and result files for multi runs.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="RNS",
        help="sampling method.",
    )
    parser.add_argument(
        "--sampling_percent",
        type=list,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 1, 2.5, 5, 7.5, 10],
        help="list of sampling proportions.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="number of samples to draw from the input graph.",
    )

    ########## parameters for hypothesis ##########
    parser.add_argument(
        "--H0",
        type=str,
        default="The avg rating of action movies is greater than 3",
        help="The null hypothesis.",
    )
    parser.add_argument(
        "--HTtype",
        type=str,
        default="one-sample",
        choices=["one-sample"],
        help="We support one-sample hypothesis testing.",
    )
    parser.add_argument(
        "--attribute",
        type=dict,
        default={
            "1-1-1": {
                "edge": "rating",
                "movie": {"Action": 1},
                "path": [
                    {"type": "user", "attribute": {}},
                    {"type": "movie", "attribute": {"Action": 1}},
                ],
            }
        },
        help="the attributes you want to test hypothesis on.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        help="aggregation function.",
    )
    parser.add_argument(
        "--hypo",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1: edge hypothesis; 2: node hypothesis; 3: path hypothesis.",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=">",
        choices=["!=", "==", ">", "<"],
        help="comparison operator, choose from: !=, ==, >, <.",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=3,
        help="a constant value in the hypothesis.",
    )

    ### our sampler hyper-parameter
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.95,
        help="significance level.",
    )

    args = parser.parse_args()
    return args
