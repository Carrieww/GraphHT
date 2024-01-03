import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default="2022", help="random seed.")
    parser.add_argument("--dataset", type=str, default="yelp", help="dataset.")
    parser.add_argument(
        "--file_num",
        type=int,
        default=30,
        help="to name log and result files for multi runs.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="RNS",
        help="sampling method.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="_auto",
        help="checkpoint filename suffix.",
    )
    parser.add_argument(
        "--sampling_percent",
        type=list,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1,
            2.5,
            5,
            7.5,
            10,
        ],  # \t1000\t1500\t2000\t2500\t3000\t5000
        help="Tab-separated list of sampling values.",
    )
    # sample size parameter
    parser.add_argument(
        "--sampling_ratio",
        type=str,
        default="auto",
        help="Tab-separated list of sampling values.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="number of samples to draw from the original graph.",
    )

    # parameters for hypothesis
    # now only support one-sample hypothesis on attributes
    parser.add_argument(
        "--H0",
        type=str,
        default="avg number of reviews of business in LA is greater than 70",
        help="The null hypothesis.",
    )
    parser.add_argument(
        "--HTtype",
        type=str,
        default="one-sample",
        help="Choose from one-sample or two-sample.",
    )
    parser.add_argument(
        "--attribute",
        type=dict,
        default={
            "2-1-1": {
                "dimension": {"business": "review_count"},
                "business": {"state": "LA"},
                "path": [
                    {"type": "user", "attribute": {}},
                    {"type": "business", "attribute": {"state": "LA"}},
                ],
            }
        },
        help="The attributes you want to test hypothesis on.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        help="choosing from: mean, max, min, number, variance",
    )
    parser.add_argument(
        "--hypo",
        type=int,
        default=2,
        help="1: edge hypothesis; 2: node hypothesis; 3: path hypothesis.",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=">",
        help="choosing from: !=, ==, >, <",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=70,
        help="a constant value in the hypothesis",
    )

    ### our sampler hyper-parameter
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.95,
        help="an integer",
    )

    args = parser.parse_args()
    return args
