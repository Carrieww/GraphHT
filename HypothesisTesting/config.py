import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default="2022", help="random seed.")
    parser.add_argument("--dataset", type=str, default="movielens", help="dataset.")
    parser.add_argument(
        "--file_num",
        type=int,
        default=99,
        help="to name log and result files for multi runs.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="RES",
        help="sampling method.",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=None,
        help="sampling mode for `SpikyBallS` sampling method.",
    )  # "edgeball", "hubball", "coreball", "fireball", "firecoreball"
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="_log",
        help="checkpoint filename suffix.",
    )
    # sample size parameter
    parser.add_argument(
        "--sampling_ratio",
        type=str,
        default="500\t1000\t1500\t2000\t5000\t10000",  # \t1000\t1500\t2000\t2500\t3000\t5000
        help="Tab-separated list of sampling values.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="number of samples to draw from the original graph.",
    )

    # parameters for hypothesis
    # now only support one-sample hypothesis on attributes
    parser.add_argument(
        "--attribute",
        type=dict,
        default={
            "3-2-1": {
                "edge": "rating",
                "path": [
                    {"type": "movie", "attribute": {"Thriller": 1}},
                    {"type": "user", "attribute": {}},
                    {"type": "movie", "attribute": {"Crime": 1}},
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
        default=3,
        help="choosing from: 1, 2...",
    )
    # movielens
    # hypo 1 (edge attribute): avg rating of adventure movies > 3.6
    # hypo 2 (node attribute): avg number of genres a movie has is > 5
    # hypo 3 (path): avg rating of on path Action-user-Documentary

    # # citation
    # hypo 1 (edge attribute): avg correlation score of papers in 2008 with its related papers > 3.6
    # hypo 2 (node attribute): avg citation of paper in 2008 > 2.6
    # hypo 3 (path): path

    parser.add_argument(
        "--comparison",
        type=str,
        default=">",
        help="choosing from: !=, ==, >, <",
    )

    args = parser.parse_args()
    if args.sampling_ratio:
        args.sampling_ratio = [int(value) for value in args.sampling_ratio.split("\t")]
        # Now, sampling_list is a Python list containing the values
        # print("Sampling list:", sampling_list)
    return args
