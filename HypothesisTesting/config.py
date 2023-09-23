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
        default=100,
        help="to name log and result files for multi runs.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="MHRWS",
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
        type=list,
        default=[
            85,
            120,
            155,
            180,
            400,
            700,
        ],  # [5000, 10000, 15000, 20000, 45000, 85000, 500000],  #
        help="sampling size list.",
    )  # default [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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
            "1-4": {"edge": "rating", "user": {"gender": "F"}}
        },  # , "article": {"Adventure": 1}}},
        help="The attributes you want to test hypothesis on.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        help="choosing from: mean, max, min, number, varaince",
    )
    parser.add_argument(
        "--hypo",
        type=int,
        default=3,
        help="choosing from: 1, 2...",
    )
    # movielens
    # hypo 1 (degree): avg number of advanture movies rated by users > 80
    # hypo 2 (node attribute): avg number of genres a movie has is > 5
    # hypo 3 (edge attribute): avg rating of advanture movies > 3.6
    # hypo 4 (variance): the variance/sd of advanture movie ratings < 0.01

    # hypo 10 (Node attribute on edge attribute): Adventure movies receive higher ratings than Comedy movies.

    # # citation
    # hypo 1 (degree): avg authors of paper in 2008 > 2.6
    # hypo 2 (node attribute): avg citation of paper in 2008 > 2.6
    # hypo 3 (edge attribute): avg correlation score of papers in 2008 with its related papers > 3.6
    # hypo 4 (triangle): number of triangles > 2.6
    # hypo 5: number of author1-paper1-paper2-author2 > 2.6

    # hypo 10 (node attribute): Avg citations of papers in 2008 is higher than that in 1962

    parser.add_argument(
        "--comparison",
        type=str,
        default=">",
        help="choosing from: !=, ==, >, <",
    )

    return parser.parse_args()
