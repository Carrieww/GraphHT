import argparse


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
        "--sampling_method", type=str, default="CommunitySES", help="sampling method."
    )
    # parser.add_argument(
    #     "--with_weight_func",
    #     type=bool,
    #     default=False,
    #     help="whether to have unbised estimator for the target property with weight function.",
    # )
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
        default=[0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2],
        help="sampling size list.",
    )  # default [0.02,0.04,0.06,0.08,0.1,0.15,0.2],[0.2, 0.4, 0.6, 0.8, 0.9]
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="number of samples to draw from the original graph.",
    )

    # parameters for hypothesis
    # now only support one-sample hypothesis on attributes
    parser.add_argument(
        "--attribute",
        type=list,
        default=["Fantasy"],
        help="genre attributes you want to test hypothesis on.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default="mean",
        help="choosing from: mean, man, min.",
    )

    return parser.parse_args()
