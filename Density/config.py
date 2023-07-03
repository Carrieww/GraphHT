import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default="2022", help="random seed.")
    parser.add_argument(
        "--dataset", type=str, default="facebook", help="dataset."
    )  # facebook, ca_GrQc
    parser.add_argument(
        "--bins", type=int, default=100, help="num of bins for getting cc dist."
    )
    # SRW, CNARW
    parser.add_argument(
        "--sampling_method", type=str, default="SRW", help="sampling method."
    )
    parser.add_argument(
        "--with_weight_func",
        type=bool,
        default=False,
        help="whether to have unbised estimator for the target property with weight function.",
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
        "--ratio_list",
        type=list,
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8],
        help="sampling size list.",
    )  # default [0.1,0.15,0.2,0.25,0.3,0.4,0.6,0.8]
    parser.add_argument(
        "--ratio_num",
        type=int,
        default=None,
        help="uniformly split [0,1] for ratio_num parts.",
    )

    # DegDistribution parameter
    parser.add_argument(
        "--n_deg",
        type=str,
        default=200,
        help="deg distribution will cover the top n degrees.",
    )

    return parser.parse_args()
