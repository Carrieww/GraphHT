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
        default="citation",
        choices=["citation", "yelp", "movielens"],
        help="choose dataset from citation, yelp, or movielens.",
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
        default="TriangleS",
        choices=[
            "RNNS",
            "SRW",
            "FFS",
            "ShortestPathS",
            "MHRWS",
            "CommunitySES",
            "NBRW",
            "SBS",
            "RW_Starter",
            "FrontierS",
            "CNARW",
            "RNS",
            "DBS",
            "PRBS",
            "RES",
            "RNES",
            "RES_Induction",
            "PHASE",
            "Opt_PHASE",
            "TriangleS",
        ],
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

    ########## parameters for time accuracy plots ##########
    parser.add_argument(
        "--time_accuracy",
        type=bool,
        default=True,
        help="If False, then your input sampling percent will take effect. If True, the algo starts from 1/1000 nodes until time >= 30s or accuracy reaches 1",
    )

    parser.add_argument(
        "--time_accuracy_time",
        type=int,
        default=30,
        help="If time_accuracy is False, then your input sampling percent will take effect. If True, the algo starts from 1/1000 nodes until time >= time_accuracy_time (sec) or accuracy reaches 1",
    )

    ########## parameters for hypothesis ##########
    parser.add_argument(
        "--H0",
        type=str,
        default="The avg rating on triangle subgraph [business in FL - high popularity user - business in LA] is greater than 0.5",
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
        "--hypothesis_pattern",
        type=dict,
        default={
            "citation-triangle": {
                "type": "subgraph",  # "edge" | "node" | "path" | "subgraph"
                "subgraph": {
                    "nodes": [
                        {
                            "id": "n1",
                            "label": "fos",
                            "attribute": {"fos_name": "Combinatorics"},
                        },
                        {
                            "id": "n2",
                            "label": "paper",
                            "attribute": {"doc_type": "Conference"},
                        },
                        {
                            "id": "n3",
                            "label": "paper",
                            "attribute": {"doc_type": "Conference"},
                        },
                    ],
                    "edges": [
                        {"from": "n2", "to": "n1"},  # paper -> fos
                        {"from": "n3", "to": "n1"},  # paper -> fos
                        {"from": "n2", "to": "n3"},  # paper -> paper
                    ],
                },
                "anchor": {
                    "type": "edge",
                    "ids": ["n2", "n1"],  # edge from paper (n2) to fos (n1)
                    "Pi": 2,
                },
                "target": {
                    "node_diff": {
                        "nodes": [1, 2],  # indices of n2 and n3 (both paper nodes)
                        "attribute": "n_citation",
                    },
                },
                "test": {
                    "comparison": ">",  # "!=" | "==" | ">" | "<"
                    "c": 20,  # constant value in hypothesis
                    "agg": "mean",  # aggregation function
                },
            }
        },
        help="the hypothesis pattern to test on. "
        "Format: {'name': {'type': 'edge'|'node'|'path'|'subgraph', 'structure': {...}, 'target': {'edge': 'attr'|'node': {...}}, 'test': {'comparison': '>', 'c': 0.5, 'agg': 'mean'}}}. "
        "For path: {'name': {'type': 'path', 'path': [...], 'target': {'edge': 'attr'}, 'test': {...}}}; "
        "For subgraph: {'name': {'type': 'subgraph', 'subgraph': {'nodes': [...], 'edges': [...]}, 'target': {'edge': 'attr'}, 'test': {...}}}",
    )

    ### our sampler hyper-parameter
    parser.add_argument(
        "--alpha",
        type=int,
        default=0.95,
        help="significance level.",
    )

    args = parser.parse_args()

    # Extract hypothesis parameters from hypothesis_pattern
    # All hypothesis parameters must be defined in hypothesis_pattern
    if not args.hypothesis_pattern or len(args.hypothesis_pattern) == 0:
        raise ValueError("hypothesis_pattern is required and cannot be empty.")

    pattern_key = list(args.hypothesis_pattern.keys())[0]
    pattern = args.hypothesis_pattern[pattern_key]

    # Extract type (hypo)
    if "type" not in pattern:
        raise ValueError("hypothesis_pattern must contain a 'type' key.")
    type_mapping = {"edge": 1, "node": 2, "path": 3, "subgraph": 4}
    if pattern["type"] not in type_mapping:
        raise ValueError(
            f"Invalid type '{pattern['type']}'. Must be one of: {list(type_mapping.keys())}"
        )
    args.hypo = type_mapping[pattern["type"]]

    # Extract test parameters
    if "test" not in pattern:
        raise ValueError("hypothesis_pattern must contain a 'test' key.")
    test_config = pattern["test"]

    if "comparison" not in test_config:
        raise ValueError("test must contain a 'comparison' key.")
    args.comparison = test_config["comparison"]

    if "c" not in test_config:
        raise ValueError("test must contain a 'c' key.")
    args.c = test_config["c"]

    if "agg" not in test_config:
        raise ValueError("test must contain an 'agg' key.")
    args.agg = test_config["agg"]

    return args
