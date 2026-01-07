import os
import time
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
import pandas as pd
from littleballoffur import RandomWalkSampler
from networkx.algorithms import isomorphism

from config import parse_args
from extraction import getEdges, getNodes, getPaths, getSubgraphs
from sampling import sample_graph
from utils import (
    HypothesisTesting,
    clean,
    compute_accuracy,
    get_data,
    log_global_info,
    logger,
    setup_device,
    setup_seed,
)


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    logger(args)
    log_global_info(args)

    graph = prepare_dataset(args)
    run_sampling_and_hypothesis_testing(args, graph)
    print_results(args)


def run_sampling_and_hypothesis_testing(args, graph):
    # sample for each sampling ratio
    args.overall_time = time.time()
    args.coverage = defaultdict(list)

    if args.time_accuracy:
        args.sampling_ratio = list(range(20, args.num_nodes, 20))
        print(f"the list of sampling size is list(range(20, args.num_nodes, 20))")
        args.logger.info(
            f"the list of sampling size is list(range(20, args.num_nodes, 20))"
        )
    else:
        args.sampling_ratio = [
            int(args.num_nodes * (percent / 100)) for percent in args.sampling_percent
        ]
        print(f"the list of sampling size is {args.sampling_ratio}")
        args.logger.info(f"the list of sampling size is {args.sampling_ratio}")

    args.time_result = defaultdict(list)
    acc_count = 0
    for ratio in args.sampling_ratio:
        # sampling setup and execution
        args.valid_edges = []
        args.variance = []
        time_ratio_start = time.time()
        args.ratio = ratio
        args.logger.info(f">>> sampling ratio: {args.ratio}")
        result_list, t_sample = samplingGraph(args, graph)

        valid_e_n = round(sum(args.valid_edges) / len(args.valid_edges), 2)
        print(f"average valid nodes/edges are {valid_e_n}")
        args.logger.info(f"average valid nodes/edges are {valid_e_n}")

        if hasattr(args, "variance"):
            if len(args.variance) != 0:
                print(f"average variance is {sum(args.variance)/len(args.variance)}")
                args.logger.info(
                    f"average variance is {sum(args.variance)/len(args.variance)}"
                )

        args.time_result[args.ratio].append(valid_e_n)

        # print total time used
        total_time = time.time() - time_ratio_start
        total_time_format = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        args.logger.info(
            f">>> Total time for sampling at {args.ratio} ratio is {total_time_format}."
        )
        accuracy = get_results(args, result_list)
        args.logger.info(f"accuracy here is {accuracy}")

        if args.time_accuracy and t_sample > args.time_accuracy_time:
            args.logger.info(
                f"time for one sampling {t_sample} reaches the limit {args.time_accuracy_time}."
            )
            break

        if args.time_accuracy and accuracy >= 1 and acc_count >= 3:
            args.logger.info(f"accuracy reaches 1 for at least three times.")
            break
        elif args.time_accuracy and accuracy >= 1 and acc_count < 3:
            acc_count += 1


def get_results(args, result_list):
    """
    Getting all results after sampling for printing.
    """
    if args.HTtype == "one-sample":
        if args.hypo == 3:
            user_cov_list = [
                i[str(list(args.hypothesis_pattern.keys())[0]) + "+user_coverage"][0]
                for i in result_list
            ]
            movie_cov_list = [
                i[str(list(args.hypothesis_pattern.keys())[0]) + "+movie_coverage"][0]
                for i in result_list
            ]
            density_list = [i["density"] for i in result_list]
            diameter_list = [i["diameter"] for i in result_list]
            total_valid_path_list = [i["total_valid"] for i in result_list]
            total_valid_path_minus_reverse_list = [
                i["total_minus_reverse"] for i in result_list
            ]

            density = sum(density_list) / len(density_list)
            diameter = sum(diameter_list) / len(diameter_list)
            total_valid = sum(total_valid_path_list) / len(total_valid_path_list)
            total_valid_path_minus_reverse = sum(
                total_valid_path_minus_reverse_list
            ) / len(total_valid_path_minus_reverse_list)

            user_coverage_avg = round(sum(user_cov_list) / len(user_cov_list), 3)
            movie_coverage_avg = round(sum(movie_cov_list) / len(movie_cov_list), 3)
            args.coverage[args.ratio].extend(
                [
                    user_coverage_avg,
                    movie_coverage_avg,
                    round(total_valid, 3),
                    round(total_valid_path_minus_reverse, 3),
                    round(density, 3),
                    round(diameter, 3),
                ]
            )
            print(
                f">>> Diameter of sampling result at {args.ratio} sampling ratio is {round(diameter,3)}."
            )
            args.logger.info(
                f">>> Diameter of sampling result at {args.ratio} sampling ratio is {round(diameter,3)}."
            )

        result = [i[str(list(args.hypothesis_pattern.keys())[0])] for i in result_list]

        # compute accuracy
        accuracy = compute_accuracy(args.ground_truth, result)
        print(
            f">>> Accuracy of sampling result is {round(accuracy,4)} at {args.ratio} sampling ratio."
        )
        args.logger.info(
            f">>> Accuracy of sampling result is {round(accuracy,4)} at {args.ratio} sampling ratio."
        )

        args.time_result[args.ratio].append(round(accuracy, 2))

        args.logger.info(
            f"The hypothesis testing for {args.ratio} sampling ratio is finished!"
        )
    else:
        raise Exception(
            "Sorry we do not support hypothesis types other than one-sample."
        )
    return accuracy


def print_results(args):
    """
    Printing the results in log and excel.
    """
    headers = [
        "Sampling time",
        "Target extraction time",
        "Total Time",
        "Accuracy",
        "node num",
        "Valid nodes/edges/paths",
        "Confidence Interval Lower",
        "Confidence Interval Upper",
        "p-value",
    ]

    # Print headers
    header_format = " | ".join([header.title().ljust(25) for header in headers])
    print(header_format)
    args.logger.info(header_format)

    list_valid = []
    txt_filepath = "_".join(args.log_filepath.split("_")[:-1]) + ".txt"
    with open(txt_filepath, "w") as file:
        file.write(
            "Sampling Time\tTarget Extraction Time\tTotal Time\tAccuracy\tSampling Ratio\tValid Nodes Edges Paths\tLower CI\tUpper CI\tp-value\n"
        )
    for index, (ratio, value) in enumerate(args.time_result.items()):
        (
            CI_lower,
            CI_upper,
            p_value,
            sampling_time,
            target_extraction_time,
            valid_nodes_edges_paths,
            accuracy,
        ) = value
        total_time = round(target_extraction_time + sampling_time, 2)
        list_valid.append(valid_nodes_edges_paths)

        # Print the results
        result_format = (
            f"{sampling_time:.2f}".ljust(25)
            + f"{target_extraction_time:.2f}".ljust(25)
            + f"{total_time:.2f}".ljust(25)
            + f"{accuracy:.2f}".ljust(25)
            + f"{args.sampling_ratio[index]}".ljust(25)
            + f"{valid_nodes_edges_paths:.2f}".ljust(25)
            + f"{CI_lower:.2f}".ljust(25)
            + f"{CI_upper:.2f}".ljust(25)
            + f"{p_value:.2f}".ljust(25)
        )
        print(result_format)
        args.logger.info(result_format)

        # Open a file in write mode
        with open(txt_filepath, "a") as file:
            # Write the headers if needed
            file.write(
                f"{sampling_time:.2f}\t{target_extraction_time:.2f}\t{total_time:.2f}\t{accuracy:.2f}\t{args.sampling_ratio[index]}\t{valid_nodes_edges_paths:.2f}\t{CI_lower:.2f}\t{CI_upper:.2f}\t{p_value:.2f}\n"
            )

    if args.hypo == 3:
        summary_statistics_headers = [
            "User Coverage",
            "Movie Coverage",
            "Total Valid Paths",
            "Reverse Paths",
            "Self-Loops",
            "Density",
            "Diameter",
        ]

        # Print hypothesis headers
        summary_statistics_header_format = " | ".join(
            [header.title().ljust(25) for header in summary_statistics_headers]
        )
        print(summary_statistics_header_format)
        args.logger.info(summary_statistics_header_format)

        for index, (ratio, value) in enumerate(args.coverage.items()):
            (
                user_coverage,
                movie_coverage,
                total_valid_paths,
                total_without_reverse_paths,
                density,
                diameter,
            ) = value
            num_reverse_paths = round(
                total_valid_paths - total_without_reverse_paths, 3
            )
            num_self_loops = round(total_without_reverse_paths - list_valid[index], 3)

            # Print the hypothesis results
            hypothesis_result_format = (
                f"{user_coverage:.3f}".ljust(25)
                + f"{movie_coverage:.3f}".ljust(25)
                + f"{total_valid_paths:.3f}".ljust(25)
                + f"{num_reverse_paths:.3f}".ljust(25)
                + f"{num_self_loops:.3f}".ljust(25)
                + f"{density:.3f}".ljust(25)
                + f"{diameter:.3f}".ljust(25)
            )

            print(hypothesis_result_format)
            args.logger.info(hypothesis_result_format)

    print(
        f"All hypothesis testing for ratio list {args.sampling_ratio} and plotting is finished!"
    )


def prepare_dataset(args):
    # get the graph
    time_dataset_prep = time.time()
    args.dataset_path = os.path.join(os.getcwd(), "datasets", args.dataset)
    graph = get_data(args)

    # Reduce graph size for citation network demonstration
    if args.dataset == "citation":
        original_num_nodes = graph.number_of_nodes()
        original_num_edges = graph.number_of_edges()
        args.logger.info(
            f"Original citation network has {original_num_nodes} nodes and {original_num_edges} edges."
        )
        print(
            f"Original citation network has {original_num_nodes} nodes and {original_num_edges} edges."
        )

        # TODO
        # Sample 1000 nodes using RandomWalkSampler for demonstration
        sample_size = 1000
        args.logger.info(
            f"Sampling {sample_size} nodes from citation network using RandomWalkSampler for demonstration..."
        )
        print(
            f"Sampling {sample_size} nodes from citation network using RandomWalkSampler for demonstration..."
        )

        sampler = RandomWalkSampler(number_of_nodes=sample_size, seed=int(args.seed))
        graph = sampler.sample(graph)

        # Re-index nodes to ensure continuous integer indices (0, 1, 2, ...)
        # This is required by littleballoffur library
        graph = nx.relabel.convert_node_labels_to_integers(
            graph, first_label=0, ordering="default"
        )

        sampled_num_nodes = graph.number_of_nodes()
        sampled_num_edges = graph.number_of_edges()
        args.logger.info(
            f"Sampled graph has {sampled_num_nodes} nodes and {sampled_num_edges} edges."
        )
        print(
            f"Sampled graph has {sampled_num_nodes} nodes and {sampled_num_edges} edges."
        )

        analyze_graph(graph, args.hypothesis_pattern)

    print(
        f">>> Total time for dataset {args.dataset} preparation is {round((time.time() - time_dataset_prep),2)}."
    )
    args.logger.info(
        f">>> Total time for dataset {args.dataset} preparation is {round((time.time() - time_dataset_prep),2)}."
    )

    # graph characteristics, ground truth setup
    args.num_nodes = graph.number_of_nodes()
    args.num_edges = graph.number_of_edges()
    is_directed = graph.is_directed()
    args.logger.info(
        f"{args.dataset} has {args.num_nodes} nodes and {args.num_edges} edges."
    )
    args.logger.info(f"{args.dataset} is directed: {is_directed}")
    args.logger.info(f"{args.dataset} is connected: {nx.is_connected(graph)}.")
    print(f"{args.dataset} graph type: {'Directed' if is_directed else 'Undirected'}")
    print(f"{args.dataset} is connected: {nx.is_connected(graph)}")

    # # Find all triangles in the graph using VF2
    # triangles = find_all_triangles_vf2(graph)
    # args.logger.info(f"Found {len(triangles)} unique triangles in the graph.")
    # print(f"Found {len(triangles)} unique triangles in the graph.")

    # # Display attributes of first few triangles
    # if triangles:
    #     args.logger.info("Sample triangle attributes:")
    #     print("\nSample triangle attributes:")
    #     for i, triangle in enumerate(triangles[:5]):  # Show first 5 triangles
    #         triangle_attrs = extract_triangle_attributes(graph, triangle)
    #         args.logger.info(
    #             f"Triangle {i+1}: {triangle} - Attributes: {triangle_attrs}"
    #         )
    #         print(
    #             f"  Triangle {i+1}: {sorted(triangle)} - Attributes: {triangle_attrs}"
    #         )

    if args.dataset in ["movielens", "citation", "yelp"]:
        args.ground_truth = getGroundTruth(args, graph)
    else:
        raise Exception(f"Sorry we do not support {args.dataset} dataset.")

    if args.HTtype == "one-sample":
        args.ground_truth = args.ground_truth[
            str(list(args.hypothesis_pattern.keys())[0])
        ]
    else:
        raise Exception("Sorry we do not support other HTtype.")

    return graph


def find_all_triangles_vf2(graph: nx.Graph) -> List[Set[int]]:
    """
    Find all unique triangles in the graph using VF2 isomorphism algorithm.

    Args:
        graph: The graph to search in

    Returns:
        List of unique triangles, each represented as a set of 3 node IDs
    """
    # Build a triangle pattern graph (3 nodes forming a triangle)
    if graph.is_directed():
        pattern_graph = nx.DiGraph()
    else:
        pattern_graph = nx.Graph()

    # Add 3 nodes forming a triangle
    pattern_graph.add_edge(0, 1)
    pattern_graph.add_edge(1, 2)
    pattern_graph.add_edge(2, 0)

    # Use VF2 matcher (no attribute matching, just structure)
    if graph.is_directed():
        matcher = isomorphism.DiGraphMatcher(graph, pattern_graph)
    else:
        matcher = isomorphism.GraphMatcher(graph, pattern_graph)

    # Find all subgraph isomorphisms
    matches = list(matcher.subgraph_isomorphisms_iter())

    # Extract unique triangles (as sets to avoid duplicates)
    unique_triangles = set()
    for match in matches:
        # match is a dict: {graph_node_id: pattern_node_id}
        # Keys are graph nodes, values are pattern nodes (0, 1, 2)
        triangle_nodes = set(match.keys())  # Get graph node IDs
        if len(triangle_nodes) == 3:
            # Use frozenset for hashing
            unique_triangles.add(frozenset(triangle_nodes))

    # Convert back to list of sets
    return [set(triangle) for triangle in unique_triangles]


def extract_triangle_attributes(graph: nx.Graph, triangle: Set[int]) -> Dict:
    """
    Extract attributes of nodes and edges in a triangle.

    Args:
        graph: The graph
        triangle: Set of 3 node IDs forming the triangle

    Returns:
        Dictionary containing node and edge attributes
    """
    triangle_list = list(triangle)
    attrs = {"nodes": {}, "edges": {}}

    # Extract node attributes
    for node_id in triangle_list:
        node_attrs = dict(graph.nodes[node_id])
        attrs["nodes"][node_id] = node_attrs

    # Extract edge attributes
    for i in range(len(triangle_list)):
        for j in range(i + 1, len(triangle_list)):
            u, v = triangle_list[i], triangle_list[j]
            if graph.has_edge(u, v):
                edge_attrs = dict(graph.edges[u, v])
                edge_key = tuple(sorted([u, v])) if not graph.is_directed() else (u, v)
                attrs["edges"][edge_key] = edge_attrs

    return attrs


def getGroundTruth(args, graph):
    time_get_ground_truth = time.time()
    dict_result = {}

    # define hypothesis and data processing for each dataset
    # 1: edge hypo; 2: node hypo; 3: path hypo; 4: subgraph hypo
    # Check if the dataset is supported
    if args.dataset not in {"movielens", "citation", "yelp"}:
        args.logger.error(f"Sorry, {args.dataset} is not supported.")
        raise Exception(f"Sorry, {args.dataset} is not supported.")

    attribute_key = str(list(args.hypothesis_pattern.keys())[0])

    if args.hypo == 1:
        dict_result[attribute_key] = getEdges(args, graph)[attribute_key]
    elif args.hypo == 2:
        args.dimension = args.hypothesis_pattern[attribute_key]["dimension"]
        dict_result[attribute_key] = getNodes(args, graph, dimension=args.dimension)[
            attribute_key
        ]
    elif args.hypo == 3:
        args.total_valid = 0
        args.total_minus_reverse = 0
        dict_result[attribute_key] = getPaths(args, graph)[attribute_key]
    elif args.hypo == 4:
        # Subgraph hypothesis using VF2 isomorphism matching
        dict_result[attribute_key] = getSubgraphs(args, graph)[attribute_key]
    else:
        args.logger.error(f"Sorry, {args.hypo} is not supported for {args.dataset}.")
        raise Exception(f"Sorry, {args.hypo} is not supported for {args.dataset}.")

    # check if there are valid nodes/edges/paths
    if len(dict_result[str(list(args.hypothesis_pattern.keys())[0])]) == 0:
        args.logger.error(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )
        raise Exception(
            f"The graph contains no node/edge satisfying the hypothesis, you may need to change the attribute."
        )
    print(
        f"total number of valid nodes/edges {len(dict_result[str(list(args.hypothesis_pattern.keys())[0])])}"
    )
    args.logger.info(
        f"total number of valid nodes/edges {len(dict_result[str(list(args.hypothesis_pattern.keys())[0])])}"
    )

    # compute the ground truth based on the aggregation method
    ground_truth = {}
    for k, v in dict_result.items():
        if args.agg == "mean":
            ground_truth_result = HypothesisTesting(args, v, 1)
            avg_v = round(sum(v) / len(v), 2)
            args.ground_truth_value = avg_v
            args.logger.info(
                f"{k}: The ground truth ({avg_v}) result is {ground_truth_result}, taking time {round(time.time()-time_get_ground_truth, 2)}."
            )
            print(
                f"{k}: The ground truth ({avg_v}) result is {ground_truth_result}, taking time {round(time.time()-time_get_ground_truth, 2)}."
            )
            ground_truth[k] = ground_truth_result  # sum(v) / len(v)
        else:
            args.logger.error(f"Sorry, we don't support {args.agg}.")
            raise Exception(f"Sorry, we don't support {args.agg}.")

    return ground_truth


def samplingGraph(args, graph):
    # initialize an empty result list and a dictionary for time tracking
    result_list = []
    time_used_list = defaultdict(list)
    # list of supported sampling methods
    supported_methods = [
        "RNNS",
        "SRW",
        "ShortestPathS",
        "MHRWS",
        "CommunitySES",
        "CNARW",
        "FFS",
        "SBS",
        "FrontierS",
        "NBRW",
        "RW_Starter",
        "RNS",
        "DBS",
        "PRBS",
        "RES",
        "RNES",
        "RES_Induction",
        "PHASE",
        "Opt_PHASE",
        "TriangleS",
    ]

    # if the sampling method is supported, call the selected function and update result and time tracking
    if args.sampling_method in supported_methods:
        # sampling_function = globals()[args.sampling_method]
        result_list, time_used = sample_graph(
            args, graph, result_list, time_used_list, args.sampling_method
        )
    else:
        # log an error and raise an exception
        args.logger.error(f"Sorry, we don't support {args.sampling_method}.")
        raise Exception(f"Sorry, we don't support {args.sampling_method}.")

    # calculate and log the avg time for the sampling method
    time_one_sample = sum(time_used["sampling"]) / len(time_used["sampling"])
    print(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample, 2)}."
    )
    args.logger.info(
        f">>> Avg time for sampling {args.sampling_method} at {args.ratio} sampling ratio one time is {round(time_one_sample, 2)}."
    )
    args.time_result[args.ratio].append(round(time_one_sample, 2))

    # calculate and log the avg time for target node/edge/path extraction
    time_extraction = sum(time_used["sample_graph_by_condition"]) / len(
        time_used["sample_graph_by_condition"]
    )
    print(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_extraction, 5)}."
    )
    args.logger.info(
        f">>> Avg time for target node/edge extraction at {args.ratio} sampling ratio one time is {round(time_extraction, 5)}."
    )
    args.time_result[args.ratio].append(round(time_extraction, 5))

    return result_list, round(time_one_sample, 2)


def analyze_graph(graph, hypothesis_pattern):
    """
    Analyze the graph by keeping only nodes and edges that match the hypothesis pattern.
    Args:
        graph: The graph to analyze
        hypothesis_pattern: The hypothesis pattern to analyze
    Returns:
        None
    """
    if len(list(hypothesis_pattern)) != 1:
        raise ValueError("analyze_graph only supports one hypothesis pattern")

    # Extract pattern from hypothesis_pattern
    pattern_key = list(hypothesis_pattern.keys())[0]
    pattern = hypothesis_pattern[pattern_key]

    if pattern.get("type") != "subgraph":
        raise ValueError("analyze_graph only supports subgraph patterns")

    if "subgraph" not in pattern:
        raise ValueError("Subgraph pattern must contain 'subgraph' key")

    subgraph_pattern = pattern["subgraph"]
    pattern_nodes = subgraph_pattern["nodes"]
    pattern_edges = subgraph_pattern["edges"]

    # Helper function to check if a graph node matches a pattern node
    def node_matches_pattern(graph_node_data, pattern_node_info):
        """
        Check if a graph node matches a pattern node (label and attributes).

        Args:
            graph_node_data: Dictionary of graph node attributes
            pattern_node_info: Dictionary from pattern with 'label' and 'attribute' keys

        Returns:
            True if node matches, False otherwise
        """
        pattern_label = pattern_node_info.get("label")
        pattern_attrs = pattern_node_info.get("attribute", {})

        # Check node label
        if graph_node_data.get("label") != pattern_label:
            return False

        # Check node attributes
        for attr_key, attr_val in pattern_attrs.items():
            graph_val = graph_node_data.get(attr_key)
            if pd.isna(graph_val):
                return False
            # Support vague match for string attributes (like citation dataset)
            if graph_val != attr_val and (
                attr_val not in graph_val if isinstance(graph_val, str) else True
            ):
                return False

        return True

    # Extract edge patterns: (from_label, to_label) pairs
    # Create a mapping from pattern node id to label
    node_id_to_label = {}
    for node_info in pattern_nodes:
        node_id = node_info.get("id")
        label = node_info.get("label")
        if node_id and label:
            node_id_to_label[node_id] = label

    # Extract allowed edge patterns
    allowed_edge_patterns = set()
    for edge_info in pattern_edges:
        from_id = edge_info.get("from")
        to_id = edge_info.get("to")
        if from_id in node_id_to_label and to_id in node_id_to_label:
            from_label = node_id_to_label[from_id]
            to_label = node_id_to_label[to_id]
            allowed_edge_patterns.add((from_label, to_label))
            # For undirected graphs, also allow reverse direction
            if not graph.is_directed():
                allowed_edge_patterns.add((to_label, from_label))

    # Filter nodes: keep only nodes that match pattern nodes (label AND attributes)
    nodes_to_keep = []
    for node in graph.nodes():
        graph_node_data = graph.nodes[node]
        # Check if this node matches any pattern node
        for pattern_node_info in pattern_nodes:
            if node_matches_pattern(graph_node_data, pattern_node_info):
                nodes_to_keep.append(node)
                break  # Node matches at least one pattern, no need to check others

    # Create filtered subgraph with allowed nodes
    filtered_graph = graph.subgraph(nodes_to_keep).copy()

    # Filter edges: keep only edges that match allowed edge patterns
    edges_to_remove = []
    for u, v in filtered_graph.edges():
        u_label = filtered_graph.nodes[u].get("label")
        v_label = filtered_graph.nodes[v].get("label")
        edge_pattern = (u_label, v_label)
        if edge_pattern not in allowed_edge_patterns:
            edges_to_remove.append((u, v))

    # Remove edges that don't match the pattern
    filtered_graph.remove_edges_from(edges_to_remove)

    # Update graph reference
    graph = filtered_graph
    # analyze the graph
    # 1. number of nodes
    # 2. number of edges
    # 3. number of triangles
    # 4. number of connected components
    # 5. number of isolated nodes
    # 6. number of isolated edges
    # 7. number of isolated triangles

    print(f"Number of nodes in the graph: {graph.number_of_nodes()}")
    print(f"Number of edges in the graph: {graph.number_of_edges()}")
    # print(f"Number of triangles in the graph: {len(find_all_triangles_vf2(graph))}")
    print(
        f"Number of connected components in the graph: {nx.number_connected_components(graph)}"
    )
    print("--------------------------------")
    # print(f"Number of isolated nodes in the graph: {nx.number_isolates(graph)}")
    # print(f"Number of isolated edges in the graph: {nx.number_isolates(graph.edges())}")
    # print(
    #     f"Number of isolated triangles in the graph: {nx.number_isolates(graph.triangles())}"
    # )


if __name__ == "__main__":
    # import cProfile

    # cProfile.run(
    #     "main()",
    #     filename="HypothesisTesting/log_and_results_2008/result.out",
    #     sort="cumulative",
    # )
    main()
