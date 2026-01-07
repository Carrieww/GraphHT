import time
from collections import defaultdict

import networkx as nx
import pandas as pd
from networkx.algorithms import isomorphism


def extract_attributes(args, subgraph_list):
    """
    Extract target attribute values from a list of subgraphs (for TriangleS).
    This is used when subgraphs are already extracted and we only need to get attributes.

    Args:
        args: Arguments containing hypothesis_pattern and target configuration
        subgraph_list: List of subgraphs, each subgraph is a Set[int] of node IDs

    Returns:
        dict: {condition_name: [list of extracted attribute values]}
    """
    attribute_dict = defaultdict(list)

    if len(list(args.hypothesis_pattern)) != 1:
        raise Exception("Sorry we only support one condition_name")

    for condition_name, condition_dict in args.hypothesis_pattern.items():
        # Get the original graph from args (needed to access node/edge attributes)
        if not hasattr(args, "original_graph"):
            raise ValueError(
                "args.original_graph must be set for subgraph list extraction"
            )

        graph = args.original_graph
        target_config = condition_dict.get("target", {})
        pattern_nodes = condition_dict["subgraph"]["nodes"]

        # Helper function to match graph nodes to pattern nodes based on attributes
        def match_nodes_to_pattern(subgraph_nodes_set, pattern_nodes_list):
            """
            Match graph nodes to pattern nodes based on label and attributes.
            Returns a dict: {pattern_node_id: graph_node_id}
            """
            match = {}
            remaining_graph_nodes = set(subgraph_nodes_set)

            for pattern_node_info in pattern_nodes_list:
                pattern_node_id = pattern_node_info.get("id")
                pattern_label = pattern_node_info.get("label")
                pattern_attrs = pattern_node_info.get("attribute", {})

                # Find matching graph node
                for graph_node_id in remaining_graph_nodes:
                    graph_node = graph.nodes[graph_node_id]

                    # Check label
                    if graph_node.get("label") != pattern_label:
                        continue

                    # Check attributes
                    matches = True
                    for attr_key, attr_val in pattern_attrs.items():
                        graph_val = graph_node.get(attr_key)
                        if pd.isna(graph_val):
                            matches = False
                            break
                        if graph_val != attr_val and (
                            attr_val not in graph_val
                            if isinstance(graph_val, str)
                            else True
                        ):
                            matches = False
                            break

                    if matches:
                        match[pattern_node_id] = graph_node_id
                        remaining_graph_nodes.remove(graph_node_id)
                        break

            return match if len(match) == len(pattern_nodes_list) else None

        if "node_diff" in target_config:
            # Extract node difference attribute (e.g., n_citation difference)
            node_diff_config = target_config["node_diff"]
            pattern_node_ids = node_diff_config[
                "nodes"
            ]  # Pattern node IDs (e.g., ["n2", "n3"])
            attr_name = node_diff_config["attribute"]

            for subgraph_nodes in subgraph_list:
                if len(subgraph_nodes) != len(pattern_nodes):
                    continue

                # Match graph nodes to pattern nodes
                node_match = match_nodes_to_pattern(subgraph_nodes, pattern_nodes)
                if node_match is None:
                    continue

                # Get attribute values for the specified pattern nodes
                attr_values = []
                for pattern_node_id in pattern_node_ids:
                    if pattern_node_id in node_match:
                        graph_node_id = node_match[pattern_node_id]
                        attr_val = graph.nodes[graph_node_id].get(attr_name)
                        if attr_val is not None:
                            attr_values.append(attr_val)

                # Calculate difference (assuming 2 nodes for difference)
                if len(attr_values) == 2:
                    diff = attr_values[0] - attr_values[1]
                    attribute_dict[condition_name].append(diff)
        else:
            # For other target types, can be extended similarly
            raise NotImplementedError(
                f"Target type {list(target_config.keys())} not yet supported for subgraph list extraction"
            )

    return attribute_dict


def extract_instances(args, new_graph):
    subgraph_type = args.hypothesis_pattern[args.condition_name]["type"]

    if "edge" in subgraph_type:
        instance_dict = getEdges(args, new_graph)
    elif "node" in subgraph_type:
        instance_dict = getNodes(args, new_graph)
    elif "path" in subgraph_type:
        args.total_valid = 0
        args.total_minus_reverse = 0
        instance_dict = getPaths(args, new_graph)
    elif "subgraph" in subgraph_type:
        instance_dict = getSubgraphs(args, new_graph)

    return instance_dict


def hypothesis_testing(args, total_result, new_graph):
    """
    Extract attribute values from instances and perform hypothesis testing.

    Args:
        args: Arguments containing hypothesis pattern and testing configuration
        total_result: dict from extract_instances() with condition_name: [list of instances]
        new_graph: The graph to extract attributes from

    Returns:
        dict: Results of hypothesis testing
    """

    # Get condition_name and condition_dict from hypothesis_pattern
    condition_name = list(args.hypothesis_pattern.keys())[0]
    condition_dict = args.hypothesis_pattern[condition_name]
    target_config = condition_dict["target"]

    for attribute, v in total_result.items():
        # Extract attribute values from instances based on hypothesis type
        attribute_values = []

        if "edge" in target_config:
            edge_attr = target_config["edge"]
            for edge_tuple in v:
                from_node, to_node = edge_tuple
                if new_graph.has_edge(from_node, to_node):
                    if edge_attr in new_graph.edges[from_node, to_node]:
                        attribute_values.append(
                            new_graph.edges[from_node, to_node][edge_attr]
                        )

        elif "node" in target_config:
            node_attr = target_config["node"]["attribute"]

            for node_id in v:
                if node_attr in new_graph.nodes[node_id]:
                    attribute_values.append(new_graph.nodes[node_id][node_attr])

        elif "node_diff" in target_config:
            pass
        else:
            raise ValueError(
                "Only 'edge', 'node', or 'node_diff' are supported in target_config"
            )

    return attribute_values


def getNodes(args, graph):
    """
    given a graph, get the nodes according to the hypothesis

    Returns:
        dict: {condition_name: [list of node IDs]}
    """

    # TODO: implement this function
    return {}


def checkCondition(args, condition_dict, new_graph, node_index):
    if new_graph.nodes[node_index]["label"] in condition_dict:
        attribute_condition_dict = condition_dict[new_graph.nodes[node_index]["label"]]
        for k, v in attribute_condition_dict.items():  # {"gender":"M"}}
            if k in new_graph.nodes[node_index] and new_graph.nodes[node_index][k] == v:
                pass
            else:
                return False
    else:
        return True
    return True


def getEdges(args, new_graph):
    """
    given a graph, get the edges according to the hypothesis

    Returns:
        dict: {condition_name: list of edge tuples (from_node, to_node)}
    """
    # get selected edge type from args.hypothesis_pattern
    if len(list(args.hypothesis_pattern)) == 1:
        for condition_name, condition_dict in args.hypothesis_pattern.items():
            selected_edge = condition_dict["target"]["edge"]
    else:
        raise Exception("Sorry we only support one condition_name")
    edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)

    total_result = {condition_name: list(edge_dict.keys())}

    return total_result


def find_paths(
    args, graph, path_info, current_path, current_node, depth, path_count_map
):
    if depth == len(path_info):
        str_current_path = [str(i) for i in current_path]
        key = ".".join(str_current_path)
        str_current_path.reverse()
        reverse_key = ".".join(str_current_path)
        args.total_valid += 1
        if path_count_map.get(key) is None and path_count_map.get(reverse_key) is None:
            args.total_minus_reverse += 1
            if len(set(current_path)) == len(current_path):
                path_count_map[key] = 1
        return

    current_label = path_info[depth]["type"]
    current_conditions = path_info[depth]["attribute"]

    for neighbor in graph.neighbors(current_node):
        neighbor_label = graph.nodes[neighbor]["label"]

        if neighbor_label == current_label:
            # Check conditions for the current node
            flag = True
            # for citation 3-1-1 vague match
            for k, v in current_conditions.items():
                if pd.isna(pd.isna(graph.nodes[neighbor][k])):
                    flag = False
                    break
                elif graph.nodes[neighbor][k] != v and (
                    v not in graph.nodes[neighbor][k] if isinstance(v, str) else True
                ):
                    flag = False
                    break

            if flag:
                # Recursively explore the next depth
                find_paths(
                    args,
                    graph,
                    path_info,
                    current_path + [neighbor],
                    neighbor,
                    depth + 1,
                    path_count_map,
                )


def getPaths(args, graph):
    """
    given a graph, get the paths according to the hypothesis

    Returns:
        dict: {condition_name: [list of path instances (node ID lists)]}
    """
    # TODO: implement this function
    return {}


def getSubgraphs(args, graph):
    """
    given a graph, get the subgraphs according to the hypothesis

    Args:
        args: Arguments containing hypothesis_pattern
        graph: The graph to search in

    Returns:
        dict: {condition_name: [list of subgraph instances (matching node mappings)]}
             Each instance is a dict: {graph_node_id: pattern_node_id}
    """
    total_result = defaultdict(list)

    if len(list(args.hypothesis_pattern)) != 1:
        raise Exception("Sorry we only support one condition_name")

    for condition_name, condition_dict in args.hypothesis_pattern.items():
        if "subgraph" not in condition_dict:
            raise ValueError(
                f"Subgraph hypothesis must contain 'subgraph' key in hypothesis_pattern."
            )

        subgraph_pattern = condition_dict["subgraph"]
        pattern_nodes = subgraph_pattern["nodes"]
        pattern_edges = subgraph_pattern["edges"]

        # Build pattern graph from hypothesis_pattern
        pattern_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        # Add nodes to pattern graph with their attributes
        for node_info in pattern_nodes:
            node_id = node_info.get("id")
            if node_id is None:
                raise ValueError(
                    "Each node in subgraph pattern must have an 'id' field."
                )
            pattern_graph.add_node(node_id)

            # Store pattern node attributes for matching
            pattern_graph.nodes[node_id]["label"] = node_info.get("label")
            pattern_graph.nodes[node_id]["attributes"] = node_info.get("attribute", {})

        # Add edges to pattern graph
        for edge_info in pattern_edges:
            from_id = edge_info.get("from")
            to_id = edge_info.get("to")
            if from_id not in pattern_graph or to_id not in pattern_graph:
                raise ValueError(
                    f"Edge references undefined node: {from_id} -> {to_id}"
                )
            pattern_graph.add_edge(from_id, to_id)

        # Define node matching function
        def node_match(n1, n2):
            """
            Check if graph node n1 matches pattern node n2.

            Args:
                n1: Graph node data (contains 'label', etc.)
                n2: Pattern node data (contains 'label', 'attributes')
            """
            # n1 is graph node, n2 is pattern node
            label = n2.get("label")
            pattern_attrs = n2.get("attributes", {})

            # Check node type (label)
            if n1.get("label") != label:
                return False

            # Check node attributes
            for attr_key, attr_val in pattern_attrs.items():
                graph_val = n1.get(attr_key)
                if pd.isna(graph_val):
                    return False
                # Support vague match for string attributes (like citation dataset)
                if graph_val != attr_val and (
                    attr_val not in graph_val if isinstance(graph_val, str) else True
                ):
                    return False
            return True

        # Define edge matching function (for edge attributes if needed)
        def edge_match(e1, e2):
            """Check if graph edge e1 matches pattern edge e2."""
            # For now, we don't match edge attributes in the pattern
            # If needed, can be extended similar to node_match
            return True

        # Use VF2 matcher
        if graph.is_directed():
            matcher = isomorphism.DiGraphMatcher(
                graph, pattern_graph, node_match=node_match, edge_match=edge_match
            )
        else:
            matcher = isomorphism.GraphMatcher(
                graph, pattern_graph, node_match=node_match, edge_match=edge_match
            )

        # Find all subgraph isomorphisms
        matches = list(matcher.subgraph_isomorphisms_iter())

        # Remove duplicate matches for undirected graphs
        # In undirected graphs, the same subgraph can be matched multiple times
        # with different node mappings (e.g., {n1: A, n2: B, n3: C} and {n1: A, n2: C, n3: B})
        if not graph.is_directed():
            unique_matches = []
            seen_subgraphs = set()
            for match in matches:
                # match is a dict: {graph_node_id: pattern_node_id}
                # Extract the set of graph nodes (subgraph)
                graph_nodes = frozenset(match.keys())
                if graph_nodes not in seen_subgraphs:
                    seen_subgraphs.add(graph_nodes)
                    unique_matches.append(match)

        total_result[condition_name].append(unique_matches)

    return total_result


def HypothesisTesting(args, attribute_values, verbose=1):
    # TODO: implement this function

    test_statistics, accept, confidence_interval, p_value = 0, 0, 0, 0
    return test_statistics, accept, confidence_interval, p_value
