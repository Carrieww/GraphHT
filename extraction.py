import time
from collections import defaultdict

import networkx as nx
import pandas as pd
from networkx.algorithms import isomorphism

from utils import HypothesisTesting


def new_graph_hypo_result(args, new_graph, result_list, time_used_list):
    time_rating_extraction_start = time.time()

    dataset_functions = {
        "movielens": {
            "edge": getEdges,
            "node": getNodes,
            "path": getPaths,
            "subgraph": getSubgraphs,
        },
        "citation": {
            "edge": getEdges,
            "node": getNodes,
            "path": getPaths,
            "subgraph": getSubgraphs,
        },
        "yelp": {
            "edge": getEdges,
            "node": getNodes,
            "path": getPaths,
            "subgraph": getSubgraphs,
        },
    }

    if args.dataset in dataset_functions:
        dataset_info = dataset_functions[args.dataset]

        if args.hypo == 1:
            total_result = dataset_info["edge"](args, new_graph)
        elif args.hypo == 2:
            total_result = dataset_info["node"](
                args, new_graph, dimension=args.dimension
            )
        elif args.hypo == 3:
            args.total_valid = 0
            args.total_minus_reverse = 0
            total_result = dataset_info["path"](args, new_graph)
        elif args.hypo == 4:
            # Subgraph hypothesis using VF2 isomorphism matching
            total_result = dataset_info["subgraph"](args, new_graph)
        else:
            raise Exception(
                f"Sorry, we don't support hypothesis {args.hypo} for {args.dataset}."
            )
    else:
        raise Exception(f"Sorry, we don't support the dataset {args.dataset}.")

    if str(list(args.hypothesis_pattern.keys())[0]) not in total_result:
        total_result[list(args.hypothesis_pattern.keys())[0]] = []

    result = {}

    for attribute, v in total_result.items():
        args.valid_edges.append(len(v))
        if len(v) <= 20:
            args.logger.info(v)
        args.logger.info(f"The sampled graph has {len(v)} relevant patterns.")
        if len(attribute.split("+")) == 1:
            if args.agg == "mean":
                if len(v) != 0:
                    hypo_result = HypothesisTesting(args, v, 1)
                    result[attribute] = hypo_result
                else:
                    result[attribute] = None

            else:
                args.logger.error(f"Sorry, we don't support {args.agg}.")
                raise Exception(f"Sorry, we don't support {args.agg}.")
        else:
            # pass
            result[attribute] = v
    time_used_list["sample_graph_by_condition"].append(
        round(time.time() - time_rating_extraction_start, 2)
    )

    if args.hypo == 3:
        result["total_valid"] = args.total_valid
        result["total_minus_reverse"] = args.total_minus_reverse
        now = time.time()
        args.logger.info("start finding diameter and density")
        print("start finding diameter and density")
        connected = nx.is_connected(new_graph)
        if connected:
            result["diameter"] = 10  # nx.diameter(new_graph)
        else:
            result["diameter"] = float("nan")
        args.logger.info(
            f"computing diameter {result['diameter']} takes {round(time.time() - now, 2)}"
        )
        print(
            f"computing diameter {result['diameter']} takes {round(time.time() - now, 2)}"
        )
        result["density"] = nx.density(new_graph)
    result_list.append(result)
    return result_list


def getNodes(args, new_graph, dimension):  # dimension = {"movie":"genre"}
    total_result = defaultdict(list)
    total_result_repeat = defaultdict(list)
    if len(list(args.hypothesis_pattern)) == 1:
        for condition_name, condition_dict in args.hypothesis_pattern.items():
            # The condition is on the dimension node only!
            if len(condition_dict) == 3 and list(dimension.keys())[0] in condition_dict:
                attribute_condition_dict = condition_dict[list(dimension.keys())[0]]
                node_attribute = nx.get_node_attributes(new_graph, "label")

                # for every node, check conditions
                for key, value in node_attribute.items():
                    flag = True
                    if new_graph.nodes[key]["label"] == list(dimension.keys())[0]:
                        for k, v in attribute_condition_dict.items():
                            if new_graph.nodes[key][k] == v:
                                pass
                            else:
                                flag = False
                    else:
                        flag = False
                    if flag:
                        total_result[condition_name].append(
                            new_graph.nodes[key][list(dimension.values())[0]]
                        )
            # The condition is on the edge or the other nodes
            # so we need to extract edges to filter
            else:
                selected_edge = condition_dict["target"]["edge"]
                edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)

                for key, value in edge_dict.items():
                    from_node, to_node = key
                    flag = checkCondition(args, condition_dict, new_graph, from_node)
                    if flag:
                        flag = checkCondition(args, condition_dict, new_graph, to_node)
                    else:
                        continue

                    if flag:
                        if (
                            new_graph.nodes[from_node]["label"]
                            == list(dimension.keys())[0]
                        ):  # dimension = {"movie":"genre"}
                            total_result_repeat[condition_name].append(from_node)
                        elif (
                            new_graph.nodes[to_node]["label"]
                            == list(dimension.keys())[0]
                        ):
                            total_result_repeat[condition_name].append(to_node)
                        else:
                            raise Exception("Wrong code")
                distinct_nodes = list(set(total_result_repeat[condition_name]))
                for i in distinct_nodes:
                    assert new_graph.nodes[i]["label"] == list(dimension.keys())[0]
                    total_result[condition_name].append(
                        new_graph.nodes[i][list(dimension.values())[0]]
                    )

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result


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
    # get selected edge type from args.hypothesis_pattern
    if len(list(args.hypothesis_pattern)) == 1:
        for condition_name, condition_dict in args.hypothesis_pattern.items():
            selected_edge = condition_dict["target"]["edge"]
    else:
        raise Exception("Sorry we only support one condition_name")
    edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)

    total_result = defaultdict(list)

    for key, value in edge_dict.items():
        from_node, to_node = key
        # check the label of every node in edge_dict
        # if we have the attribute constraints in args.hypothesis_pattern, we check the condition
        for (
            condition_name,
            condition_dict,
        ) in args.hypothesis_pattern.items():  # {"1-1":{condition_dict}}
            # check both from and to node conditions
            flag = checkCondition(args, condition_dict, new_graph, from_node)
            if flag:
                flag = checkCondition(args, condition_dict, new_graph, to_node)
            else:
                continue

            if flag:
                total_result[condition_name].append(value)

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


def getPaths(args, new_graph):
    total_result = defaultdict(list)
    # path_count_map is a hash map: keys = path node index join by "." and values = 1
    path_count_map = {}
    if len(list(args.hypothesis_pattern)) == 1:
        for condition_name, condition_dict in args.hypothesis_pattern.items():
            assert (
                "path" in condition_dict
            ), f"hypo {args.hypo} must require path information in args.hypothesis_pattern."
            path_info = condition_dict["path"]

            for ini_node in new_graph.nodes():
                if new_graph.nodes[ini_node]["label"] == path_info[0]["type"]:
                    flag = True
                    for k, v in path_info[0]["attribute"].items():
                        # for citation 3-1-1 vague match
                        if pd.isna(new_graph.nodes[ini_node][k]):
                            flag = False
                            break
                        elif new_graph.nodes[ini_node][k] != v and (
                            v not in new_graph.nodes[ini_node][k]
                            if isinstance(v, str)
                            else True
                        ):
                            flag = False
                            break
                    if flag:
                        find_paths(
                            args,
                            new_graph,
                            path_info,
                            [ini_node],
                            ini_node,
                            1,
                            path_count_map,
                        )
            args.logger.info("finished path extraction!")
            print("finished path extraction!")

            user_set = set()
            movie_set = set()
            if "target" in condition_dict and "edge" in condition_dict["target"]:
                extract_edge_attr = condition_dict["target"]["edge"]
                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    user_set.update([r[1]])
                    movie_set.update([r[0], r[2]])
                    res = []
                    for index in range(len(r) - 1):
                        e = (r[index], r[index + 1])
                        if extract_edge_attr in new_graph.edges[e].keys():
                            res.append(new_graph.edges[e][extract_edge_attr])

                    # average
                    total_result[condition_name].append(sum(res) / len(res))

                    # difference
                    # total_result[condition_name].append(abs(res[0] - res[1]))

                    # only one valid edge on the path
                    # total_result[condition_name].append(res[0])
            elif (
                "target" in condition_dict and "node" in condition_dict["target"]
            ):  # node in dict, "node": {"index":2,"attribute":"age"}
                node_config = condition_dict["target"]["node"]
                extract_node_index = node_config["index"]
                extract_node_attr = node_config["attribute"]

                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    r = [int(i) for i in r]
                    user_set.update([r[1]])
                    movie_set.update([r[0], r[2]])
                    res = []
                    for node in r:
                        if new_graph.nodes[node]["label"] == extract_node_index:
                            res.append(new_graph.nodes[node][extract_node_attr])

                    total_result[condition_name].append(sum(res) / len(res))

            else:
                raise Exception("You must provide the dimension in the attribute.")
            total_result[condition_name + "+user_coverage"].append(len(user_set))
            total_result[condition_name + "+movie_coverage"].append(len(movie_set))

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result


def getSubgraphs(args, new_graph):
    """
    Extract subgraphs matching the pattern using VF2 isomorphism algorithm.
    Supports attributed node and edge matching.

    Args:
        args: Arguments containing hypothesis_pattern
        new_graph: The graph to search in

    Returns:
        dict: {condition_name: [list of extracted attribute values]}
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
        pattern_graph = nx.DiGraph() if new_graph.is_directed() else nx.Graph()

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
        if new_graph.is_directed():
            matcher = isomorphism.DiGraphMatcher(
                new_graph, pattern_graph, node_match=node_match, edge_match=edge_match
            )
        else:
            matcher = isomorphism.GraphMatcher(
                new_graph, pattern_graph, node_match=node_match, edge_match=edge_match
            )

        # Find all subgraph isomorphisms
        matches = list(matcher.subgraph_isomorphisms_iter())

        # Remove duplicate matches for undirected graphs
        # In undirected graphs, the same subgraph can be matched multiple times
        # with different node mappings (e.g., {n1: A, n2: B, n3: C} and {n1: A, n2: C, n3: B})
        if not new_graph.is_directed():
            unique_matches = []
            seen_subgraphs = set()
            for match in matches:
                # match is a dict: {graph_node_id: pattern_node_id}
                # Extract the set of graph nodes (subgraph)
                graph_nodes = frozenset(match.keys())
                if graph_nodes not in seen_subgraphs:
                    seen_subgraphs.add(graph_nodes)
                    unique_matches.append(match)
            matches = unique_matches

        # Extract target attribute values from matched subgraphs
        target_config = condition_dict.get("target", {})

        if "edge" in target_config:
            # Extract edge attribute
            edge_attr = target_config["edge"]
            for match in matches:
                # match is a dict: {graph_node_id: pattern_node_id}
                # Need to reverse lookup to get graph nodes from pattern nodes
                reverse_match = {v: k for k, v in match.items()}
                # Reconstruct edges from pattern using matched nodes
                edge_values = []
                for edge_info in pattern_edges:
                    from_id = edge_info.get("from")
                    to_id = edge_info.get("to")
                    graph_from = reverse_match[from_id]
                    graph_to = reverse_match[to_id]

                    # Check if edge exists in graph (should exist for subgraph isomorphism)
                    if new_graph.has_edge(graph_from, graph_to):
                        if edge_attr in new_graph.edges[graph_from, graph_to]:
                            edge_values.append(
                                new_graph.edges[graph_from, graph_to][edge_attr]
                            )

                if edge_values:
                    # Aggregate edge values (e.g., mean, sum, etc.)
                    # For now, use mean as default
                    total_result[condition_name].append(
                        sum(edge_values) / len(edge_values)
                    )

        elif "node" in target_config:
            # Extract node attribute
            node_config = target_config["node"]
            node_index = node_config.get("index")  # Index in pattern nodes list
            node_attr = node_config.get("attribute")

            if node_index is None or node_attr is None:
                raise ValueError("Node target must specify 'index' and 'attribute'.")

            # Get the pattern node id at the specified index
            if node_index >= len(pattern_nodes):
                raise ValueError(
                    f"Node index {node_index} out of range for pattern with {len(pattern_nodes)} nodes."
                )

            pattern_node_id = pattern_nodes[node_index].get("id")
            if pattern_node_id is None:
                raise ValueError(f"Node at index {node_index} must have an 'id' field.")

            for match in matches:
                # match is a dict: {graph_node_id: pattern_node_id}
                # Need to reverse lookup to get graph nodes from pattern nodes
                reverse_match = {v: k for k, v in match.items()}
                if pattern_node_id in reverse_match:
                    graph_node_id = reverse_match[pattern_node_id]
                    if node_attr in new_graph.nodes[graph_node_id]:
                        total_result[condition_name].append(
                            new_graph.nodes[graph_node_id][node_attr]
                        )

        elif "node_diff" in target_config:
            # Extract difference between two node attributes
            node_diff_config = target_config["node_diff"]
            node_indices = node_diff_config.get("nodes")  # List of two node indices
            node_attr = node_diff_config.get("attribute")

            if node_indices is None or len(node_indices) != 2:
                raise ValueError(
                    "Node_diff target must specify 'nodes' as a list of exactly 2 indices."
                )
            if node_attr is None:
                raise ValueError("Node_diff target must specify 'attribute'.")

            # Get pattern node ids at the specified indices
            pattern_node_ids = []
            for node_index in node_indices:
                if node_index >= len(pattern_nodes):
                    raise ValueError(
                        f"Node index {node_index} out of range for pattern with {len(pattern_nodes)} nodes."
                    )
                pattern_node_id = pattern_nodes[node_index].get("id")
                if pattern_node_id is None:
                    raise ValueError(
                        f"Node at index {node_index} must have an 'id' field."
                    )
                pattern_node_ids.append(pattern_node_id)

            for match in matches:
                # match is a dict: {graph_node_id: pattern_node_id}
                # Need to reverse lookup to get graph nodes from pattern nodes
                reverse_match = {v: k for k, v in match.items()}
                # Get graph node IDs for both pattern nodes
                graph_node1 = reverse_match.get(pattern_node_ids[0])
                graph_node2 = reverse_match.get(pattern_node_ids[1])

                if graph_node1 is not None and graph_node2 is not None:
                    # Extract attribute values from both nodes
                    if (
                        node_attr in new_graph.nodes[graph_node1]
                        and node_attr in new_graph.nodes[graph_node2]
                    ):
                        val1 = new_graph.nodes[graph_node1][node_attr]
                        val2 = new_graph.nodes[graph_node2][node_attr]
                        # Calculate absolute difference
                        diff = abs(val1 - val2)
                        total_result[condition_name].append(diff)

        else:
            raise ValueError(
                "Target must specify 'edge', 'node', or 'node_diff' to extract."
            )

    return total_result
