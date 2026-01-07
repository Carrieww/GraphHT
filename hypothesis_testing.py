def HypothesisTesting(args, attribute_values, verbose=1):
    # TODO: implement this function

    test_statistics, accept, confidence_interval, p_value = 0, 0, 0, 0
    return test_statistics, accept, confidence_interval, p_value


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
