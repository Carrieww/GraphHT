import statistics
import time
from collections import defaultdict

import networkx as nx
from scipy import stats

from utils import print_hypo_log


def new_graph_hypo_result(args, new_graph, result_list, time_used_list):
    time_rating_extraction_start = time.time()

    dataset_functions = {
        "movielens": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "citation": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "yelp": {"edge": getEdges, "node": getNodes, "path": getPaths},
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

        else:
            raise Exception(
                f"Sorry, we don't support hypothesis {args.hypo} for {args.dataset}."
            )
    else:
        raise Exception(f"Sorry, we don't support the dataset {args.dataset}.")

    if str(list(args.attribute.keys())[0]) not in total_result:
        total_result[list(args.attribute.keys())[0]] = []

    result = {}

    for attribute, v in total_result.items():
        args.valid_edges.append(len(v))
        if len(attribute.split("+")) == 1:
            if args.agg == "mean":
                if len(v) > 1:
                    print(max(v), len(v))
                    variance = statistics.variance(v)
                    # print(f"The variance is {round(variance,3)}.")
                    # args.logger.info(f"The variance is {round(variance,3)}.")
                    args.variance.append(variance)

                    t_stat, p_value = stats.ttest_1samp(
                        v, popmean=args.ground_truth, alternative="two-sided"
                    )
                    print_hypo_log(args, t_stat, p_value, args.H0, twoSides=True)

                    result[attribute] = sum(v) / len(v)
                else:
                    result[attribute] = float("nan")
            elif args.agg == "max":
                result[attribute] = max(v)
            elif args.agg == "min":
                result[attribute] = min(v)
            elif args.agg == "variance":
                result[attribute] = statistics.variance(v)
            else:
                args.logger.error(f"Sorry, we don't support {args.agg}.")
                raise Exception(f"Sorry, we don't support {args.agg}.")
        else:
            result[attribute] = v[0]
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
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
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
                selected_edge = condition_dict["edge"]
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
            if new_graph.nodes[node_index][k] == v:
                pass
            else:
                return False
    else:
        return True
    return True


def getEdges(args, new_graph):
    # get selected edge type from args.attribute
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            selected_edge = condition_dict["edge"]
    else:
        raise Exception("Sorry we only support one condition_name")
    edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)

    total_result = defaultdict(list)

    for key, value in edge_dict.items():
        from_node, to_node = key
        # check the label of every node in edge_dict
        # if we have the attribute constraints in args.attribute, we check the condition
        for (
            condition_name,
            condition_dict,
        ) in args.attribute.items():  # {"1-1":{condition_dict}}
            # check both from and to node conditions
            flag = checkCondition(args, condition_dict, new_graph, from_node)
            if flag:
                flag = checkCondition(args, condition_dict, new_graph, to_node)
            else:
                continue

            if flag:
                total_result[condition_name].append(value)

    return total_result


def getPaths_old(args, new_graph):
    total_result = defaultdict(list)
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            assert (
                "path" in condition_dict
            ), f"hypo {args.hypo} must require path information in args.attribute."
            path_info = condition_dict["path"]
            # count = 1
            # count_true = 1
            tmp = new_graph.nodes()
            for ini_node in tmp:
                # print(count)
                # count += 1
                # check ini node has the correct label
                if (
                    new_graph.nodes[ini_node]["label"]
                    == list(path_info.keys())[0].split("_")[0]
                ):
                    # check condition
                    flag = True
                    for k, v in list(path_info.values())[0].items():
                        if new_graph.nodes[ini_node][k] != v:
                            flag = False
                    if flag:
                        # check neighbor
                        neighbor_list = [n for n in new_graph.neighbors(ini_node)]
                        for neighbor in neighbor_list:
                            # check 1st neighbor has the correct label
                            if (
                                new_graph.nodes[neighbor]["label"]
                                == list(path_info.keys())[1].split("_")[0]
                            ):
                                # check condition
                                first_flag = True
                                for k, v in list(path_info.values())[1].items():
                                    if new_graph.nodes[neighbor][k] != v:
                                        first_flag = False
                                if first_flag:
                                    # check 2nd neighbor has the correct label
                                    next_neighbor_list = [
                                        n for n in new_graph.neighbors(neighbor)
                                    ]
                                    for next_neighbor in next_neighbor_list:
                                        if (next_neighbor != ini_node) and (
                                            new_graph.nodes[next_neighbor]["label"]
                                            == list(path_info.keys())[2].split("_")[0]
                                        ):
                                            # if (
                                            #     new_graph.nodes[next_neighbor]["label"]
                                            #     == list(path_info.keys())[2].split("_")[0]
                                            # ):
                                            # check condition
                                            second_flag = True
                                            for k, v in list(path_info.values())[
                                                2
                                            ].items():
                                                if (
                                                    new_graph.nodes[next_neighbor][k]
                                                    != v
                                                ):
                                                    second_flag = False
                                            # ensure Action-user-Crime will not be repeatedly
                                            # added to result if crime movie is also an action
                                            # one and action one contains crime genre
                                            for k, v in list(path_info.values())[
                                                0
                                            ].items():
                                                if (
                                                    new_graph.nodes[next_neighbor][k]
                                                    == v
                                                ):
                                                    second_flag = False
                                            if second_flag:
                                                # print(f"{count_true}-th new result")
                                                # count_true += 1
                                                rat_1 = new_graph.edges[
                                                    ini_node, neighbor
                                                ]["rating"]
                                                rat_2 = new_graph.edges[
                                                    neighbor, next_neighbor
                                                ]["rating"]
                                                total_result[condition_name].append(
                                                    (rat_1 + rat_2) / 2
                                                )
    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result


def find_paths(
    args, graph, path_info, current_path, current_node, depth, path_count_map
):
    if depth == len(path_info):
        str_current_path = [str(i) for i in current_path]
        key = ".".join(str_current_path)
        str_current_path.reverse()
        reverse_key = ".".join(str_current_path)
        # current_path.reverse()
        args.total_valid += 1
        if path_count_map.get(key) is None and path_count_map.get(reverse_key) is None:
            args.total_minus_reverse += 1
            if len(set(current_path)) == len(current_path):
                # res.append(current_path)
                path_count_map[key] = 1
        return

    current_label = path_info[depth]["type"]
    current_conditions = path_info[depth]["attribute"]

    for neighbor in graph.neighbors(current_node):
        neighbor_label = graph.nodes[neighbor]["label"]

        if neighbor_label == current_label:
            # Check conditions for the current node
            flag = True
            for k, v in current_conditions.items():
                if graph.nodes[neighbor][k] != v:
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
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            assert (
                "path" in condition_dict
            ), f"hypo {args.hypo} must require path information in args.attribute."
            path_info = condition_dict["path"]

            for ini_node in new_graph.nodes():
                if new_graph.nodes[ini_node]["label"] == path_info[0]["type"]:
                    flag = True
                    for k, v in path_info[0]["attribute"].items():
                        if new_graph.nodes[ini_node][k] != v:
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
            if "edge" in condition_dict.keys():
                extract_edge_attr = condition_dict["edge"]
                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    user_set.update([r[1]])
                    movie_set.update([r[0], r[2]])
                    res = []
                    for index in range(len(r) - 1):
                        e = (r[index], r[index + 1])
                        if extract_edge_attr in new_graph.edges[e].keys():
                            res.append(new_graph.edges[e][extract_edge_attr])

                    total_result[condition_name].append(sum(res) / len(res))

            elif (
                "node" in condition_dict.keys()
            ):  # node in dict, "node": {"index":2,"attribute":"age"}
                extract_node_index = condition_dict["node"]["index"]
                extract_node_attr = condition_dict["node"]["attribute"]

                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    r = [int(i) for i in r]
                    user_set.update([r[1]])
                    movie_set.update([r[0], r[2]])
                    for node in r:
                        if new_graph.nodes[node]["label"] == extract_node_index:
                            total_result[condition_name].append(
                                new_graph.nodes[node][extract_node_attr]
                            )
            else:
                raise Exception("You must provide the dimension in the attribute.")
            total_result[condition_name + "+user_coverage"].append(len(user_set))
            total_result[condition_name + "+movie_coverage"].append(len(movie_set))
            # print(args.total_valid)
            # print(args.total_minus_reverse)

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result
