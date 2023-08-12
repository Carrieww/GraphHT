import statistics
import time

import networkx as nx


def new_graph_hypo_result(args, new_graph, result_list, num_sample):
    if args.dataset == "movielens":
        if args.hypo == 1 or args.hypo == 2:
            result_dict = nx.get_edge_attributes(new_graph, name="rating")
            result = getRatings(args, new_graph, result_dict)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {result}.")
    elif args.dataset == "citation":
        if args.hypo == 1:
            result = getAuthors(args, new_graph)
            args.logger.info(
                f"sample {num_sample}: {args.agg} number of author is {result}."
            )
        elif args.hypo == 2:
            result = getCitations(args, new_graph)
        elif args.hypo == 3:
            # number of triangles
            result = sum(nx.triangles(new_graph).values()) / 3
    result_list.append(result)
    return result_list


def getCitations(args, new_graph):
    total_citations = []
    node_attribute = nx.get_node_attributes(new_graph, "year")
    for key, value in node_attribute.items():
        if value == args.attribute:
            total_citations.append(new_graph.nodes[key]["citation"])

    if len(total_citations) == 0:
        total_citations.append(0)

    if args.agg == "mean":
        result = sum(total_citations) / len(total_citations)
    elif args.agg == "max":
        result = max(total_citations)
    elif args.agg == "min":
        result = min(total_citations)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getRatings(args, new_graph, result_dict):
    total_rating = []
    for key, value in result_dict.items():
        from_node, to_node = key
        # print(from_node, to_node)
        if new_graph.nodes[from_node]["label"] == "movie":
            # print("from")
            if new_graph.nodes[from_node][args.attribute[0]] == 1:
                total_rating.append(value)
        elif new_graph.nodes[to_node]["label"] == "movie":
            # print("to")
            if new_graph.nodes[to_node][args.attribute[0]] == 1:
                total_rating.append(value)

    if len(total_rating) == 0:
        total_rating.append(0)

    if args.agg == "mean":
        result = sum(total_rating) / len(total_rating)
    elif args.agg == "max":
        result = max(total_rating)
    elif args.agg == "min":
        result = min(total_rating)
    elif args.agg == "variance":
        result = statistics.variance(total_rating)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getAuthors(args, new_graph):
    paper_set = set()
    author_list = []
    node_attribute = nx.get_node_attributes(new_graph, "year")
    for key, value in node_attribute.items():
        if value == args.attribute:
            degree = new_graph.degree(key, "writes")
            if degree != 0:
                paper_set.add(key)
                author_list.append(new_graph.degree(key, "writes"))

    if args.agg == "mean":
        if len(paper_set) == 0:
            print(paper_set)
            raise Exception(
                f"no author satisfying {args.attribute} by sampling {args.ratio} of {args.dataset}."
            )
        return sum(author_list) / len(paper_set)

    elif args.agg == "max":
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    elif args.agg == "min":
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
