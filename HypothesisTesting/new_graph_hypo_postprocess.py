import statistics
import time
from collections import defaultdict

import networkx as nx


def new_graph_hypo_result(args, new_graph, result_list, num_sample):
    if args.dataset == "movielens":
        ##### one side #####
        # degree
        if args.hypo == 1:
            result = getMovies(args, new_graph)
            args.logger.info(
                f"sample {num_sample}: {args.agg} number of {args.hypo} movies rated by users is {result}."
            )
        # node attribute
        elif args.hypo == 2:
            result = getGenres(args, new_graph)
        # edge attribute mean and variance
        elif args.hypo == 3 or args.hypo == 4:
            result_dict = nx.get_edge_attributes(new_graph, name="rating")
            result = getRatings(args, new_graph, result_dict)[args.attribute[0]]
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {result}.")

        ##### two side #####
        elif args.hypo == 10:
            result_dict = nx.get_edge_attributes(new_graph, name="rating")
            # result is a dictionary
            result = getRatings(args, new_graph, result_dict)
            args.logger.info(
                f"sample {num_sample}: {args.agg} rating of {args.attribute[0]} is {result[args.attribute[0]]} and {args.agg} rating of {args.attribute[1]} is {result[args.attribute[1]]}."
            )
        else:
            raise Exception(f"Sorry, we don't support {args.hypo} for {args.dataset}.")
    elif args.dataset == "citation":
        ##### one side #####
        # degree
        if args.hypo == 1:
            result = getAuthors(args, new_graph)
            args.logger.info(
                f"sample {num_sample}: {args.agg} number of author is {result}."
            )

        # node attribute
        elif args.hypo == 2:
            result = getCitations(args, new_graph)

        # edge attribute
        elif args.hypo == 3:
            result_dict = nx.get_edge_attributes(new_graph, name="correlation")
            result = getCorrelation(args, new_graph, result_dict)

        # number of triangles
        elif args.hypo == 4:
            result = sum(nx.triangles(new_graph).values()) / 3

        ##### two side #####
        elif args.hypo == 10:
            result = getCitations(args, new_graph)
        else:
            raise Exception(f"Sorry, we don't support {args.hypo} for {args.dataset}.")
    result_list.append(result)
    return result_list


def getCorrelation(args, new_graph, result_dict):
    total_correlation = defaultdict(list)
    # paper_set = set()
    for key, value in result_dict.items():
        from_node, to_node = key
        # print(from_node, to_node)
        if new_graph.nodes[from_node]["year"] == int(args.attribute[0]):
            total_correlation[from_node].append(value)

        elif new_graph.nodes[to_node]["year"] == int(args.attribute[0]):
            total_correlation[to_node].append(value)

    if len(total_correlation) == 0:
        raise Exception(
            f"No nodes with valid correlation satisfying the current hypothesis and sampling ratio."
        )
    # total_correlation = list(map(sum, list(total_correlation.values())))
    value_list = list(total_correlation.values())
    total_correlation = list(map(lambda idx: sum(idx) / float(len(idx)), value_list))

    if args.agg == "mean":
        result = sum(total_correlation) / len(total_correlation)
    elif args.agg == "max":
        result = max(total_correlation)
    elif args.agg == "min":
        result = min(total_correlation)
    elif args.agg == "variance":
        result = statistics.variance(total_correlation)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getGenres(args, new_graph):
    total_genre = []
    genre_list = []
    node_attribute = nx.get_node_attributes(new_graph, "label")
    for key, value in node_attribute.items():
        num_genre = 0
        if value == "movie":
            if genre_list == []:
                genre_list = list(new_graph.nodes[key].keys())[2:]
            else:
                for genre in genre_list:
                    num_genre += new_graph.nodes[key][genre]
            total_genre.append(num_genre)

    # sum(total_genre)/len(total_genre)
    if len(total_genre) == 0:
        total_genre.append(0)

    if args.agg == "mean":
        result = sum(total_genre) / len(total_genre)
    elif args.agg == "max":
        result = max(total_genre)
    elif args.agg == "min":
        result = min(total_genre)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getMovies(args, new_graph):
    if args.attribute == None:
        degree_list = []
        user_list = []
        for i in new_graph.nodes:
            if new_graph.nodes[i]["label"] == "user":
                # print(new_graph.degree(i))
                degree_list.append(new_graph.degree(i))
                user_list.append(i)
    else:
        result_dict = nx.get_edge_attributes(new_graph, name="rating")
        degree_list = []
        user_degree_dict = defaultdict(list)
        user_list = set()
        for i, _ in result_dict.items():
            from_node, to_node = i
            # if new_graph.nodes[from_node]["label"]=="user" or new_graph.nodes[to_node]["label"]=="user" :
            if new_graph.nodes[from_node]["label"] == "user":
                if new_graph.nodes[to_node][args.attribute[0]] == 1:
                    user_list.add(from_node)
                    user_degree_dict[from_node].append(1)
            else:
                if new_graph.nodes[from_node][args.attribute[0]] == 1:
                    user_list.add(to_node)
                    # total_degree += 1
                    user_degree_dict[to_node].append(1)

        for _, v in user_degree_dict.items():
            degree_list.append(sum(v))

    if len(user_list) == 0:
        raise Exception(f"No qualified users can be extracted.")

    if args.agg == "mean":
        result = sum(degree_list) / len(user_list)
    elif args.agg == "max":
        result = max(degree_list)
    elif args.agg == "min":
        result = min(degree_list)
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getCitations(args, new_graph):
    total_citations = defaultdict(list)
    node_attribute = nx.get_node_attributes(new_graph, "year")
    for key, value in node_attribute.items():
        for attribute in args.attribute:
            if value == int(attribute):
                total_citations[attribute].append(new_graph.nodes[key]["citation"])

    if len(total_citations) == 0:
        raise Exception(f"No qualified nodes can be extracted.")

    result = {}
    for attribute, v in total_citations.items():
        if args.agg == "mean":
            result[attribute] = sum(v) / len(v)
        elif args.agg == "max":
            result[attribute] = max(v)
        elif args.agg == "min":
            result[attribute] = min(v)
        else:
            args.logger.error(f"Sorry, we don't support {args.agg}.")
            raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getRatings(args, new_graph, result_dict):
    total_rating = defaultdict(list)

    for key, value in result_dict.items():
        from_node, to_node = key
        if new_graph.nodes[from_node]["label"] == "movie":
            for attribute in args.attribute:
                if new_graph.nodes[from_node][attribute] == 1:
                    total_rating[attribute].append(value)
        elif new_graph.nodes[to_node]["label"] == "movie":
            for attribute in args.attribute:
                if new_graph.nodes[to_node][attribute] == 1:
                    total_rating[attribute].append(value)

    if len(total_rating) == 0:
        raise Exception(f"No qualified edges can be extracted.")

    result = {}
    for attribute, v in total_rating.items():
        if args.agg == "mean":
            result[attribute] = sum(v) / len(v)
        elif args.agg == "max":
            result[attribute] = max(total_rating)
        elif args.agg == "min":
            result[attribute] = min(total_rating)
        elif args.agg == "variance":
            result[attribute] = statistics.variance(total_rating)
        else:
            args.logger.error(f"Sorry, we don't support {args.agg}.")
            raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getAuthors(args, new_graph):
    paper_set = set()
    author_list = []
    node_attribute = nx.get_node_attributes(new_graph, "year")
    for key, value in node_attribute.items():
        if value == int(args.attribute[0]):
            degree = new_graph.degree(key, "writes")
            if degree != 0:
                paper_set.add(key)
                author_list.append(new_graph.degree(key, "writes"))

    if args.agg == "mean":
        if len(paper_set) == 0:
            print(paper_set)
            raise Exception(
                f"no author satisfying {int(args.attribute[0])} by sampling {args.ratio} of {args.dataset}."
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
