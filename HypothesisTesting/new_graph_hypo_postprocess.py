import statistics
import time
from collections import defaultdict

import networkx as nx
from scipy import stats

from utils import print_hypo_log


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
            total_result = getGenres(args, new_graph, dimension={"movie": "genre"})
        # edge attribute mean and variance
        elif args.hypo == 3 or args.hypo == 4:
            # result_dict = nx.get_edge_attributes(new_graph, name="rating")
            total_result = getEdges(args, new_graph)
            # return is a list

        ##### two side #####
        elif args.hypo == 10:
            # result_dict = nx.get_edge_attributes(new_graph, name="rating")
            # result is a dictionary
            result = getEdges(args, new_graph)
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
            total_result = getGenres(args, new_graph, dimension={"paper": "citation"})

        # edge attribute
        elif args.hypo == 3:
            total_result = getEdges(args, new_graph)

        ##### two side #####
        elif args.hypo == 10:
            result = getCitations(args, new_graph)
        else:
            raise Exception(f"Sorry, we don't support {args.hypo} for {args.dataset}.")
    elif args.dataset == "yelp":
        if args.hypo == 3:
            # result_dict = nx.get_edge_attributes(new_graph, name="stars")
            total_result = getEdges(args, new_graph)
        elif args.hypo == 2:
            total_result = getGenres(args, new_graph, dimension={"business": "stars"})
        else:
            raise Exception(f"Sorry, we don't support {args.hypo} for {args.dataset}.")

    result = {}

    for attribute, v in total_result.items():
        args.valid_edges.append(len(v))
        # print(len(v))
        # import collections

        # frequency = collections.Counter(v)
        # print(dict(frequency))
        # print(v)

        if args.agg == "mean":
            variance = statistics.variance(v)
            # print(f"The variance is {round(variance,3)}.")
            # args.logger.info(f"The variance is {round(variance,3)}.")
            args.variance.append(variance)

            t_stat, p_value = stats.ttest_1samp(
                v, popmean=args.ground_truth, alternative="two-sided"
            )
            print_hypo_log(args, t_stat, p_value, args.H0, twoSides=True)

            result[attribute] = sum(v) / len(v)
        elif args.agg == "max":
            result[attribute] = max(v)
        elif args.agg == "min":
            result[attribute] = min(v)
        elif args.agg == "variance":
            result[attribute] = statistics.variance(v)
        else:
            args.logger.error(f"Sorry, we don't support {args.agg}.")
            raise Exception(f"Sorry, we don't support {args.agg}.")

    # print(f"avg is {sum(avg_length) / len(avg_length)}")
    result_list.append(result)
    return result_list


def getGenres(args, new_graph, dimension):  # dimension = {"movie":"genre"}
    total_result = defaultdict(list)
    total_result_repeat = defaultdict(list)
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            # The condition is on the dimension node only!
            if (
                len(list(condition_dict)) == 2
                and list(condition_dict.keys())[1] == list(dimension.keys())[0]
            ):
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
                    if flag == True:
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
                    if flag == True:
                        flag = checkCondition(args, condition_dict, new_graph, to_node)
                    else:
                        continue

                    if flag == True:
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
                    total_result[condition_name].append(
                        new_graph.nodes[i][list(dimension.values())[0]]
                    )

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result

    # node_attribute = nx.get_node_attributes(new_graph, "label")
    # for key, value in node_attribute.items():
    #     if value == dimension:

    # total_genre = []
    # genre_list = []

    # for key, value in node_attribute.items():
    #     num_genre = 0
    #     if value == "movie":
    #         if genre_list == []:
    #             genre_list = list(new_graph.nodes[key].keys())[2:]
    #         else:
    #             for genre in genre_list:
    #                 num_genre += new_graph.nodes[key][genre]
    #         total_genre.append(num_genre)

    # # sum(total_genre)/len(total_genre)
    # if len(total_genre) == 0:
    #     total_genre.append(0)

    # if args.agg == "mean":
    #     result = sum(total_genre) / len(total_genre)
    # elif args.agg == "max":
    #     result = max(total_genre)
    # elif args.agg == "min":
    #     result = min(total_genre)
    # else:
    #     args.logger.error(f"Sorry, we don't support {args.agg}.")
    #     raise Exception(f"Sorry, we don't support {args.agg}.")
    # return result


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


def checkCondition(args, condition_dict, new_graph, node_index):
    if (
        new_graph.nodes[node_index]["label"] in condition_dict
    ):  # condition_dict={"node label":{attribute condition}}
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
            if flag == True:
                flag = checkCondition(args, condition_dict, new_graph, to_node)
            else:
                continue

            if flag == True:
                # print(new_graph.nodes[from_node])
                # print(new_graph.nodes[to_node])
                total_result[condition_name].append(value)

    return total_result


def getCorrelation(args, new_graph):
    result_dict = nx.get_edge_attributes(new_graph, name="correlation")
    total_correlation = defaultdict(list)
    print(len(result_dict))

    for key, value in result_dict.items():
        from_node, to_node = key
        # print(from_node, to_node)
        if new_graph.nodes[from_node]["year"] == int(args.attribute[0]):
            for attribute in args.attribute:
                total_correlation[attribute].append(value)

        elif new_graph.nodes[to_node]["year"] == int(args.attribute[0]):
            for attribute in args.attribute:
                total_correlation[attribute].append(value)

    return total_correlation


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
