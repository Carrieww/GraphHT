import networkx as nx


def new_graph_hypo_result(args, new_graph, result_list, num_sample):
    if args.dataset == "movielens":
        result_dict = nx.get_edge_attributes(new_graph, name="rating")
        result = getRatings(args, new_graph, result_dict)
        args.logger.info(f"sample {num_sample}: {args.agg} rating is {result}.")
    elif args.dataset == "citation":
        result_dict = nx.get_edge_attributes(new_graph, name="label")
        result = getAuthors(args, new_graph, result_dict)
        args.logger.info(
            f"sample {num_sample}: {args.agg} number of author is {result}."
        )
    result_list.append(result)
    return result_list


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
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result


def getAuthors(args, new_graph, result_dict):
    count_author = 0
    paper_set = set()
    if args.agg == "mean":
        for key, value in result_dict.items():
            if value == "writes":
                from_id, to_id = key
                # print(new_graph.edges[(from_id, to_id)])
                # node=new_graph.nodes[authorId]
                if (
                    new_graph.nodes[from_id]["label"] == "author"
                    and new_graph.nodes[to_id]["year"] == args.attribute
                ):
                    paper_set.add(to_id)
                    # print("true")
                    count_author += 1
                elif (
                    new_graph.nodes[from_id]["label"] == "paper"
                    and new_graph.nodes[from_id]["year"] == args.attribute
                ):
                    paper_set.add(from_id)
                    count_author += 1
                else:
                    pass

        if len(paper_set) == 0:
            args.logger.error(
                f"The graph contains no node/edge satisfying the hypothesis, you may need to increase the sampling ratio."
            )
            raise Exception(
                f"The graph contains no node/edge satisfying the hypothesis, you may need to increase the sampling ratio."
            )

        avg = count_author / len(paper_set)
        args.logger.info(
            f"There are {len(paper_set)} papers and {count_author} authors"
        )
        return avg

    elif args.agg == "max":
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    elif args.agg == "min":
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
    else:
        args.logger.error(f"Sorry, we don't support {args.agg}.")
        raise Exception(f"Sorry, we don't support {args.agg}.")
