import os
import pickle
import re

import networkx as nx
import numpy as np
import pandas as pd


def citation_prep(args, author_list_flag=False):
    if not os.path.isfile(
        os.path.join(os.getcwd(), "datasets", args.dataset, "graph.pickle")
    ):
        df_paper_author = pd.read_csv(
            os.path.join(args.dataset_path, "citation_network.csv")
        )
        df_paper_paper = df_paper_author
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        # prepare node list
        graph = nx.Graph()
        paper_list = getPaperList(args, df_paper_author)
        graph.add_nodes_from(paper_list)
        if author_list_flag:
            author_list = getAuthorList(args, df_paper_author)
            graph.add_nodes_from(author_list)
            assert graph.number_of_nodes() == (
                len(df_paper_author.authorId.unique()) + len(paper_list)
            ), f"number of nodes != unique author + unique paper"

        # prepare edge lists
        author_paper_relation_list, paper_paper_relation_list = getRelationLists(
            args, graph, df_paper_paper
        )
        graph.add_edges_from(author_paper_relation_list)
        graph.add_edges_from(paper_paper_relation_list)

        # get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_cc)
        graph = nx.relabel.convert_node_labels_to_integers(
            largest_graph, first_label=0, ordering="default"
        )
        # save the graph
        pickle.dump(
            graph,
            open(
                os.path.join(os.getcwd(), "datasets", args.dataset, "graph.pickle"),
                "wb",
            ),
        )
    else:
        print("loading dataset.")
        graph = pickle.load(
            open(
                os.path.join(os.getcwd(), "datasets", args.dataset, "graph.pickle"),
                "rb",
            )
        )
    return graph


def getPaperList(args, df_paper_author):
    # df_paper_author
    # print(df_paper_author.shape)
    df_paper_subset = df_paper_author.loc[
        :, ["paperTitle", "year", "index", "citation"]
    ]
    df_paper_subset = df_paper_subset.drop_duplicates()
    # df_paper_subset["citation"] = np.random.randint(
    #     0, 5000, size=(df_paper_subset.shape[0],)
    # )

    # print(df_paper_subset.shape)

    attr_index = -1
    for col_name in df_paper_subset.columns:
        attr_index += 1
        if col_name == "paperTitle":
            paperTitle_index = attr_index
        elif col_name == "year":
            year_index = attr_index
        elif col_name == "index":
            index_index = attr_index
        elif col_name == "citation":
            citation_index = attr_index

    paper_list = []
    for _, row in df_paper_subset.iterrows():
        # print(row)
        node_attribute = {}
        node_attribute["label"] = "paper"
        node_attribute["title"] = row[paperTitle_index]
        node_attribute["year"] = row[year_index]
        node_attribute["citation"] = row[citation_index]
        # print(row[9])
        node_name = row[index_index]
        paper_list.append((node_name, node_attribute))
    print(f"There are {len(paper_list)} papers in the dataset.")
    args.logger.info(f"There are {len(paper_list)} papers in the dataset.")
    return paper_list


def getAuthorList(args, df_paper_author):
    df_paper_author["authorId"] = pd.factorize(df_paper_author["author"])[0]
    subset = df_paper_author.loc[:, ["author", "authorId"]]
    subset = subset.drop_duplicates()

    attr_index = -1
    for col_name in subset.columns:
        attr_index += 1
        if col_name == "author":
            author_index = attr_index
        elif col_name == "authorId":
            authorId_index = attr_index

    author_list = []
    for _, row in subset.iterrows():
        node_attribute = {}
        node_attribute["label"] = "author"
        node_attribute["author_name"] = row[author_index]
        node_name = "author" + str(int(row[authorId_index]))
        author_list.append((node_name, node_attribute))
    print(f"There are {len(author_list)} authors in the dataset.")
    args.logger.info(f"There are {len(author_list)} authors in the dataset.")
    return author_list


def getRelationLists(args, graph, df_paper_paper, df_paper_author=None):
    if df_paper_author:
        relation_subset = df_paper_author.loc[:, ["index", "authorId"]]
        if relation_subset.shape != relation_subset.drop_duplicates().shape:
            # print(relation_subset[relation_subset.duplicated()])
            args.logger.error(relation_subset[relation_subset.duplicated()])
            args.logger.error(
                f"There are duplicated paper-author relations, which shall not exist."
            )
            raise Exception(
                f"There are duplicated paper-author relations, which shall not exist."
            )

        # paper_author relation
        attr_index = -1
        for col_name in relation_subset.columns:
            attr_index += 1
            if col_name == "index":
                index_index = attr_index
            elif col_name == "authorId":
                authorId_index = attr_index

        author_paper_relation_list = []
        for _, row in relation_subset.iterrows():
            from_node = "author" + str(int(row[authorId_index]))
            assert from_node in graph.nodes(), f"{from_node} is not in g."
            to_node = row[index_index]
            assert to_node in graph.nodes(), f"{to_node} is not in g."
            edge_attribute = {}
            edge_attribute["writes"] = 1
            # edge_attribute["correlation"] = 0
            edge_attribute["relates_to"] = 0
            author_paper_relation_list.append((from_node, to_node, edge_attribute))
        print(f"There are {len(author_paper_relation_list)} author paper relations.")
        args.logger.info(
            f"There are {len(author_paper_relation_list)} author paper relations."
        )
    else:
        author_paper_relation_list = []

    # paper_paper relation
    df_paper_paper_valid = df_paper_paper.loc[:, ["id", "references"]]
    df_paper_paper_valid = df_paper_paper_valid[
        ~df_paper_paper_valid.isnull().any(axis=1)
    ]
    df_paper_paper_valid = pd.DataFrame(
        df_paper_paper_valid.references.str.split(";").tolist(),
        index=df_paper_paper_valid.id,
    ).stack()
    df_paper_paper_valid = df_paper_paper_valid.reset_index([0, "id"])
    df_paper_paper_valid.columns = ["index", "referenceIndex"]

    df_paper_paper_valid.loc[:, ["referenceIndex"]] = df_paper_paper_valid[
        "referenceIndex"
    ].astype(int)

    df_paper_paper_valid["correlation"] = np.random.rand(
        df_paper_paper_valid.shape[0],
    )

    attr_index = -1
    for col_name in df_paper_paper_valid.columns:
        attr_index += 1
        if col_name == "index":
            index_index = attr_index
        elif col_name == "referenceIndex":
            referenceId_index = attr_index
        # elif col_name == "correlation":
        #     correlationId_index = attr_index

    paper_paper_relation_list = []
    for _, row in df_paper_paper_valid.iterrows():
        from_node = row[index_index]
        assert from_node in graph.nodes(), f"{from_node} is not in g."
        to_node = "index" + str(int(row[referenceId_index]))
        assert to_node in graph.nodes(), f"{to_node} is not in g."
        edge_attribute = {}
        edge_attribute["relates_to"] = 1
        # edge_attribute["correlation"] = round(row[correlationId_index], 2)
        edge_attribute["writes"] = 0
        paper_paper_relation_list.append((from_node, to_node, edge_attribute))

    print(f"There are {len(paper_paper_relation_list)} paper paper relations.")
    args.logger.info(
        f"There are {len(paper_paper_relation_list)} paper paper relations."
    )

    return author_paper_relation_list, paper_paper_relation_list
