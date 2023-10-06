import os
import pickle
import re

import networkx as nx
import pandas as pd
from config import ROOT_DIR
from dataprep.movielens_prep import getNodeList, getRelationList


def yelp_prep(args):
    if not os.path.isfile(
        os.path.join(ROOT_DIR, "../datasets", "yelp_dataset", "graph.pickle")
    ):
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        df_business, df_user, df_review = get_dataset_yelp()

        # prepare node lists
        movie_list = getNodeList(df_business)
        graph = nx.Graph()
        graph.add_nodes_from(movie_list)

        user_list = getNodeList(df_user)
        graph.add_nodes_from(user_list)

        # prepare edge lists
        relation_list = getRelationList("user", "business", df_review, graph)
        graph.add_edges_from(relation_list)
        # print(nx.is_connected(graph))

        # get the largest connected component
        if nx.is_connected(graph):
            print("The constructed graph is connected")
            pass
        else:
            print("The constructed graph is NOT connected")
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_graph = graph.subgraph(largest_cc)
            graph = nx.relabel.convert_node_labels_to_integers(
                largest_graph, first_label=0, ordering="default"
            )

        # save the graph
        pickle.dump(
            graph,
            open(
                os.path.join(
                    os.getcwd(), "../datasets", "yelp_dataset", "graph.pickle"
                ),
                "wb",
            ),
        )
    else:
        print("loading dataset.")
        graph = pickle.load(
            open(
                os.path.join(ROOT_DIR, "../datasets", "yelp_dataset", "graph.pickle"),
                "rb",
            )
        )
    return graph


def get_dataset_yelp():
    ### business dataframe
    df_business = pd.read_csv(
        f"{ROOT_DIR}/../datasets/yelp_dataset/yelp_academic_dataset_business.csv"
    )
    df_business = df_business.loc[
        :, ["city", "state", "stars", "business_id", "review_count"]
    ].reset_index(drop=True)

    df_business["business_id_new"] = (
        df_business["business_id"].astype("category").cat.codes
    )
    # columns = {"categories": "genres"}

    # df_business.rename(columns=columns, inplace=True)
    # df_business = genrePreprocess(df_business, ",")

    ### user dataframe
    df_user = pd.read_csv(
        f"{ROOT_DIR}/../datasets/yelp_dataset/yelp_academic_dataset_user.csv"
    )
    df_user = df_user.loc[
        :,
        [
            "average_stars",
            "fans",
            "useful",
            "funny",
            "cool",
            "user_id",
            "review_count",
            "compliment_writer",
        ],
    ].reset_index(drop=True)
    df_user["user_id_new"] = df_user["user_id"].astype("category").cat.codes

    ### review dataframe
    df_review = pd.read_csv(
        f"{ROOT_DIR}/../datasets/yelp_dataset/yelp_academic_dataset_review.csv"
    )
    df_review = df_review.loc[
        :, ["stars", "useful", "cool", "funny", "user_id", "business_id"]
    ]
    df_review = df_review.merge(
        df_business.loc[:, ["business_id", "business_id_new"]],
        how="left",
        on="business_id",
    )
    df_review = df_review.merge(
        df_user.loc[:, ["user_id", "user_id_new"]], how="left", on="user_id"
    )
    df_user = df_user.drop("user_id", axis=1)
    columns = {"user_id_new": "user_id"}
    df_user.rename(columns=columns, inplace=True)
    df_business = df_business.drop("business_id", axis=1)
    columns = {"business_id_new": "business_id"}
    df_business.rename(columns=columns, inplace=True)
    df_review = df_review.drop(["business_id", "user_id"], axis=1)
    columns = {"business_id_new": "to_id", "user_id_new": "from_id"}
    df_review.rename(columns=columns, inplace=True)

    # remove nan in df_review
    if (
        df_review["from_id"].isnull().sum() != 0
        or df_review["to_id"].isnull().sum() != 0
    ):
        df_review = df_review.dropna()

    return df_business, df_user, df_review  # , features2values  # df.shape=(996656, 11)


# def getNodeList(df):
#     node_list = []
#     for _, row in df.iterrows():
#         node_attribute = {}
#         for k, v in row.items():
#             if k[-3:] == "_id":
#                 node_name = k[:-3] + str(int(v))
#                 node_attribute["label"] = k[:-3]
#             else:
#                 node_attribute[k] = v
#         node_list.append((node_name, node_attribute))

#     return node_list


# def getRelationList(from_node_name, to_node_name, df, graph):
#     edge_list = []
#     for _, row in df.iterrows():
#         edge_attribute = {}
#         for k, v in row.items():
#             if k[:-3] == "from":
#                 from_node = from_node_name + str(int(v))
#                 assert from_node in graph.nodes(), f"{from_node} is not in g"
#             elif k[:-3] == "to":
#                 to_node = to_node_name + str(int(v))
#                 assert to_node in graph.nodes(), f"{to_node} is not in g"
#             else:
#                 edge_attribute[k] = v
#         edge_list.append((from_node, to_node, edge_attribute))
#     return edge_list
