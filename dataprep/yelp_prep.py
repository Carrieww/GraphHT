import os
import pickle
import networkx as nx
import pandas as pd
from config import ROOT_DIR
from dataprep.movielens_prep import getNodeList, getRelationList


def yelp_prep(args):
    """
    A function for preparing the Yelp graph
    """
    graph_pickle_path = os.path.join(
        ROOT_DIR, "../datasets", args.dataset, "graph.pickle"
    )
    # check if the graph is already prepared
    if not os.path.isfile(graph_pickle_path):
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        # get dataframes for businesses, users, and reviews
        df_business, df_user, df_review = get_dataset_yelp(args)

        # prepare node lists for business and users
        print("start preparing business nodes")
        business_df = getNodeList(df_business)
        print("start preparing user nodes")
        user_list = getNodeList(df_user)

        # create an empty graph and add nodes (businesses and users) to the graph
        graph = nx.Graph()
        graph.add_nodes_from(business_df)
        graph.add_nodes_from(user_list)

        # prepare edge lists based on reviews.
        # Graph is inputted for asserting that nodes in the relation list are already in the graph.
        print("start preparing edges")
        relation_list = getRelationList("user", "business", df_review, graph)
        graph.add_edges_from(relation_list)

        # check if the graph is connected
        if nx.is_connected(graph):
            print("The constructed graph is connected")
        else:
            print("The constructed graph is NOT connected")

            # get the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_graph = graph.subgraph(largest_cc)

            # relabel nodes to have integer labels
            graph = nx.relabel.convert_node_labels_to_integers(
                largest_graph, first_label=0, ordering="default"
            )

        # save the graph
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(graph, f)
    else:
        print("Loading dataset.")
        with open(graph_pickle_path, "rb") as f:
            graph = pickle.load(f)
    return graph


def get_dataset_yelp(args):
    """A function to retrieve and prepare the Yelp dataframes from raw files"""
    business_cols = [
        "city",
        "state",
        "stars",
        "business_id",
        "review_count",
        "name",
        "categories",
    ]
    user_cols = [
        "average_stars",
        "fans",
        "useful",
        "funny",
        "cool",
        "user_id",
        "review_count",
        "compliment_writer",
    ]
    review_cols = ["stars", "useful", "cool", "funny", "user_id", "business_id"]

    # load and prepare business data
    df_business = pd.read_csv(
        f"{ROOT_DIR}/../datasets/"
        + args.dataset
        + "/yelp_academic_dataset_business.csv"
    )[business_cols].reset_index(drop=True)
    df_business.loc[df_business["categories"].isna(), "categories"] = ""
    df_business["business_id_new"] = (
        df_business["business_id"].astype("category").cat.codes
    )

    # Add a 'popularity' column based on 'review_count'
    df_business["popularity"] = pd.cut(
        df_business["review_count"],
        bins=[-1, 29, 49, float("inf")],
        labels=["low", "medium", "high"],
    )

    # load and prepare user data
    df_user = pd.read_csv(
        f"{ROOT_DIR}/../datasets/" + args.dataset + "/yelp_academic_dataset_user.csv"
    )[user_cols].reset_index(drop=True)
    df_user["user_id_new"] = df_user["user_id"].astype("category").cat.codes

    # Add a 'prolificacy' column based on 'review_count'
    df_user["prolificacy"] = pd.cut(
        df_user["review_count"],
        bins=[-1, 2, 9, float("inf")],
        labels=["low", "medium", "high"],
    )
    # Add a 'popularity' column based on 'fans'
    df_user["popularity"] = pd.cut(
        df_user["fans"], bins=[-1, 0, 3, float("inf")], labels=["low", "medium", "high"]
    )

    # load and prepare review data
    df_review = pd.read_csv(
        f"{ROOT_DIR}/../datasets/" + args.dataset + "/yelp_academic_dataset_review.csv"
    )[review_cols].reset_index(drop=True)
    df_review = df_review.merge(
        df_business.loc[:, ["business_id", "business_id_new"]],
        how="left",
        on="business_id",
    )
    df_review = df_review.merge(
        df_user.loc[:, ["user_id", "user_id_new"]], how="left", on="user_id"
    )
    # Rename columns and drop rows with NaN values in df_review
    df_user = df_user.drop("user_id", axis=1).rename(columns={"user_id_new": "user_id"})
    df_business = df_business.drop("business_id", axis=1).rename(
        columns={"business_id_new": "business_id"}
    )
    df_review = df_review.drop(["business_id", "user_id"], axis=1)
    df_review = df_review.rename(
        columns={"business_id_new": "to_id", "user_id_new": "from_id"}
    ).dropna()

    return df_business, df_user, df_review  # , features2values  # df.shape=(996656, 11)
