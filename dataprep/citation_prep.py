import os
import pickle
import re
import math
import networkx as nx
import numpy as np
import pandas as pd
from config import ROOT_DIR


def citation_prep(args):
    """
    A function for preparing the DBLP citation graph
    """
    graph_pickle_path = os.path.join(
        ROOT_DIR, "../datasets", args.dataset, "graph.pickle"
    )
    if not os.path.isfile(graph_pickle_path):
        # prepare node list
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        graph = nx.Graph()
        dfs = ["papers", "authors", "fos", "venues"]
        for df_name in dfs:
            df = pd.read_csv(
                os.path.join(ROOT_DIR, "../datasets", args.dataset, f"{df_name}.csv")
            )
            node_list = getNodeList(df)
            graph.add_nodes_from(node_list)

        print("preparing relationships")
        dfs_relations = [
            ["paper_paper", "references"],
            ["paper_author", "author_id"],
            ["paper_venue", "venue_id"],
            ["paper_fos", "fos_id"],
        ]
        for df_name in dfs_relations:
            df = pd.read_csv(
                os.path.join(ROOT_DIR, "../datasets", args.dataset, f"{df_name[0]}.csv")
            )
            relation_list = getRelationList("id", df_name[1], df, graph)
            graph.add_edges_from(relation_list)

        # get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_cc)
        graph = nx.relabel.convert_node_labels_to_integers(
            largest_graph, first_label=0, ordering="default"
        )

        # save the graph
        print("Number of nodes:", graph.number_of_nodes())
        print("Number of edges:", graph.number_of_edges())
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(graph, f)
    else:
        print("Loading dataset.")
        with open(graph_pickle_path, "rb") as f:
            graph = pickle.load(f)

    return graph


def getNodeList(df):
    node_list = []
    for _, row in df.iterrows():
        node_attribute = {}
        for k, v in row.items():
            if k == "id":
                node_name = v
                node_attribute["label"] = "paper"
            if k == "author_org":
                if pd.isna(v):
                    node_attribute[k] = ""
                else:
                    attr = re.sub(r"#N#|#TAB#", "", v)
                    node_attribute[k] = attr
            elif k[-3:] == "_id":
                node_name = v
                node_attribute["label"] = k[:-3]
            else:
                node_attribute[k] = v
        node_list.append((node_name, node_attribute))

    return node_list


def getRelationList(from_node_name, to_node_name, df, graph):
    edge_list = []
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        continue_row = False
        edge_attribute = {}
        # Check if the key corresponds to the 'from' or 'to' node
        for k, v in row.items():
            if k == from_node_name:
                from_node = v
                if from_node not in graph.nodes():
                    continue_row = True
                    break
            elif k == to_node_name:
                to_node = v
                if to_node not in graph.nodes():
                    continue_row = True
                    break
            else:
                edge_attribute[k] = v

        # If the flag is set, skip the current row
        if continue_row == True:
            continue

        # Determine the label of the edge based on the 'from' and 'to' node names
        if from_node_name == "id" and to_node_name == "references":
            edge_attribute["label"] = "cite"
        elif from_node_name == "id" and to_node_name == "author_id":
            edge_attribute["label"] = "write_by"
        elif from_node_name == "id" and to_node_name == "venue_id":
            edge_attribute["label"] = "publish_in"
        elif from_node_name == "id" and to_node_name == "fos_id":
            edge_attribute["label"] = "belong_to"

        edge_list.append((from_node, to_node, edge_attribute))
    return edge_list
