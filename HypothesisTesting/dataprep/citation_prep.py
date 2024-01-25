import os
import pickle
import re
import math
import networkx as nx
import numpy as np
import pandas as pd
from config import ROOT_DIR


def citation_prep(args):
    # args.dataset = "DBLP-v12"
    if not os.path.isfile(
        os.path.join(ROOT_DIR, "../datasets", args.dataset, "graph.pickle")
    ):
        print(ROOT_DIR)
        # prepare node list
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        graph = nx.Graph()
        df_papers = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "papers.csv")
        )
        paper_list = getNodeList(df_papers)
        graph.add_nodes_from(paper_list)
        paper_list = []
        # print(len(paper_list))

        df_authors = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "authors.csv")
        )
        author_list = getNodeList(df_authors)
        graph.add_nodes_from(author_list)
        author_list = []
        # print(len(author_list))

        df_fos = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "fos.csv")
        )
        fos_list = getNodeList(df_fos)
        graph.add_nodes_from(fos_list)
        fos_list = []

        df_venues = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "venues.csv")
        )
        venue_list = getNodeList(df_venues)
        graph.add_nodes_from(venue_list)
        venue_list = []

        print("preparing relationships")
        print(graph.number_of_nodes())
        print("preparing paper paper")

        df_paper_paper = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "paper_paper.csv")
        )
        paper_paper_list = getRelationList("id", "references", df_paper_paper, graph)
        graph.add_edges_from(paper_paper_list)
        paper_paper_list = []

        print("preparing paper author")
        df_paper_author = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "paper_author.csv")
        )
        paper_author_list = getRelationList("id", "author_id", df_paper_author, graph)
        graph.add_edges_from(paper_author_list)
        paper_author_list = []

        print("preparing paper venue")
        df_paper_venue = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "paper_venue.csv")
        )
        paper_venue_list = getRelationList("id", "venue_id", df_paper_venue, graph)
        graph.add_edges_from(paper_venue_list)
        paper_venue_list = []

        print("preparing paper fos")
        df_paper_fos = pd.read_csv(
            os.path.join(ROOT_DIR, "../datasets", args.dataset, "paper_fos.csv")
        )
        paper_fos_list = getRelationList("id", "fos_id", df_paper_fos, graph)
        graph.add_edges_from(paper_fos_list)
        paper_fos_list = []

        # get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_cc)
        graph = nx.relabel.convert_node_labels_to_integers(
            largest_graph, first_label=0, ordering="default"
        )

        # save the graph
        print(graph.number_of_nodes())
        print(graph.number_of_edges())
        pickle.dump(
            graph,
            open(
                os.path.join(ROOT_DIR, "../datasets", args.dataset, "graph.pickle"),
                "wb",
            ),
        )
    else:
        print("loading dataset.")
        graph = pickle.load(
            open(
                os.path.join(ROOT_DIR, "../datasets", args.dataset, "graph.pickle"),
                "rb",
            )
        )
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
                # print("=============")
                # print(v)
                if pd.isna(v):
                    node_attribute[k] = ""
                else:
                    attr = re.sub(r"#N#|#TAB#", "", v)
                    # print(attr)
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
    count = 0
    for _, row in df.iterrows():
        continue_row = False
        count += 1
        edge_attribute = {}
        for k, v in row.items():
            if k == from_node_name:
                from_node = v
                if from_node not in graph.nodes():
                    # print(f"{from_node} is not in g")
                    continue_row = True
                    break
            elif k == to_node_name:
                to_node = v
                if to_node not in graph.nodes():
                    # print(f"{to_node} is not in g")
                    continue_row = True
                    break
            else:
                edge_attribute[k] = v

        if continue_row == True:
            continue

        if from_node_name == "id" and to_node_name == "references":
            edge_attribute["label"] = "cite"
        elif from_node_name == "id" and to_node_name == "author_id":
            edge_attribute["label"] = "write_by"
        elif from_node_name == "id" and to_node_name == "venue_id":
            edge_attribute["label"] = "publish_in"
        elif from_node_name == "id" and to_node_name == "fos_id":
            edge_attribute["label"] = "belong_to"

        if count % 100000 == 0:
            print(count)
        edge_list.append((from_node, to_node, edge_attribute))
    return edge_list
