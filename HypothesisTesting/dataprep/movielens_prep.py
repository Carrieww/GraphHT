import os
import pickle
import re

import networkx as nx
import pandas as pd
from config import ROOT_DIR

# columns_to_keep = ["article_id", "gender", "cust_id", "location", "age"]
ages_ml = {
    1: "<18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: ">56",
}

decades = {
    "0": "2000's",
    "1": "2010's",
    "2": "2020's",
    "99": "90's",
    "98": "80's",
    "97": "70's",
    "96": "60's",
    "95": "50's",
    "94": "40's",
    "93": "30's",
    "92": "20's",
    "91": "10's",
    "90": "00's",
    "89": "IXX",
    "37": "XIV",
}
occupations = {
    0: "other",
    1: "academic-educator",
    2: "artist",
    3: "clerical-admin",
    4: "college-grad student",
    5: "customer service",
    6: "doctor-health care",
    7: "executive-managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales-marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician-engineer",
    18: "tradesman-craftsman",
    19: "unemployed",
    20: "writer",
}


def movielens_prep(args):
    if not os.path.isfile(
        os.path.join(os.getcwd(), "datasets", "MovieLens1", "graph.pickle")
    ):
        print(f"preparing dataset {args.dataset}.")
        args.logger.info(f"preparing dataset {args.dataset}.")

        df_movies, df_users, df_ratings = get_dataset_movielens()

        # prepare node lists
        movie_list = getNodeList(df_movies)
        graph = nx.Graph()
        graph.add_nodes_from(movie_list)

        user_list = getNodeList(df_users)
        graph.add_nodes_from(user_list)

        # prepare edge lists
        relation_list = getRelationList("user", "movie", df_ratings, graph)
        graph.add_edges_from(relation_list)
        # print(nx.is_connected(graph))

        # get the largest connected component
        if nx.is_connected(graph):
            graph = nx.relabel.convert_node_labels_to_integers(
                graph, first_label=0, ordering="default"
            )
            print("The constructed graph is connected")

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
                os.path.join(os.getcwd(), "datasets", "MovieLens1", "graph.pickle"),
                "wb",
            ),
        )
    else:
        print("loading dataset.")
        graph = pickle.load(
            open(
                os.path.join(os.getcwd(), "datasets", "MovieLens1", "graph.pickle"),
                "rb",
            )
        )
    return graph


def get_dataset_movielens():
    datContent = [
        str(i).strip().split("::")
        for i in open(f"{ROOT_DIR}/../datasets/MovieLens1/movies.dat", "rb").readlines()
    ]
    item = pd.DataFrame(datContent, columns=["movieId", "name", "genres"])

    item.movieId = item.movieId.map(lambda x: x.strip("b"))
    item.movieId = item.movieId.apply(
        lambda x: int(x.split("'")[1])
        if len(x.split("'")) > 1
        else int(x.split('"')[1])
    )
    item.genres = item.genres.map(lambda x: x.split("\\")[0])

    datContent = [
        str(i).strip().split("::")
        for i in open(
            f"{ROOT_DIR}/../datasets/MovieLens1/ratings.dat", "rb"
        ).readlines()
    ]
    item_rating = pd.DataFrame(
        datContent, columns=["userId", "itemId", "rating", "timestamp"]
    )

    item_rating.itemId = item_rating.itemId.astype("int64")
    item_rating.rating = item_rating.rating.astype("int64")
    item_rating.userId = item_rating.userId.map(lambda x: int(x.strip("b'")))
    item_rating.timestamp = item_rating.timestamp.map(lambda x: int(x.split("\\")[0]))

    item_rating.timestamp = pd.to_datetime(item_rating.timestamp, unit="s")

    datContent = [
        str(i).strip().split("::")
        for i in open(f"{ROOT_DIR}/../datasets/MovieLens1/users.dat", "rb").readlines()
    ]

    user = pd.DataFrame(
        datContent, columns=["userId", "Gender", "Age", "Occupation", "Zip"]
    )

    user.userId = user.userId.apply(lambda x: int(x.strip("b'")))
    user.Gender = user.Gender.apply(lambda x: str(x))
    user.Age = user.Age.apply(lambda x: int(x))
    user.Occupation = user.Occupation.apply(lambda x: int(x))

    user = user[["userId", "Gender", "Age", "Occupation"]]
    user.Occupation = user.Occupation.map(lambda x: occupations[x])
    user.Age = user.Age.map(lambda x: ages_ml[x])

    item.rename(columns={"movieId": "itemId"}, inplace=True)
    item_rating.rename(columns={"movieId": "itemId"}, inplace=True)

    movies_links = pd.read_csv(
        f"{ROOT_DIR}/../datasets/MovieLens25/links.csv"
    )  # Link between MovieLen and IMDB
    movies_links.rename(columns={"movieId": "itemId"}, inplace=True)
    movies_links = movies_links[["itemId", "imdbId"]]

    # imdb_movies = pd.read_csv(f'{ROOT_DIR}/../datasets/IMDB/title.basics.tsv',sep='\t') #SHOWS
    imdb_movies = pd.read_csv(
        f"{ROOT_DIR}/../datasets/IMDB/title.basics.tsv",
        sep="\t",
        dtype={"isAdult": object, "startYear": object},
    )  # SHOWS

    imdb_movies.tconst = imdb_movies.tconst.map(lambda x: x.strip("t"))
    imdb_movies.tconst = pd.to_numeric(imdb_movies.tconst)

    imdb_movies.rename(columns={"tconst": "imdbId"}, inplace=True)
    imdb_movies = imdb_movies[["imdbId", "runtimeMinutes", "genres", "startYear"]]

    movies_merged = pd.merge(movies_links, imdb_movies, on="imdbId")

    item.rename(columns={"genres": "genres_first"}, inplace=True)
    item = pd.merge(
        item, movies_merged, on="itemId"
    )  # Merge between MoviesLen and IMDB attr

    list_genres = list(item.genres_first.unique())
    li = item.apply(
        missing_genres, genres=list_genres, axis=1
    )  # Filling the missing genres of MoviesLen by IMDB ones
    item.genres_first = li

    item = item[["itemId", "genres_first", "runtimeMinutes", "startYear"]]
    item.rename(columns={"genres_first": "genres", "startYear": "year"}, inplace=True)

    item = item[item.runtimeMinutes != "\\N"]
    item.runtimeMinutes = pd.to_numeric(item.runtimeMinutes)
    item.runtimeMinutes = item.runtimeMinutes.map(lambda x: run(x / 60))

    item = item[item.year != "\\N"]
    item.year = pd.to_numeric(item.year)
    item.year = item.year.map(lambda x: decades[str(int(x / 10) % 100)])

    item_rating = pd.merge(item_rating, user, on="userId")
    item_rating = pd.merge(item_rating, item, on="itemId")

    df = item_rating

    # df["purchase"] = 1
    df.index = pd.to_datetime(
        pd.to_datetime(df.timestamp, unit="s").dt.date,
    )

    columns = {
        "userId": "user_id",
        "itemId": "movie_id",
        "Gender": "gender",
        "Age": "age",
        "Occupation": "occupation",
        "timestamp": "transaction_date",
    }

    df.rename(columns=columns, inplace=True)
    df = df.reset_index(drop=True)

    df_movies = df.loc[:, ["movie_id", "year", "genres", "runtimeMinutes"]]
    df_movies = df_movies.drop_duplicates().reset_index(drop=True)
    df_movies = genrePreprocess(df_movies, "|")
    df_users = df.loc[:, ["user_id", "gender", "age", "occupation"]]
    df_users = df_users.drop_duplicates().reset_index(drop=True)
    df_ratings = df.loc[
        :, ["movie_id", "user_id", "transaction_date", "rating"]
    ].reset_index(drop=True)

    columns = {
        "movie_id": "to_id",
        "user_id": "from_id",
        "transaction_date": "transaction_date",
        "rating": "rating",
    }
    df_ratings.rename(columns=columns, inplace=True)

    # features2values = dict()
    # for feature in ["age", "gender", "genre", "year", "occupation", "runtimeMinutes"]:
    #     features2values[feature] = list(df[feature].unique())
    return df_movies, df_users, df_ratings  # , features2values  # df.shape=(996656, 11)


def missing_genres(x, genres):
    if "(" in str(x.genres_first):
        g = str(x.genres).split(",")
        genr = ""
        for i in g:
            if i in genres:
                genr = genr + "|" + i
        return genr[1:]
    else:
        return str(x.genres_first)


def run(x):
    if x <= 1:
        return "Short"
    elif x <= 3.5:
        return "Long"
    else:
        return "Very Long"


def getNodeList(df):
    node_list = []
    for _, row in df.iterrows():
        node_attribute = {}
        for k, v in row.items():
            if k[-3:] == "_id":
                node_name = k[:-3] + str(int(v))
                node_attribute["label"] = k[:-3]
            else:
                node_attribute[k] = v
        node_list.append((node_name, node_attribute))

    return node_list


def getRelationList(from_node_name, to_node_name, df, graph):
    edge_list = []
    for _, row in df.iterrows():
        edge_attribute = {}
        for k, v in row.items():
            if k[:-3] == "from":
                from_node = from_node_name + str(int(v))
                assert from_node in graph.nodes(), f"{from_node} is not in g"
            elif k[:-3] == "to":
                to_node = to_node_name + str(int(v))
                assert to_node in graph.nodes(), f"{to_node} is not in g"
            else:
                edge_attribute[k] = v
        # edge_attribute["rate"] = "1"
        edge_list.append((from_node, to_node, edge_attribute))
    return edge_list


def genrePreprocess(df_movies, delimiter):
    movies_df_mod = df_movies.copy()

    genres_list = []
    for index, row in df_movies.iterrows():
        try:
            genres = row.genres.split(delimiter)
            genres_list.extend(genres)
        except:
            genres_list.append(row.genres)

    genres_list = list(set(genres_list))

    for genre in genres_list:  # Creating new columns with names as genres
        movies_df_mod[genre] = 0  # 0 = movie is not considered in that genre

    for index, row in movies_df_mod.iterrows():
        try:
            genres = row.genres.split(
                delimiter
            )  ## Multiple genres for the movie is separated by '|' in the one string; converts to list
        except Exception:
            genres = list(
                row.genres
            )  ## In the case that there is only one genre for the movie

        # Changing all columns that are labelled as genres to 1 if the movie is in that genre:
        if "IMAX" in genres:
            genres.remove("IMAX")

        if "(no genres listed)" in genres:
            genres.remove("(no genres listed)")
            genres.append("None")

        for genre in genres:
            movies_df_mod.loc[index, genre] = 1

    movies_df_mod = movies_df_mod.drop(["genres"], axis=1)
    movies_df_mod["genre"] = movies_df_mod[genres_list].sum(axis=1)

    return movies_df_mod
