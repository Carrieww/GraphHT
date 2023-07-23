import os
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from config import parse_args
from scipy.stats import ttest_1samp
from utils import clean, drawAllRatings, logger, setup_device, setup_seed

# only undirected and unweighted graph


def main():
    clean()
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    logger(args)

    # get the original graph
    if args.dataset == "movielens":
        df_movies = pd.read_csv(
            "/Users/wangyun/Documents/GitHub/GraphHT/datasets/ml-latest-small/movies.csv"
        )
        df_ratings = pd.read_csv(
            "/Users/wangyun/Documents/GitHub/GraphHT/datasets/ml-latest-small/ratings.csv"
        )
        # df_tags = pd.read_csv(
        #     "/Users/wangyun/Documents/GitHub/GraphHT/datasets/ml-latest-small/tags.csv"
        # )
        df_movies = moviePreprocess(df_movies)
        movie_list = getMovieList(args, df_movies)
        graph = nx.Graph()
        graph.add_nodes_from(movie_list)
        user_list = getUserList(df_movies, df_ratings)
        graph.add_nodes_from(user_list)
        relation_list = getRelationList(args, graph, df_movies, df_ratings)

        # initiate graph
        # graph = nx.Graph()
        # graph.add_nodes_from(movie_list)
        # graph.add_nodes_from(user_list)
        graph.add_edges_from(relation_list)
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_graph = graph.subgraph(largest_cc)
        graph = nx.relabel.convert_node_labels_to_integers(
            largest_graph, first_label=0, ordering="default"
        )

    else:
        raise Exception(f"Sorry, we don't support {args.dataset}.")

    # print graph characteristics
    args.num_nodes = graph.number_of_nodes()
    args.num_edges = graph.number_of_edges()
    args.logger.info(
        f"{args.dataset} has {args.num_nodes} nodes and {args.num_edges} edges."
    )
    args.logger.info(f"{args.dataset} is connected: {nx.is_connected(graph)}.")
    args.ground_truth = getGroundTruth(args, df_movies, df_ratings)
    args.CI = []
    args.popmean_lower = round(args.ground_truth - 0.05, 2)
    args.popmean_higher = round(args.ground_truth + 0.05, 2)
    args.result = defaultdict(list)
    for ratio in args.sampling_ratio:
        args.ratio = ratio
        args.logger.info(" ")
        args.logger.info(f">>> sampling ratio: {args.ratio}")
        result_list = samplingGraph(args, graph)
        args.result[ratio] = result_list
        testHypothesis(args, result_list)
        args.logger.info(
            f"The hypothesis testing for {args.ratio} sampling ratio is finished!"
        )
    drawAllRatings(args, args.result)
    print(
        f"All hypothesis testing for ratio list {args.sampling_ratio} and ploting is finished!"
    )


def getGroundTruth(args, df_movies, df_ratings):
    if (args.attribute is not None) and (len(args.attribute) == 1):
        movie = df_movies.loc[:, ["movieId"] + args.attribute]
        df_ratings = pd.merge(movie, df_ratings, on="movieId", how="inner")
        # args.logger.info(df_ratings[df_ratings.loc[:, args.attribute] == 1].shape)
        attribute = args.attribute[0]
        rating = df_ratings[df_ratings[attribute] == 1].rating.values
        # rating = df_.rating.values()
        # for i in args.attribute:
        #     df_ = df_ratings[df_ratings[i] == 1]
    else:
        raise Exception(f"Sorry, args.attribute is None or len(args.attribute) != 1.")

    # check aggregation method
    if args.agg == "mean":
        ground_truth = sum(rating) / rating.shape[0]
    elif args.agg == "max":
        ground_truth = max(rating)
    elif args.agg == "min":
        ground_truth = min(rating)
    else:
        raise Exception(f"Sorry, we don't support {args.agg}.")

    args.logger.info(f"The ground truth rating is {round(ground_truth,2)}.")
    return ground_truth


def testHypothesis(args, result_list):
    # test H1: avg rating = popmean
    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.ground_truth, alternative="two-sided"
    )
    args.logger.info("====================")
    args.logger.info(
        f"H0: {args.agg} rating of {args.attribute} users = {args.ground_truth}."
    )
    args.logger.info(
        f"H1: {args.agg} rating of {args.attribute} users != {args.ground_truth}."
    )
    args.logger.info(f"T-statistic value: {t_stat}, P-value: {p_value}.")
    CI = st.t.interval(
        confidence=0.95,
        df=len(result_list) - 1,
        loc=np.mean(result_list),
        scale=st.sem(result_list),
    )
    args.CI.append(CI)
    args.logger.info(CI)
    if p_value < 0.05:
        args.logger.info(
            f"The test is significant, we shall reject the null hypothesis."
        )
        args.logger.info(f"Population {args.agg} != {args.ground_truth}.")
    else:
        args.logger.info(
            f"The test is NOT significant, we shall accept the null hypothesis."
        )
        args.logger.info(f"Population {args.agg} == {args.ground_truth}.")

    # test H1: avg rating > popmean_lower
    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.popmean_lower, alternative="greater"
    )
    args.logger.info("====================")
    args.logger.info(
        f"H0: {args.agg} rating of {args.attribute} users = {args.popmean_lower}."
    )
    args.logger.info(
        f"H1: {args.agg} rating of {args.attribute} users > {args.popmean_lower}."
    )
    args.logger.info(f"T-statistic value: {t_stat}, P-value: {p_value}.")
    if p_value < 0.05:
        args.logger.info(
            f"The test is significant, we shall reject the null hypothesis."
        )
        args.logger.info(f"Population {args.agg} > {args.popmean_lower}.")
    else:
        args.logger.info(
            f"The test is NOT significant, we shall accept the null hypothesis."
        )
        args.logger.info(
            f"Population {args.agg} is NO larger than {args.popmean_lower}."
        )

    # test H1: avg rating < popmean_higher
    t_stat, p_value = ttest_1samp(
        result_list, popmean=args.popmean_higher, alternative="less"
    )
    args.logger.info("====================")
    args.logger.info(
        f"H0: {args.agg} rating of {args.attribute} users = {args.popmean_higher}."
    )
    args.logger.info(
        f"H1: {args.agg} rating of {args.attribute} users < {args.popmean_higher}."
    )
    args.logger.info(f"T-statistic value: {t_stat}, P-value: {p_value}.")
    if p_value < 0.05:
        args.logger.info(
            f"The test is significant, we shall reject the null hypothesis."
        )
        args.logger.info(f"Population {args.agg} < {args.popmean_higher}.")
    else:
        args.logger.info(
            f"The test is NOT significant, we shall accept the null hypothesis."
        )
        args.logger.info(
            f"Population {args.agg} is NO smaller than {args.popmean_higher}."
        )


def samplingGraph(args, graph):
    result_list = []
    if args.sampling_method == "RNNS":
        from littleballoffur import RandomNodeNeighborSampler

        for num_sample in range(args.num_samples):
            model = RandomNodeNeighborSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            rating_dict = nx.get_edge_attributes(new_graph, name="rating")
            rating = getRatings(args, new_graph, rating_dict)
            result_list.append(rating)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {rating}.")
    elif args.sampling_method == "SRW":
        from littleballoffur import RandomWalkSampler

        for num_sample in range(args.num_samples):
            model = RandomWalkSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            rating_dict = nx.get_edge_attributes(new_graph, name="rating")
            rating = getRatings(args, new_graph, rating_dict)
            result_list.append(rating)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {rating}.")
    elif args.sampling_method == "ShortestPathS":
        from littleballoffur import ShortestPathSampler

        for num_sample in range(args.num_samples):
            model = ShortestPathSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            rating_dict = nx.get_edge_attributes(new_graph, name="rating")
            rating = getRatings(args, new_graph, rating_dict)
            result_list.append(rating)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {rating}.")
    elif args.sampling_method == "MHRS":
        from littleballoffur import MetropolisHastingsRandomWalkSampler

        for num_sample in range(args.num_samples):
            model = MetropolisHastingsRandomWalkSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            rating_dict = nx.get_edge_attributes(new_graph, name="rating")
            rating = getRatings(args, new_graph, rating_dict)
            result_list.append(rating)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {rating}.")
    elif args.sampling_method == "CommunitySES":
        from littleballoffur import CommunityStructureExpansionSampler

        for num_sample in range(args.num_samples):
            model = CommunityStructureExpansionSampler(
                number_of_nodes=int(args.num_nodes * args.ratio),
                seed=(int(args.seed) * num_sample),
            )
            new_graph = model.sample(graph)
            rating_dict = nx.get_edge_attributes(new_graph, name="rating")
            rating = getRatings(args, new_graph, rating_dict)
            result_list.append(rating)
            args.logger.info(f"sample {num_sample}: {args.agg} rating is {rating}.")
    else:
        raise Exception(f"Sorry, we don't support {args.sampling_method}.")
    # drawAvgRating(args, avg_rating)
    return result_list


def getRatings(args, new_graph, rating_dict):
    total_rating = []
    for key, value in rating_dict.items():
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
        raise Exception(f"Sorry, we don't support {args.agg}.")
    return result
    # avg_rating.append(avg)
    # print(avg)


def getMovieList(args, df_movies):
    attr_index = -1
    for col_name in df_movies.columns:
        attr_index += 1
        if col_name == args.attribute[0]:
            break

    movie_list = []
    for _, row in df_movies.iterrows():
        node_attribute = {}
        node_attribute["label"] = "movie"
        node_attribute["title"] = row[1]
        node_attribute[args.attribute[0]] = row[attr_index]
        node_name = "movie" + str(row[0])
        movie_list.append((node_name, node_attribute))

    return movie_list


def getUserList(df_movies, df_ratings):
    movie = df_movies.movieId
    df_ratings = pd.merge(movie, df_ratings, on="movieId", how="inner")
    user_list = []
    for i in df_ratings.userId.unique():
        node_attribute = {}
        node_attribute["label"] = "user"
        # node_attribute['title'] = row[1]
        node_name = "user" + str(int(i))
        user_list.append((node_name, node_attribute))

    return user_list


def getRelationList(args, graph, df_movies, df_ratings):
    movie = df_movies.movieId
    movie = df_movies.loc[:, ["movieId"] + args.attribute]
    df_ratings = pd.merge(movie, df_ratings, on="movieId", how="inner")

    user_index = -1
    for col_name in df_ratings.columns:
        user_index += 1
        if col_name == "userId":
            break
    movie_index = -1
    for col_name in df_ratings.columns:
        movie_index += 1
        if col_name == "movieId":
            break
    attr_index = -1
    for col_name in df_ratings.columns:
        attr_index += 1
        if col_name == "rating":
            break

    relation_list = []
    for _, row in df_ratings.iterrows():
        from_node = "user" + str(int(row[user_index]))
        assert from_node in graph.nodes(), f"{from_node} is not in g"
        to_node = "movie" + str(int(row[movie_index]))
        assert to_node in graph.nodes(), f"{to_node} is not in g"
        edge_attribute = {}
        edge_attribute["rating"] = row[attr_index]
        relation_list.append((from_node, to_node, edge_attribute))
    return relation_list


def moviePreprocess(df_movies):
    movies_df_mod = df_movies.copy()

    movies_df_mod["Year"] = 0
    # movies_df_mod['UPPER_STD'] = 0
    # movies_df_mod['LOWER_STD'] = 0
    # movies_df_mod['AVG_RATING'] = 0
    # movies_df_mod['VIEW_COUNT'] = 0

    # Making the genres into columns:
    ## First, need to obtain a list of all the genres in the dataset.
    #### !!!! Note: "IMAX" is not listed in the readme but is present in the dataset. "Children's" in the readme is "Children" in the dataset.
    genres_list = []
    for index, row in df_movies.iterrows():
        try:
            genres = row.genres.split("|")
            genres_list.extend(genres)
        except:
            genres_list.append(row.genres)

    genres_list = list(set(genres_list))
    genres_list.remove("IMAX")
    genres_list.remove("(no genres listed)")  # Replace with 'None'
    genres_list.append("None")
    for genre in genres_list:  # Creating new columns with names as genres
        movies_df_mod[genre] = 0  # 0 = movie is not considered in that genre

    for index, row in movies_df_mod.iterrows():
        # movieId = row.movieId
        title = row.title

        try:
            genres = row.genres.split(
                "|"
            )  ## Multiple genres for the movie is separated by '|' in the one string; converts to list
        except Exception:
            genres = list(
                row.genres
            )  ## In the case that there is only one genre for the movie

        # print(index)

        # Extracting the year from the title:
        try:  ## Some titles do not have the year--these will be removed downstream to remove the need to access the IMDB API (http://www.omdbapi.com/)
            matcher = re.compile(
                "\(\d{4}\)"
            )  ## Need to extract '(year)' from the title in case there is a year in the title
            parenthesis_year = matcher.search(title).group(0)
            matcher = re.compile(
                "\d{4}"
            )  ## Matching the year from the already matched '(year)'
            year = matcher.search(parenthesis_year).group(0)

            movies_df_mod.loc[index, "Year"] = int(year)

        except Exception:
            pass

        # Changing all columns that are labelled as genres to 1 if the movie is in that genre:
        if "IMAX" in genres:
            genres.remove("IMAX")

        if "(no genres listed)" in genres:
            genres.remove("(no genres listed)")
            genres.append("None")

        for genre in genres:
            movies_df_mod.loc[index, genre] = 1

    movies_df_mod = movies_df_mod[
        movies_df_mod.Year != 0
    ]  ## Removing all movies without years in the title
    movies_df_mod["title"] = movies_df_mod["title"].str.split("(", expand=True)[0]
    movies_df_mod["title"] = movies_df_mod["title"].str[:-1]
    movies_df_mod = movies_df_mod.drop(["genres"], axis=1)
    movies_df_mod.head()
    return movies_df_mod


if __name__ == "__main__":
    main()
