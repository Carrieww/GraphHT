import re

import pandas as pd


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


def getMovieList(args, df_movies):
    # attr_index = -1
    # for col_name in df_movies.columns:
    #     attr_index += 1
    #     if col_name == args.attribute[0]:
    #         break

    movie_list = []
    for _, row in df_movies.iterrows():
        node_attribute = {}
        node_attribute["label"] = "movie"
        node_attribute["title"] = row[1]
        # node_attribute[args.attribute[0]] = row[attr_index]
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
    # movie = df_movies.loc[:, ["movieId"] + args.attribute]
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
        # edge_attribute["rating"] = row[attr_index]
        relation_list.append((from_node, to_node, edge_attribute))
    return relation_list
