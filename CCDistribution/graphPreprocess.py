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
    """For dataset movielens: to obtain movie_list"""
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
    args.logger.info(f"There are {len(movie_list)} movies in the dataset.")
    return movie_list


def getUserList(args, df_movies, df_ratings):
    """For dataset movielens: to obtain user_list"""
    movie = df_movies.movieId
    df_ratings = pd.merge(movie, df_ratings, on="movieId", how="inner")
    user_list = []
    for i in df_ratings.userId.unique():
        node_attribute = {}
        node_attribute["label"] = "user"
        # node_attribute['title'] = row[1]
        node_name = "user" + str(int(i))
        user_list.append((node_name, node_attribute))
    args.logger.info(f"There are {len(user_list)} users in the dataset.")
    return user_list


def getRelationList(args, graph, df_movies, df_ratings):
    """For dataset movielens: to obtain relation_list"""
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
    args.logger.info(f"There are {len(relation_list)} user movie rating relations.")
    return relation_list


def getPaperList(args, df_paper_author):
    """For dataset citation: to obtain paper_list"""
    # df_paper_author
    # print(df_paper_author.shape)
    df_paper_subset = df_paper_author.loc[:, ["paperTitle", "year", "index"]]
    df_paper_subset = df_paper_subset.drop_duplicates()
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

    paper_list = []
    for _, row in df_paper_subset.iterrows():
        # print(row)
        node_attribute = {}
        node_attribute["label"] = "paper"
        node_attribute["title"] = row[paperTitle_index]
        node_attribute["year"] = row[year_index]
        # print(row[9])
        node_name = row[index_index]
        paper_list.append((node_name, node_attribute))
    print(f"There are {len(paper_list)} papers in the dataset.")
    args.logger.info(f"There are {len(paper_list)} papers in the dataset.")
    return paper_list


def getAuthorList(args, df_paper_author):
    """For dataset citation: to obtain author_list"""
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


def getRelationLists(args, graph, df_paper_author, df_paper_paper):
    """For dataset citation: to obtain author_paper_relation_list and paper_paper_relation_list"""
    relation_subset = df_paper_author.loc[:, ["index", "authorId"]]
    if relation_subset.shape != relation_subset.drop_duplicates().shape:
        print(relation_subset[relation_subset.duplicated()])
        args.logger.info(relation_subset[relation_subset.duplicated()])
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
        edge_attribute["label"] = "writes"
        author_paper_relation_list.append((from_node, to_node, edge_attribute))
    print(f"There are {len(author_paper_relation_list)} author paper relations.")
    args.logger.info(
        f"There are {len(author_paper_relation_list)} author paper relations."
    )

    # paper_paper relation
    df_paper_paper_valid = df_paper_paper[~df_paper_paper.isnull().any(axis=1)]
    df_paper_paper_valid["referenceIndex"] = df_paper_paper_valid[
        "referenceIndex"
    ].astype(int)

    attr_index = -1
    for col_name in df_paper_paper_valid.columns:
        attr_index += 1
        if col_name == "index":
            index_index = attr_index
        elif col_name == "referenceIndex":
            referenceId_index = attr_index

    paper_paper_relation_list = []
    for _, row in df_paper_paper_valid.iterrows():
        from_node = row[index_index]
        assert from_node in graph.nodes(), f"{from_node} is not in g."
        to_node = "index" + str(int(row[referenceId_index]))
        assert to_node in graph.nodes(), f"{to_node} is not in g."
        edge_attribute = {}
        edge_attribute["label"] = "relates_to"
        paper_paper_relation_list.append((from_node, to_node, edge_attribute))

    print(f"There are {len(paper_paper_relation_list)} paper paper relations.")
    args.logger.info(
        f"There are {len(paper_paper_relation_list)} paper paper relations."
    )

    return author_paper_relation_list, paper_paper_relation_list
