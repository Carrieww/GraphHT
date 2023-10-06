import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx

selected_edge = "rating"
dataset_name = "MovieLens1"


def checkCondition(condition_dict, g, node_index):
    if (
        g.nodes[node_index]["label"] in condition_dict
    ):  # condition_dict={"node label":{attribute condition}}
        attribute_condition_dict = condition_dict[g.nodes[node_index]["label"]]
        for k, v in attribute_condition_dict.items():  # {"gender":"M"}}
            if g.nodes[node_index][k] == v:
                pass
            else:
                return False
    else:
        return True
    return True


def get_list(g, condition_dict):
    edge_dict = nx.get_edge_attributes(g, name=selected_edge)
    num_list = []
    for condition in condition_dict:
        # all_keys = []
        all_values = []
        for key, value in edge_dict.items():
            from_node, to_node = key
            flag = checkCondition(condition, g, from_node)
            if flag:
                flag = checkCondition(condition, g, to_node)
            else:
                continue

            if flag:
                # all_keys.append(key)
                all_values.append(value)
        num = len(all_values)
        num_list.append(num)
        print(f"Number of edges satisfying {condition} is {num}.")
        print(f"The true value of average rating is {round(sum(all_values) / num, 2)}.")
    # print(sum(num_list))
    # print(len(edge_dict))
    if len(edge_dict) != sum(num_list):
        print(
            f"There exists movies with more than one genres because there are {len(edge_dict)} edges and {sum(num_list)} movies if we count them by genres."
        )
    return num_list


def plot_bar_chart(values, labels, num_edges, subject, img_output_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, values, color="skyblue")

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        percentage = (height / num_edges) * 100
        ax.annotate(
            f"{int(height)} ({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 1),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add a grid for better visualization
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Set axis labels and title
    ax.set_ylabel("Number of Edges")
    ax.set_title(f"{subject} Distribution of {dataset_name}")

    # Save the bar chart
    plt.tight_layout()
    plt.savefig(img_output_path)
    plt.close()  # clear the previous axis


def get_movieLength_plot(g, num_edges, output_filename="length_bar.png"):
    """This function is used to plot the bar chart of edges related to various movie lengths"""
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)

    condition_dict = []
    labels = ["Short", "Long", "Very Long"]
    for i in labels:
        condition_dict.append({"movie": {"runtimeMinutes": i}})
    # print(condition_dict)

    length_num_list = get_list(g, condition_dict)

    # Bar chart
    plot_bar_chart(
        length_num_list, labels, num_edges, "Length", img_output_path=img_output_path
    )


def get_genre_plot(g, num_edges, output_filename="genre_bar.png"):
    """This function is used to plot the bar chart of edges related to various genre"""
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)

    sample_node = g.nodes[0]
    condition_dict = []
    labels = []
    for k, v in sample_node.items():
        if isinstance(v, int) and k != "genre":
            labels.append(k)
            condition_dict.append({"movie": {k: 1}})
    # print(condition_dict)

    genre_num_list = get_list(g, condition_dict)

    # Bar chart
    plot_bar_chart(
        genre_num_list, labels, num_edges, "Genre", img_output_path=img_output_path
    )


def get_year_plot(g, num_edges, output_filename="year_bar.png"):
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)

    labels = [
        "10's",
        "20's",
        "30's",
        "40's",
        "50's",
        "60's",
        "70's",
        "80's",
        "90's",
        "2000's",
    ]

    condition_dict = []
    for i in labels:
        condition_dict.append({"movie": {"year": i}})

    year_num_list = get_list(g, condition_dict)

    # Bar chart
    plot_bar_chart(
        year_num_list, labels, num_edges, "Year", img_output_path=img_output_path
    )


def get_gender_plot(g, output_filename="gender_pie.png"):
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)
    condition_dict = [{"user": {"gender": "M"}}, {"user": {"gender": "F"}}]

    gender_num_list = get_list(g, condition_dict)

    # Pie chart
    labels = ["Male", "Female"]
    colors = ["skyblue", "#ff99cc"]

    fig = plt.figure(figsize=(10, 7))
    plt.pie(
        gender_num_list, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
    )
    plt.title(f"Gender distribution of {dataset_name}")
    plt.savefig(img_output_path)
    plt.close()  # clear the previous axis


def get_age_plot(g, output_filename="age_pie.png"):
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)

    condition_dict = []
    labels = ["18-24", "25-34", "35-44", "45-49", "50-55", "<18", ">56"]
    for i in labels:
        condition_dict.append({"user": {"age": i}})

    age_num_list = get_list(g, condition_dict)

    # Pie chart
    colors = [
        "#ff99cc",
        "skyblue",
        "#ffcc99",
        "#66c2a5",
        "#c2a5a5",
        "#99ffcc",
        "#ffb3e6",
    ]

    fig = plt.figure(figsize=(10, 7))
    plt.pie(
        age_num_list, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
    )
    plt.title(f"Age distribution of {dataset_name}")
    plt.savefig(img_output_path)
    plt.close()  # clear the previous axis


def get_occupation_plot(g, num_edges, output_filename="occupation_bar.png"):
    img_output_path = os.path.join("StatisticsAnalysis", dataset_name, output_filename)

    labels = set()
    for node in g.nodes():
        if g.nodes[node]["label"] == "user":
            labels.add(g.nodes[node]["occupation"])
    labels = list(labels)
    condition_dict = []
    for i in labels:
        condition_dict.append({"user": {"occupation": i}})

    labels.sort()
    occupation_num_list = get_list(g, condition_dict)

    # Bar chart
    plot_bar_chart(
        occupation_num_list,
        labels,
        num_edges,
        "Occupation",
        img_output_path=img_output_path,
    )


if __name__ == "__main__":
    graph = pickle.load(
        open(
            os.path.join(os.getcwd(), "datasets", dataset_name, "graph.pickle"),
            "rb",
        )
    )
    num_edges = graph.number_of_edges()

    # plot movie genre bar chart
    if not os.path.isfile(
        os.path.join(os.getcwd(), "StatisticsAnalysis", "MovieLens1", "genre_bar.png")
    ):
        print("start preparing the bar chart for movie genres.")
        get_genre_plot(graph, num_edges)

    # plot movie length categories' bar chart
    if not os.path.isfile(
        os.path.join(os.getcwd(), "StatisticsAnalysis", "MovieLens1", "length_bar.png")
    ):
        print("start preparing the bar chart for movie lengths.")
        get_movieLength_plot(graph, num_edges)

    # plot movie year bar chart
    if not os.path.isfile(
        os.path.join(os.getcwd(), "StatisticsAnalysis", "MovieLens1", "year_bar.png")
    ):
        print("start preparing the bar chart for movie years.")
        get_year_plot(graph, num_edges)

    # plot user gender pie chart
    if not os.path.isfile(
        os.path.join(os.getcwd(), "StatisticsAnalysis", "MovieLens1", "gender_pie.png")
    ):
        print("start preparing the pie chart for user genders.")
        get_gender_plot(graph)

    # plot user age pie chart
    if not os.path.isfile(
        os.path.join(os.getcwd(), "StatisticsAnalysis", "MovieLens1", "age_pie.png")
    ):
        print("start preparing the pie chart for user ages.")
        get_age_plot(graph)

    # plot user occupation bar chart
    if not os.path.isfile(
        os.path.join(
            os.getcwd(),
            "StatisticsAnalysis",
            "MovieLens1",
            "occupation_bar.png",
        )
    ):
        print("start preparing the bar chart for user occupations.")
        get_occupation_plot(graph, num_edges)
