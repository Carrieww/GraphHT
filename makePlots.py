import matplotlib.pyplot as plt
import os

dataset_name = "citation"
hypo_type = "3"
hypo_name = "Path"
# Accuracy Time CI P-value
y_label = "Accuracy"
file_code = "APPA_MS"
plot_code = file_code + ""
# accuracy = 3; time = 0; p-value = 8
# For CI does not need to specify target column (But need to provide true_y);
target_column = 3
true_y = 0.45

nonexist_sm = []
unwanted_row = [9, 10, 11, 12]
x = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2.5, 5]
x_ticks = [1, 2, 3, 4, 5]

current_directory = os.getcwd()
file_name = (
    current_directory
    + "/result/log_and_results_"
    + hypo_type
    + "-1-1/"
    + dataset_name
    + "_hypo"
    + hypo_type
    + "_"
    + hypo_type
    + "-1-1_"
)

save_path = (
    current_directory
    + "/result/log_and_results_"
    + hypo_type
    + "-1-1/"
    + dataset_name
    + "_"
    + hypo_type
    + "-1-1_"
    + y_label.lower()
    + "_plot_"
    + plot_code
    + ".png"
)

sm = [
    "RES",
    # "NBRW",
    # "ShortestPathS",
    # "RW_Starter",
    # "ours",
]

# Create a list of colors from the color map
num_colors = len(sm)
colors = ["#1f77b4", "#006400", "#4B0082", "#8B4513"]
ours_color = "#FF4500"

linestyle = ["--", ":", "-.", "-"]
markerstyle = ["o", "v", ">", "X", "D"]

###############################################
############## can change above ###############
###############################################

index = 0
plt.figure(figsize=(8, 6))

if y_label in ["Accuracy", "Time"]:
    for sampler in sm:
        if sampler in nonexist_sm:
            index += 1
            continue

        filename = file_name + str(sampler) + "_mean_" + file_code + ".txt"
        accuracy_values = []

        with open(filename, "r") as file:
            next(file)  # Skip header
            ind = 0
            for line in file:
                ind += 1
                parts = line.split()
                if ind in unwanted_row:
                    pass
                else:
                    accuracy = float(parts[target_column])
                    accuracy_values.append(accuracy)

        # Plotting the accuracy values
        if sampler == "ours":
            plt.plot(
                x,
                accuracy_values,
                marker="*",
                linestyle="-",
                label=sampler,
                color=ours_color,
                linewidth=3,
                markersize=(14),
            )
        else:
            plt.plot(
                x,
                accuracy_values,
                marker=markerstyle[index % 10],
                linestyle=linestyle[index % 4],
                label=sampler,
                linewidth=3,
                color=colors[index],
                markersize=(12),
            )
        index += 1

elif y_label == "P-value":
    for sampler in sm:
        if sampler in nonexist_sm:
            index += 1
            continue
        filename = file_name + str(sampler) + "_mean_" + file_code + ".txt"
        accuracy_values = []
        x_ = x

        with open(filename, "r") as file:
            next(file)
            ind = 0
            for line in file:
                ind += 1
                parts = line.split()
                if ind in nonexist_sm:
                    pass
                else:
                    accuracy = float(parts[target_column])
                    if accuracy != -1:
                        accuracy_values.append(accuracy)
                    else:
                        x_ = x_[1:]

        # Plotting the accuracy values
        if sampler == "ours":
            plt.plot(
                x_,
                accuracy_values,
                marker="*",
                linestyle="-",
                label=sampler,
                linewidth=3,
                markersize=(14),
                color=ours_color,
            )
        else:
            plt.plot(
                x_,
                accuracy_values,
                marker=markerstyle[index % 10],
                linestyle=linestyle[index % 3],
                label=sampler,
                linewidth=3,
                color=colors[index],
                markersize=(12),
            )
        index += 1
elif y_label == "CI":
    for sampler in sm:
        if sampler in nonexist_sm:
            index += 1
            continue
        filename = file_name + str(sampler) + "_mean_" + file_code + ".txt"
        lower_list = []
        upper_list = []
        x_ = x

        with open(filename, "r") as file:
            next(file)  # Skip header
            ind = 0
            for line in file:
                ind += 1
                parts = line.split()
                if ind in nonexist_sm:
                    pass
                else:
                    lower = float(parts[6])
                    upper = float(parts[7])
                    if lower != -1 and upper != -1:
                        lower_list.append(lower)
                        upper_list.append(upper)
                    else:
                        x_ = x_[1:]

        # Plotting the CI values
        if sampler == "ours":
            plt.plot(
                x_,
                lower_list,
                color=ours_color,
                linewidth=3,
                linestyle="-",
            )
            plt.plot(
                x_,
                upper_list,
                color=ours_color,
                linewidth=3,
                linestyle="-",
            )

            # Filling the area between the bounds
            plt.fill_between(
                x_, lower_list, upper_list, color="r", alpha=0.3, label=sampler
            )
        else:
            plt.plot(
                x_,
                lower_list,
                color=colors[index],
                linewidth=3,
            )
            plt.plot(
                x_,
                upper_list,
                color=colors[index],
                linewidth=3,
            )

            # Filling the area between the bounds
            plt.fill_between(
                x_,
                lower_list,
                upper_list,
                color=colors[index],
                alpha=0.3,
                label=sampler,
            )

        index += 1
    plt.axhline(
        y=true_y, color="black", linestyle="--", linewidth=3, label="ground truth"
    )


plt.xticks(x_ticks, x_ticks)
plt.xlabel("Sampling Proportion (%)", fontsize=18, fontweight="bold")
plt.ylabel(y_label, fontsize=18, fontweight="bold")

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
if y_label == "Time":
    plt.legend(loc="upper left", prop={"size": 18, "weight": "bold"})
elif y_label == "Accuracy":
    plt.legend(loc="lower right", prop={"size": 18, "weight": "bold"})
elif y_label == "CI":
    plt.legend(loc="lower right", prop={"size": 18, "weight": "bold"})
elif y_label == "P-value":
    plt.legend(loc="upper right", prop={"size": 18, "weight": "bold"})

plt.tight_layout()
plt.savefig(save_path)
