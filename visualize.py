import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from constants import *
from repository import Repository

repo = Repository()


def make_legend_elements(elements):
    line2d_dict = {
        "user of ground bs": ["o", None, "w", "k", None, 7],
        "user of aerial bs": ["X", None, "w", "k", None, 7],
        "ground bs": ["s", 0, "k", "w", "k", 9],
        "aerial bs": ["o", None, "w", "w", "k", 10],
    }

    legend_elements = []
    for element in elements:
        line2d = Line2D(
            [0],
            [0],
            marker=line2d_dict[element][0],
            linewidth=line2d_dict[element][1],
            color=line2d_dict[element][2],
            label=element,
            markerfacecolor=line2d_dict[element][3],
            markeredgecolor=line2d_dict[element][4],
            markersize=line2d_dict[element][5],
        )
        legend_elements.append(line2d)
    return legend_elements


def plot_network(
    bss,
    association_array,
    title,
    result_path=None,
    aerial_bss_indexes=None,
    aerial_user_indexes=None,
    bss_weights=None,
    user_numbers=None,
):
    users = repo.users
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FG_SIZE)
    ax.set(xlim=[0, HEIGHT], ylim=[0, WIDTH], title=title)
    # draw users
    for i in range(len(association_array)):
        marker = "o"
        if (aerial_user_indexes is not None) and (i in aerial_user_indexes):
            marker = "X"
        ax.scatter(
            x=users[i, 0],
            y=users[i, 1],
            color=np.array(COLORS[association_array[i]]) / 255.0,
            marker=marker,
        )
    # draw bss
    for j in range(len(bss)):
        c = 1 if j < 10 else 2
        ax.scatter(
            x=bss[j, 0],
            y=bss[j, 1],
            color=np.array(COLORS[j]) / 255.0,
            marker=NUMBER_MARKER[j],
            s=BS_SIZE * c,
        )
        if (aerial_bss_indexes is not None) and (j in aerial_bss_indexes):
            ax.scatter(
                x=bss[j, 0],
                y=bss[j, 1],
                s=450,
                linewidths=1.7,
                facecolors="none",
                edgecolors=np.array(COLORS[j]) / 255.0,
            )
        else:
            rect = patches.Rectangle(
                (bss[j, 0] - 65, bss[j, 1] - 65),
                140,
                140,
                linewidth=2,
                edgecolor=np.array(COLORS[j]) / 255.0,
                facecolor="none",
            )
            ax.add_patch(rect)

    # power weights
    if bss_weights is not None:
        text = "PI , User Qty:\n"
        for k in range(len(bss_weights)):
            pi2digit = "{0:.2f}".format(bss_weights[k])
            text = (
                text
                + str(k + 1)
                + ") "
                + pi2digit
                + " , "
                + str(user_numbers[k])
                + "\n"
            )
        ax.text(
            1.015,
            0.99,
            text,
            transform=ax.transAxes,
            bbox=dict(facecolor="green", alpha=0.1),
            horizontalalignment="left",
            verticalalignment="top",
        )
    if aerial_user_indexes is not None:
        legend_elements = make_legend_elements(
            ["user of ground bs", "user of aerial bs", "ground bs", "aerial bs"]
        )
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, -0.03), ncol=4)
    else:
        legend_elements = make_legend_elements(["user of ground bs", "ground bs"])
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.7, -0.03), ncol=4)
    if result_path:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")
        return
    fig.show()


def plot_upas_changes(
    bss_beginning,
    bss_final,
    changed_positions,
    title,
    result_path=None,
    aerial_bss_indexes=None,
):
    users = repo.users
    user_color = np.array([100, 150, 255]) / 255.0
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FG_SIZE)
    ax.set(xlim=[0, HEIGHT], ylim=[0, WIDTH], title=title)
    # draw users
    for i in range(len(users)):
        ax.scatter(
            x=users[i, 0],
            y=users[i, 1],
            color=user_color,
            marker="o",
        )

    # draw beginning bss
    for j in range(len(bss_beginning)):
        if (aerial_bss_indexes is not None) and (j in aerial_bss_indexes):
            ax.scatter(
                x=bss_beginning[j, 0],
                y=bss_beginning[j, 1],
                color=np.array([255, 255, 255, 0]) / 255.0,
                edgecolors=np.array([0, 50, 0]) / 255.0,
                marker="o",
                s=50,
            )

    # draw final bss
    for j in range(len(bss_final)):
        if (aerial_bss_indexes is not None) and (j in aerial_bss_indexes):
            ax.scatter(
                x=bss_final[j, 0],
                y=bss_final[j, 1],
                color=np.array([0, 150, 50]) / 255.0,
                marker="o",
                s=50,
            )
        else:
            ax.scatter(
                x=bss_final[j, 0],
                y=bss_final[j, 1],
                color=np.array([0, 150, 50]) / 255.0,
                marker="s",
                s=50,
            )
    positions_history = [[position] for position in bss_beginning.tolist()]
    for k in range(len(changed_positions)):
        positions = changed_positions[k]["positions"]
        indexes = changed_positions[k]["indexes"]
        for p, i in zip(positions, indexes):
            positions_history[i].append(p)

    for i in range(len(bss_final)):
        positions_history[i].append(bss_final[i].tolist())

    for i in range(len(positions_history)):
        if (aerial_bss_indexes is not None) and (i in aerial_bss_indexes):
            for j in range(1, len(positions_history[i])):
                pre_pos = positions_history[i][j]
                cur_pos = positions_history[i][j - 1]
                if pre_pos != cur_pos:
                    ax.annotate(
                        "",
                        xy=(cur_pos[0], cur_pos[1]),
                        xytext=(pre_pos[0], pre_pos[1]),
                        arrowprops=dict(arrowstyle="->"),
                    )

    if result_path:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")
        return
    fig.show()


def plot_topology(
    users,
    title,
    color,
    x_window,
    y_window,
    result_path=None,
):
    plt.style.use("default")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FG_SIZE)
    ax.set(xlim=x_window, ylim=y_window, title=title)
    # draw users
    for i in range(len(users)):
        ax.scatter(
            x=users[i, 0],
            y=users[i, 1],
            color=np.array(color) / 255.0,
            marker="o",
        )
    if result_path:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")
        return
    fig.show()


def plot_convergence(x_axis, y_axis, alg_list, result_path=None):
    x_label, y_label = "iteration", "obf"
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    color_dict = {"PROB": "red", "UAC": "black", "UPAS": "blue"}
    for i in range(1, len(x_axis)):
        ax.plot(
            [x_axis[i - 1], x_axis[i]],
            [y_axis[i - 1], y_axis[i]],
            linewidth=2,
            color=color_dict[alg_list[i]],
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    legend_elements = [
        Line2D([0], [0], color=color_dict["PROB"], label="PROB"),
        Line2D([0], [0], color=color_dict["UAC"], label="UAC"),
        Line2D([0], [0], color=color_dict["UPAS"], label="UPAS"),
    ]
    legend = ax.legend(handles=legend_elements, loc=4)
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.set_facecolor((1.0, 1.0, 1.0))
    if result_path is None:
        fig.show()
    else:
        fig.savefig(
            result_path, bbox_inches="tight", facecolor="white", edgecolor="white"
        )
        plt.close("all")


def plot_results(x_axis, y_axis, x_label, y_label, result_path=None):
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x_axis, y_axis)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    for i_x, i_y in zip(x_axis, y_axis):
        ax.text(i_x, i_y, "%.2f" % (i_y), size="x-small")
    if result_path is None:
        fig.show()
    else:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")


def plot_bss_weights(weights, result_path):
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=1)
    x_axis = [i + 1 for i in range(len(weights))]
    mean_weight = sum(weights) / len(weights)
    plot = plt.bar(x_axis, weights)
    for value in plot:
        height = value.get_height()
        ax.text(
            value.get_x() + value.get_width() / 2.0,
            1.002 * height,
            "%0.2f" % height,
            ha="center",
            va="bottom",
        )
    ax.axhline(y=mean_weight, color="r", linewidth=2, linestyle="-")
    ax.text(
        len(weights) + 1.2,
        mean_weight,
        "Avg = %.2f" % (mean_weight),
        ha="left",
        va="center",
        color="r",
    )
    ax.set_xlabel("Base Station")
    ax.set_ylabel("PI")
    if result_path is None:
        fig.show()
    else:
        fig.savefig(result_path, bbox_inches="tight")


def multi_plot(x, x_label, y_list, marker_list, color_list, legends, result_path=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("obf")
    for i in range(len(y_list)):
        if marker_list is not None:
            ax.plot(
                x,
                y_list[i],
                marker=marker_list[i],
                color=color_list[i],
                label=legends[i],
            )
        else:
            x = list(range(len(y_list[i])))
            ax.plot(x, y_list[i], color=color_list[i], label=legends[i])
    ax.legend()
    if result_path is None:
        fig.show()
    else:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")


def plot_pi_values(
    legends, dataset_titles, color_list, data_list, ylim, result_path=None
):
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=1)
    barWidth = 0.10
    br_list = []
    for i in range(len(data_list)):
        # Set position of bar on X axis
        if i == 0:
            br = np.arange(len(data_list[0]))
        else:
            br = [x + barWidth for x in br_list[i - 1]]
        br_list.append(br)
        # Make the plot
        ax.bar(
            br_list[i],
            data_list[i],
            color=color_list[i],
            width=barWidth,
            edgecolor="grey",
            label=legends[i],
        )

    # Adding Xticks
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=15)
    ax.set_ylabel("Objective Function Value ", fontweight="bold", fontsize=15)
    ax.set_xticks([r + barWidth for r in range(len(data_list[0]))], dataset_titles)
    ax.axis(ymin=ylim[0], ymax=ylim[1])
    ax.legend()
    if result_path is None:
        fig.show()
    else:
        fig.savefig(result_path, bbox_inches="tight")
        plt.close("all")
