import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
EUBO: Evidence Upper Bound Objective
ELBO: Evidence Lower Bound Objective
"""


def plot_likelihood_diffs(
    folder="plot_data",
    exper="moon",
    appr="gauss_fullcov",
    plot_retrain_vs_full=True,
    plot_legend=True,
    plot_eubo=True,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plot_data = pickle.load(
        open("{}/likelihood_diff_{}_{}.p".format(folder, exper, appr), "rb")
    )
    full_elbo_mode_percentages = plot_data["elbo_mode_percentage"]
    full_eubo_mode_percentages = plot_data["eubo_mode_percentage"]
    removed_likelihood_diffs = plot_data["removed_likelihood_diff"]
    remain_likelihood_diffs = plot_data["remain_likelihood_diff"]
    std_removed_likelihood_diffs = plot_data["std_removed_likelihood_diff"]
    std_remain_likelihood_diffs = plot_data["std_remain_likelihood_diff"]

    if exper.startswith("moon"):
        selected_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
    elif exper == "banknote_authentication1":
        if appr == "gauss_fullcov":
            selected_percentages = [0.5, 0.1, 1e-3, 1e-5, 1e-9, 0.0]
        elif appr == "maf":
            selected_percentages = [1e-3, 1e-5, 1e-7, 1e-9, 0.0]
    else:
        raise Exception("Unknown experiment {}".format(exper))

    remove_idxs = []
    for i, percentage in enumerate(full_elbo_mode_percentages):
        if percentage not in selected_percentages:
            remove_idxs.append(i)

    removed_likelihood_diffs["elbo"]["vs_retrain"] = np.delete(
        removed_likelihood_diffs["elbo"]["vs_retrain"], remove_idxs
    )
    removed_likelihood_diffs["elbo"]["vs_full"] = np.delete(
        removed_likelihood_diffs["elbo"]["vs_full"], remove_idxs
    )
    remain_likelihood_diffs["elbo"]["vs_retrain"] = np.delete(
        remain_likelihood_diffs["elbo"]["vs_retrain"], remove_idxs
    )
    remain_likelihood_diffs["elbo"]["vs_full"] = np.delete(
        remain_likelihood_diffs["elbo"]["vs_full"], remove_idxs
    )

    std_removed_likelihood_diffs["elbo"]["vs_retrain"] = np.delete(
        std_removed_likelihood_diffs["elbo"]["vs_retrain"], remove_idxs
    )
    std_removed_likelihood_diffs["elbo"]["vs_full"] = np.delete(
        std_removed_likelihood_diffs["elbo"]["vs_full"], remove_idxs
    )
    std_remain_likelihood_diffs["elbo"]["vs_retrain"] = np.delete(
        std_remain_likelihood_diffs["elbo"]["vs_retrain"], remove_idxs
    )
    std_remain_likelihood_diffs["elbo"]["vs_full"] = np.delete(
        std_remain_likelihood_diffs["elbo"]["vs_full"], remove_idxs
    )

    if plot_eubo:
        remove_idxs = []
        for i, percentage in enumerate(full_eubo_mode_percentages):
            if percentage not in selected_percentages:
                remove_idxs.append(i)

        removed_likelihood_diffs["eubo"]["vs_retrain"] = np.delete(
            removed_likelihood_diffs["eubo"]["vs_retrain"], remove_idxs
        )
        removed_likelihood_diffs["eubo"]["vs_full"] = np.delete(
            removed_likelihood_diffs["eubo"]["vs_full"], remove_idxs
        )
        remain_likelihood_diffs["eubo"]["vs_retrain"] = np.delete(
            remain_likelihood_diffs["eubo"]["vs_retrain"], remove_idxs
        )
        remain_likelihood_diffs["eubo"]["vs_full"] = np.delete(
            remain_likelihood_diffs["eubo"]["vs_full"], remove_idxs
        )

        std_removed_likelihood_diffs["eubo"]["vs_retrain"] = np.delete(
            std_removed_likelihood_diffs["eubo"]["vs_retrain"], remove_idxs
        )
        std_removed_likelihood_diffs["eubo"]["vs_full"] = np.delete(
            std_removed_likelihood_diffs["eubo"]["vs_full"], remove_idxs
        )
        std_remain_likelihood_diffs["eubo"]["vs_retrain"] = np.delete(
            std_remain_likelihood_diffs["eubo"]["vs_retrain"], remove_idxs
        )
        std_remain_likelihood_diffs["eubo"]["vs_full"] = np.delete(
            std_remain_likelihood_diffs["eubo"]["vs_full"], remove_idxs
        )

    n = len(selected_percentages)

    if plot_legend:
        figsize = (2.8, 1.8)
    else:
        figsize = (1.8, 1.8)

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    ax.plot(
        list(range(n)),
        removed_likelihood_diffs["elbo"]["vs_retrain"],
        linestyle="-",
        color=colors[0],
        marker="v",
        label="rKL",
    )
    ax.fill_between(
        list(range(n)),
        removed_likelihood_diffs["elbo"]["vs_retrain"],
        removed_likelihood_diffs["elbo"]["vs_retrain"]
        + std_removed_likelihood_diffs["elbo"]["vs_retrain"],
        color=colors[0],
        alpha=0.3,
    )

    if plot_eubo:
        ax.plot(
            list(range(n)),
            removed_likelihood_diffs["eubo"]["vs_retrain"],
            linestyle="--",
            color=colors[1],
            marker="^",
            label="EUBO",
        )
        ax.fill_between(
            list(range(n)),
            removed_likelihood_diffs["eubo"]["vs_retrain"],
            removed_likelihood_diffs["eubo"]["vs_retrain"]
            + std_removed_likelihood_diffs["eubo"]["vs_retrain"],
            color=colors[1],
            alpha=0.3,
        )

    if plot_retrain_vs_full:
        ax.plot(
            list(range(n)),
            np.ones(n) * removed_likelihood_diffs["full_vs_retrain"],
            linestyle=(0, (1, 1)),
            color=colors[3],
            label="full",
        )
        ax.fill_between(
            list(range(n)),
            np.ones(n) * removed_likelihood_diffs["full_vs_retrain"],
            removed_likelihood_diffs["full_vs_retrain"]
            + np.ones(n) * std_removed_likelihood_diffs["full_vs_retrain"],
            color=colors[3],
            alpha=0.3,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels=[str(i) for i in selected_percentages], rotation=90)
    ax.set_xlabel(r"$\lambda$")

    if plot_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    ax.plot(
        list(range(n)),
        remain_likelihood_diffs["elbo"]["vs_retrain"],
        linestyle="-",
        color=colors[0],
        marker="v",
        label="rKL",
    )
    ax.fill_between(
        list(range(n)),
        remain_likelihood_diffs["elbo"]["vs_retrain"],
        std_remain_likelihood_diffs["elbo"]["vs_retrain"],
        color=colors[0],
        alpha=0.3,
    )

    if plot_eubo:
        ax.plot(
            list(range(n)),
            remain_likelihood_diffs["eubo"]["vs_retrain"],
            linestyle="--",
            color=colors[1],
            marker="^",
            label="EUBO",
        )
        ax.fill_between(
            list(range(n)),
            remain_likelihood_diffs["eubo"]["vs_retrain"],
            remain_likelihood_diffs["eubo"]["vs_retrain"]
            + std_remain_likelihood_diffs["eubo"]["vs_retrain"],
            color=colors[1],
            alpha=0.3,
        )

    if plot_retrain_vs_full:
        ax.plot(
            list(range(n)),
            np.ones(n) * remain_likelihood_diffs["full_vs_retrain"],
            linestyle=(0, (1, 1)),
            color=colors[3],
            label="full",
        )
        ax.fill_between(
            list(range(n)),
            np.ones(n) * remain_likelihood_diffs["full_vs_retrain"],
            remain_likelihood_diffs["full_vs_retrain"]
            + np.ones(n) * std_remain_likelihood_diffs["full_vs_retrain"],
            color=colors[3],
            alpha=0.3,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels=[str(i) for i in selected_percentages], rotation=90)
    ax.set_xlabel(r"$\lambda$")

    if plot_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.show()


# Example usage:
# plot_likelihood_diffs(folder="plot_data", exper="moon", appr="gauss_fullcov", plot_retrain_vs_full=True, plot_legend=True, plot_eubo=True)
