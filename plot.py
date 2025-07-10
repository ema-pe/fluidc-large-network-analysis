from pathlib import Path
import argparse
import functools

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from tqdm import tqdm

import ground_truth


def load_results_data(graph_name):
    communities = dict()

    results = list(Path("results/").glob(f"{graph_name}.fluidc.*.txt.gz"))
    name, seed, max_iter, time = [], [], [], []
    for result in tqdm(results, desc="Loading results data"):
        # I need to extract data directly form the path's name, as example:
        # "com-amazon.fluidc.513226.200.74.txt.gz"
        info = result.name.split(".")[2:5]

        name.append(result.name)
        seed.append(info[0])
        max_iter.append(info[1])
        time.append(info[2])

        communities[result.name] = ground_truth.get_unique_communities(result)

    data = pd.DataFrame(
        {
            "name": pd.Series(name, dtype="str"),
            "seed": pd.Series(seed, dtype="int32"),
            "max_iter": pd.Series(max_iter, dtype="int32"),
            "time": pd.Series(time, dtype="int32"),
        }
    )

    return data, communities


@functools.cache
def clusters_to_labels(clusters, all_vertices):
    clusters_list = list(clusters)

    label_map = {}
    for label, cluster in enumerate(clusters):
        for v in cluster:
            label_map[v] = label

    # Assign -1 for vertices not present in any community.
    return [label_map.get(v, -1) for v in all_vertices]


def calc_metric(fluidc_comm, ground_truth_comm, metric="nmi"):
    metrics = ("nmi", "ari", "purity")
    if metric not in metrics:
        raise ValueError(f"Found {metric!r}, expected one of {metrics}")

    data = []
    for result in tqdm(fluidc_comm, desc=f"Calculating {metric.upper()}"):
        comm = fluidc_comm[result]

        # Gather all vertices present in both clustering. I need tuple to be an
        # hashable object.
        all_vertices = tuple(sorted(set().union(*ground_truth_comm, *comm)))

        # NMI requires a list of labels: each index matches a vertex of the
        # graph, whereas each value matches the corresponding found community.
        ground_labels = clusters_to_labels(ground_truth_comm, all_vertices)
        result_labels = clusters_to_labels(comm, all_vertices)

        if metric == "nmi":
            score = normalized_mutual_info_score(ground_labels, result_labels)
        else:
            score = adjusted_rand_score(ground_labels, result_labels)

        data.append((result, score))

    return pd.DataFrame(data, columns=["name", metric])


def plot_metric(
    fluidc_metadata, fluidc_comm, ground_truth_comm, graph_name, metric="nmi"
):
    metrics = ("nmi", "ari", "purity")
    if metric not in metrics:
        raise ValueError(f"Found {metric!r}, expected one of {metrics}")

    # Calculate metric for all results.
    fluidc_nmi = calc_metric(fluidc_comm, ground_truth_comm, metric=metric)

    # Update the first DataFrame with metric score.
    fluidc_metadata = pd.merge(fluidc_metadata, fluidc_nmi, on="name")

    # Exclude the unused columns for metric plot.
    data = fluidc_metadata.drop(["name", "time"], axis=1)

    # Sort by seed and max_iter.
    data = data.sort_values(by=["seed", "max_iter"])

    # Average NMI for each max_iter across all seeds.
    stats = data.groupby("max_iter", as_index=False)[metric].agg(["mean", "std"])

    ax = stats.plot(x="max_iter", y="mean", marker="o", legend=False)

    # Show standard deviation around the mean.
    ax.fill_between(
        stats["max_iter"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=ax.lines[0].get_color(),
        alpha=0.2,
    )

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Max iterations")
    ax.set_ylabel(metric.upper())
    # Since we know in advance max_iter values, show the values on log scale and
    # fix the ticks to max_iter values.
    ax.set_xscale("log")
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    plot_name = f"plots/{graph_name}_{metric}.pdf"
    plt.savefig(plot_name)
    print(f"Saved {plot_name!r}")
    plt.close()


def time_plot(fluidc_metadata, graph_name):
    # Exclude the unused "name" column.
    data = fluidc_metadata.drop("name", axis=1)

    # Sort by seed and max_iter.
    data = data.sort_values(by=["seed", "max_iter"])

    # Average execution time for each max_iter across all seeds.
    stats = data.groupby("max_iter", as_index=False)["time"].agg(["mean", "std"])

    ax = stats.plot(x="max_iter", y="mean", marker="o", legend=False)
    # Show standard deviation around the mean.
    ax.fill_between(
        stats["max_iter"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=ax.lines[0].get_color(),
        alpha=0.2,
    )

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Max iterations")
    ax.set_ylabel("Seconds")
    # Since we know in advance max_iter values, show the values on log scale and
    # fix the ticks to max_iter values.
    ax.set_xscale("log")
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    plot_name = f"plots/{graph_name}_time-plot.pdf"
    plt.savefig(plot_name)
    print(f"Saved {plot_name!r}")
    plt.close()


def main(graph_name):
    ground = ground_truth.load(graph_name)
    fluidc_metadata, fluidc_comm = load_results_data(graph_name)

    time_plot(fluidc_metadata, graph_name)

    plot_metric(fluidc_metadata, fluidc_comm, ground, graph_name, metric="nmi")

    plot_metric(fluidc_metadata, fluidc_comm, ground, graph_name, metric="ari")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", help="Graph name", required=True)

    args = parser.parse_args()

    main(args.graph_name)
