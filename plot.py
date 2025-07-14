from pathlib import Path
import argparse
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    contingency_matrix,
)
from tqdm import tqdm

import ground_truth


def load_results_data(graph_name, results_dir):
    communities = dict()

    results = list(Path(results_dir).glob(f"{graph_name}.fluidc.*.txt.gz"))
    name, seed, max_iter, time = [], [], [], []
    for result in tqdm(results, desc="Loading results data"):
        # I need to extract data directly from the path's name, as example:
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


def calc_metrics(fluidc_comm, ground_truth_comm):
    data = []
    for result in tqdm(fluidc_comm, desc=f"Calculating metrics"):
        comm = fluidc_comm[result]

        # Gather all vertices present in both clustering. I need tuple to be an
        # hashable object.
        all_vertices = tuple(sorted(set().union(*ground_truth_comm, *comm)))

        # NMI, ARI and Cluster Purity score require a list of labels: each index
        # matches a vertex of the graph, whereas each value matches the
        # corresponding found community.
        ground_labels = clusters_to_labels(ground_truth_comm, all_vertices)
        result_labels = clusters_to_labels(comm, all_vertices)

        nmi = normalized_mutual_info_score(ground_labels, result_labels)
        ari = adjusted_rand_score(ground_labels, result_labels)

        # The purity score is computed by:
        #
        #   1. For each predicted cluster, finding the most common true label
        #   (the "majority class") among the points assigned to that cluster.
        #   2. Counting how many points in each cluster belong to their
        #   cluster's majority class.
        #   3. Summing these counts for all clusters and dividing by the total
        #   number of data points.
        #
        # I can build the contingency matrix for the first two points.
        matrix = contingency_matrix(ground_labels, result_labels)
        # Use np.amax to get the largest value in each column (max for each
        # predicted cluster).
        purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

        data.append((result, nmi, ari, purity))

    df = pd.DataFrame(data, columns=["name", "nmi", "ari", "purity"])

    return df


def plot_metric(fluidc_metrics, graph_name, metric="nmi", output_dir=Path("results")):
    metrics = ("nmi", "ari", "purity")
    if metric not in metrics:
        raise ValueError(f"Found {metric!r}, expected one of {metrics}")

    # Exclude the unused columns for metric plot.
    excluded_metrics = [x for x in metrics if x != metric]
    data = fluidc_metrics.drop(["name", "time", *excluded_metrics], axis=1)

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
    # ax.set_xscale("log")
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path(f"{graph_name}_{metric}.pdf")
    plt.savefig(plot_name)
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def time_plot(fluidc_metadata, graph_name, output_dir=Path("results")):
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

    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path(f"{graph_name}_time.pdf")
    plt.savefig(plot_name)
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def main(graph_name, use_cache, results_dir):
    ground = ground_truth.load(graph_name)
    fluidc_metadata, fluidc_comm = load_results_data(graph_name, results_dir)

    time_plot(fluidc_metadata, graph_name, output_dir=results_dir)

    # Calculate metrics one and save/load from cache.
    metrics_cache_file = Path(results_dir) / Path(f"{graph_name}_metrics.csv")
    fluidc_metrics = None
    if use_cache:
        try:
            fluidc_metrics = pd.read_csv(metrics_cache_file)
        except Exception:
            print(
                f"Failed to read {metrics_cache_file.as_posix()!r}, recalculating metrics for {graph_name!r}"
            )

    if fluidc_metrics is None:
        fluidc_metrics = calc_metrics(fluidc_comm, ground)

        # Update the first DataFrame with metric score.
        fluidc_metrics = pd.merge(fluidc_metadata, fluidc_metrics, on="name")

    fluidc_metrics.to_csv(metrics_cache_file, index=False)
    print(f"Saved {metrics_cache_file.as_posix()!r}")

    for metric in ["nmi", "ari", "purity"]:
        plot_metric(fluidc_metrics, graph_name, metric=metric, output_dir=results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        help="Directory where are the result files",
        type=Path,
        required=True,
    )
    parser.add_argument("--graph-name", help="Graph name", required=True, nargs="+")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached metrics calculation and recalculate metrics",
    )

    args = parser.parse_args()

    for graph_name in args.graph_name:
        print(f"Plotting {graph_name!r}...")
        main(graph_name, not args.no_cache, args.results_dir)
