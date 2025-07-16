"""This module generates plots that visualize the results of FluidC community
detection experiments. The plots are saved in the "plots" directory under the
results directory where the original data can be found. The module is designed
for use as a script.

For each complex network, the module calculates the normalized mutual
information (NMI), adjusted rand index (ARI), and cluster purity metrics for
each experiment against the ground truth communities. The module also produces
an NMI-based similarity matrix for all experiments and ground truth for a single
network. These metrics are cached to avoid recalculating them with each run.It
also generates execution time plots for each network and an aggregated plot of
all networks.

Example usage:
    python plot.py --results-dir results --graph-name com-amazon ...
"""

# pylint: disable=import-error,redefined-outer-name,too-many-locals

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


@functools.cache
def load_results_data(graph_name, results_dir):
    """Returns FluidC result data for a given graph, extracting metadata and
    community assignments.

    Args:
        graph_name (str): Name of the graph.
        results_dir (Path): Directory containing result files.

    Returns:
        data (pd.DataFrame): DataFrame with metadata (name, seed, max_iter,
            time) for each result file.
        communities (dict): Mapping from result file names to their unique
            community assignments.
    """
    communities = {}

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
def load_metrics(graph_name, results_dir, use_cache=True):
    """Returns the clustering metrics (NMI, ARI and purity) for the given graph.

    Args:
        graph_name (str): Name of the graph.
        results_dir (Path): Directory containing result files.
        use_cache (bool): Whether to use cached metrics if available.

    Returns:
        fluidc_metrics (pd.DataFrame): DataFrame containing metadata and
            clustering metrics.
    """
    fluidc_metadata, fluidc_comm = load_results_data(graph_name, results_dir)

    metrics_cache_file = Path(results_dir) / Path(f"{graph_name}_metrics.csv")

    fluidc_metrics = None
    if use_cache:
        try:
            fluidc_metrics = pd.read_csv(metrics_cache_file)
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"Failed to read {metrics_cache_file.as_posix()!r}")

    if fluidc_metrics is None:
        print(f"Calculating metrics for {graph_name!r}")
        ground = ground_truth.load(graph_name)
        fluidc_metrics = calc_metrics(fluidc_comm, ground)

    # We want the two dataframes, fluidc_metadata and fluidc_metrics, to be
    # merged in a single one. If fluid_metrics is loaded from disk, it is
    # already merged, otherwise if it is calculated we need to merge.
    if not set(fluidc_metadata.columns).issubset(fluidc_metrics.columns):
        fluidc_metrics = pd.merge(fluidc_metadata, fluidc_metrics, on="name")

        fluidc_metrics.to_csv(metrics_cache_file, index=False)
        print(f"Saved {metrics_cache_file.as_posix()!r}")

    return fluidc_metrics


@functools.cache
def clusters_to_labels(clusters, all_vertices):
    """Converts a list of clusters to a label vector for all vertices.

    Args:
        clusters (list): List of clusters (each cluster is a list of vertex IDs).
        all_vertices (list): List of all vertex IDs to assign labels.

    Returns:
        labels (list): List of integer labels for each vertex (or -1 if not in
            any cluster).
    """
    label_map = {}
    for label, cluster in enumerate(clusters):
        for v in cluster:
            label_map[v] = label

    # Assign -1 for vertices not present in any community.
    return [label_map.get(v, -1) for v in all_vertices]


def calc_metrics(fluidc_comm, ground_truth_comm):
    """Returns clustering metrics (NMI, ARI and purity) for each FluidC
    experiment against the ground truth communities.

    Args:
        fluidc_comm (dict): Mapping from result names to their community
            assignments.

        ground_truth_comm (list): Ground truth community assignments.

    Returns:
        df (pd.DataFrame): DataFrame with columns: name, nmi, ari, purity.
    """
    data = []
    for result in tqdm(fluidc_comm, desc="Calculating metrics"):
        comm = fluidc_comm[result]

        # Gather all vertices present in both clustering. I need tuple to be an
        # hashable object.
        all_vertices = tuple(sorted(set().union(*ground_truth_comm, *comm)))

        # NMI, ARI and Cluster Purity score require a list of labels: each index
        # matches a vertex of the graph, whereas each value matches the
        # corresponding found community.
        ground_labels = clusters_to_labels(ground_truth_comm, all_vertices)
        result_labels = clusters_to_labels(comm, all_vertices)

        # Remove nodes where either label is -1, to avoid distortion in purity
        # and NMI, since sklearn treats -1 as valid cluster label.
        filtered = [
            (t, p) for t, p in zip(ground_labels, result_labels) if t != -1 and p != -1
        ]
        if not filtered:
            raise ValueError("filtered is empty")
        ground_labels, result_labels = zip(*filtered)

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
        purity = np.sum(np.amax(matrix, axis=1)) / np.sum(matrix)

        data.append((result, nmi, ari, purity))

    df = pd.DataFrame(data, columns=["name", "nmi", "ari", "purity"])

    return df


def plot_similarity_matrix_nmi(
    graph_name, results_dir, output_dir=Path("results"), use_cache=True
):
    """Plots a similarity matrix (using NMI metric) for a specific graph,
    showing similarity between different FluidC seed runs and the ground truth
    for the highest max_iter value.

    Args:
        graph_name (str): Name of the graph.
        results_dir (Path): Directory containing result files.
        output_dir (Path): Directory to save the output plot.
        use_cache (bool): Whether to use cached metrics if available.

    Returns:
        None
    """
    # Load the metrics data and community assignments.
    fluidc_metrics = load_metrics(graph_name, results_dir, use_cache)
    _, fluidc_comm = load_results_data(graph_name, results_dir)

    # Load ground truth communities.
    ground_truth_comm = ground_truth.load(graph_name)

    # Find the highest max_iter value.
    highest_max_iter = fluidc_metrics["max_iter"].max()

    # Filter to only include results with the highest max_iter.
    filtered_metrics = fluidc_metrics[fluidc_metrics["max_iter"] == highest_max_iter]

    # Create a list of all community assignments to compare (including ground
    # truth).
    all_comms = {}
    for _, row in filtered_metrics.iterrows():
        result_name = row["name"]
        all_comms[f"Seed {row['seed']}"] = fluidc_comm[result_name]

    # Add ground truth to the comparison.
    all_comms["Ground Truth"] = ground_truth_comm

    # Create a similarity matrix.
    labels = list(all_comms.keys())
    matrix_size = len(labels)
    similarity_matrix = np.zeros((matrix_size, matrix_size))

    # Calculate similarity metrics for each pair
    for i, label_i in enumerate(tqdm(labels, desc="Calculating NMI similarity matrix")):
        comm_i = all_comms[label_i]

        for j, label_j in enumerate(labels):
            if i == j:
                # Same communities, perfect similarity
                similarity_matrix[i, j] = 1.0
                continue

            comm_j = all_comms[label_j]

            # See calc_metrics() for docs.
            all_vertices = tuple(sorted(set().union(*comm_i, *comm_j)))

            labels_i = clusters_to_labels(comm_i, all_vertices)
            labels_j = clusters_to_labels(comm_j, all_vertices)

            filtered = [
                (t, p) for t, p in zip(labels_i, labels_j) if t != -1 and p != -1
            ]
            if not filtered:
                raise ValueError("filtered is empty")
            labels_i, labels_j = zip(*filtered)

            similarity_matrix[i, j] = normalized_mutual_info_score(labels_i, labels_j)

    # Create the heatmap.
    fig, ax = plt.subplots(figsize=(10, 8))
    # Show the matrix as heatmap, with a specific color map and min/max values.
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)

    # Add colorbar.
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("NMI", rotation=-90, va="bottom", fontsize=14)

    # Show ticks and labels.
    ax.set_xticks(np.arange(matrix_size))
    ax.set_yticks(np.arange(matrix_size))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Rotate the tick labels and set their alignment, because they too long (eg.
    # "Seed XXXX" or "Ground Truth").
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(matrix_size):
        for j in range(matrix_size):
            text_color = "white" if similarity_matrix[i, j] < 0.7 else "black"
            ax.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    fig.tight_layout()

    # Save the plot.
    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path(
        f"{graph_name}_similarity_matrix_nmi_{highest_max_iter}.pdf"
    )
    plt.savefig(plot_name)
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def plot_metric(fluidc_metrics, graph_name, metric="nmi", output_dir=Path("results")):
    """Plots the specified clustering metric (NMI, ARI, purity) as a function of
    max_iter for a graph.

    Args:
        fluidc_metrics (pd.DataFrame): DataFrame with clustering metrics.
        graph_name (str): Name of the graph.
        metric (str): Metric to plot ("nmi", "ari", or "purity").
        output_dir (Path): Directory to save the output plot.

    Returns:
        None
    """
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

    _, ax = plt.subplots(figsize=(8, 6))
    stats.plot(x="max_iter", y="mean", marker="o", legend=False, ax=ax)

    # Show standard deviation around the mean.
    ax.fill_between(
        stats["max_iter"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=ax.lines[0].get_color(),
        alpha=0.2,
    )

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Max iterations", fontsize=14)
    ax.set_ylabel(metric.upper(), fontsize=14)
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    # Set the Y axis to scientific notation if the values are too small.
    yaxis_formatter = plt.ScalarFormatter(useMathText=True)
    yaxis_formatter.set_scientific(True)
    yaxis_formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(yaxis_formatter)

    # Increase tick label size.
    ax.tick_params(axis="both", which="major", labelsize=12)

    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path(f"{graph_name}_{metric}.pdf")
    plt.savefig(plot_name, bbox_inches="tight")  # Do not cut text in figure.
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def time_plot(fluidc_metrics, graph_name, output_dir=Path("results")):
    """Plots the average execution time of FluidC experiment on a single graph
    as a function of max_iter.

    Args:
        fluidc_metrics (pd.DataFrame): DataFrame with execution times.
        graph_name (str): Name of the graph.
        output_dir (Path): Directory to save the output plot.

    Returns:
        None
    """
    # Exclude unused columns.
    data = fluidc_metrics[["name", "seed", "max_iter", "time"]]

    # Sort by seed and max_iter.
    data = data.sort_values(by=["seed", "max_iter"])

    # Average execution time for each max_iter across all seeds.
    stats = data.groupby("max_iter", as_index=False)["time"].agg(["mean", "std"])

    _, ax = plt.subplots(figsize=(8, 6))
    stats.plot(x="max_iter", y="mean", marker="o", legend=False, ax=ax)
    # Show standard deviation around the mean.
    ax.fill_between(
        stats["max_iter"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=ax.lines[0].get_color(),
        alpha=0.2,
    )

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Max iterations", fontsize=14)
    ax.set_ylabel("Seconds", fontsize=14)
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    # Increase tick label size.
    ax.tick_params(axis="both", which="major", labelsize=12)

    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path(f"{graph_name}_time.pdf")
    plt.savefig(plot_name, bbox_inches="tight")  # Do not cut text on figure.
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def time_aggregated_plot(graphs_data, output_dir=Path("results")):
    """Saves the aggregated plot for execution time of FluidC on various graphs.
    It is the aggregated version of time_plot().

    Args:
        graphs_data (dict): Dictionary mapping graph names to their metrics
            DataFrames.
        output_dir (Path): Directory to save the output plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # pylint: disable=unused-variable
    for name, metrics in graphs_data.items():
        # See time_plot for comments on the loop instructions.
        data = metrics[["name", "seed", "max_iter", "time"]]
        data = data.sort_values(by=["seed", "max_iter"])
        stats = data.groupby("max_iter", as_index=False)["time"].agg(["mean", "std"])

        # Get the next color, so we can use also for "fill_between".
        color = ax._get_lines.get_next_color()  # pylint: disable=protected-access

        stats.plot(x="max_iter", y="mean", ax=ax, marker="o", color=color, label=name)
        ax.fill_between(
            stats["max_iter"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.2,
        )

    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Max iterations", fontsize=14)
    ax.set_ylabel("Seconds", fontsize=14)
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    # Increase tick label size.
    ax.tick_params(axis="both", which="major", labelsize=12)

    assert isinstance(output_dir, Path)
    output_dir = output_dir / Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = output_dir / Path("aggregated_time.pdf")
    plt.savefig(plot_name, bbox_inches="tight")  # Do not cut text on figure.
    print(f"Saved {plot_name.as_posix()!r}")
    plt.close()


def plot_single_graph(graph_name, use_cache, results_dir):
    """Generates and saves all relevant plots (execution time, NMI, ARI, purity,
    similarity matrix) for a single graph.

    Args:
        graph_name (str): Name of the graph.
        use_cache (bool): Whether to use cached metrics if available.
        results_dir (Path): Directory containing result files used to generate
            the plots (and to save the plots).

    Returns:
        None
    """
    fluidc_metrics = load_metrics(graph_name, results_dir, use_cache)

    time_plot(fluidc_metrics, graph_name, output_dir=results_dir)

    for metric in ["nmi", "ari", "purity"]:
        plot_metric(fluidc_metrics, graph_name, metric=metric, output_dir=results_dir)

    plot_similarity_matrix_nmi(
        graph_name, results_dir, output_dir=results_dir, use_cache=use_cache
    )


def plot_aggregate_graphs(graph_names, use_cache, results_dir):
    """Generates and saves the aggregated execution time plot for multiple
    graphs.

    Args:
        graph_names (list): List of graph names.
        use_cache (bool): Whether to use cached metrics if available.
        results_dir (Path): Directory containing result files, used to generate
            the plots (and save the plots).

    Returns:
        None
    """
    data = {}
    for graph_name in graph_names:
        data[graph_name] = load_metrics(graph_name, results_dir, use_cache)

    time_aggregated_plot(data, results_dir)


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
        plot_single_graph(graph_name, not args.no_cache, args.results_dir)

    print("Plotting aggregate plots...")
    plot_aggregate_graphs(args.graph_name, not args.no_cache, args.results_dir)
