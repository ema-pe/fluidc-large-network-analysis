from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score

import ground_truth


def load_results_data(graph_name, only_metadata=False):
    name, seed, max_iter, time = [], [], [], []
    for result in Path("results/").glob(f"{graph_name}.fluidc.*.txt.gz"):
        # I need to extract data directly form the path's name, as example:
        # "com-amazon.fluidc.513226.200.74.txt.gz"
        info = result.name.split(".")[2:5]

        name.append(result.name)
        seed.append(info[0])
        max_iter.append(info[1])
        time.append(info[2])

    data = pd.DataFrame(
        {
            "name": pd.Series(name, dtype="str"),
            "seed": pd.Series(seed, dtype="int32"),
            "max_iter": pd.Series(max_iter, dtype="int32"),
            "time": pd.Series(time, dtype="int32"),
        }
    )

    if only_metadata:
        return data

    communities = dict()
    for result in Path("results/").glob(f"{graph_name}.fluidc.*.txt.gz"):
        communities[result.name] = ground_truth.get_unique_communities(result)

    return data, communities


def calc_nmi(graph_name):
    """Returns a pd.DataFrame with Normalized Mutual Information (NMI) score for
    found communities using FluidC and the grund truth communities on the given
    graph.

    The NMI metric quantifies the similarity between the two clusterings.
    """

    def clusters_to_labels(clusters, all_vertices):
        """Returns the converted list of communities as a list of labels for
        each vertex.

        Args:
            clusters (list of lists): Each inner list contains the vertices in a cluster.

            all_vertices (list): List of all vertices to assign labels to.

        Returns:
            list: A list of integer labels corresponding to the cluster
                  assignment of each vertex in all_vertices. Vertices not present in
                  any cluster are assigned a label of -1.
        """
        label_map = {}
        for label, cluster in enumerate(clusters):
            for v in cluster:
                label_map[v] = label
        # Assign -1 for vertices not present in any community.
        return [label_map.get(v, -1) for v in all_vertices]

    def calc_nmi_row(row):
        """Returns the NMI score for a single experiment row."""
        # Get the communities for the given result experiment (by its name).
        result_comm = communities[row["name"]]

        # Gather all vertices present in both clustering.
        all_vertices = sorted(set().union(*ground, *result_comm))

        # NMI requires a list of labels: each index matches a vertex of the
        # graph, whereas each value matches the corresponding found community.
        ground_labels = clusters_to_labels(list(ground), all_vertices)
        result_labels = clusters_to_labels(list(result_comm), all_vertices)

        return normalized_mutual_info_score(ground_labels, result_labels)

    ground = ground_truth.load(graph_name)
    data, communities = load_results_data(graph_name)

    # Add the NMI column to the original DataFrame.
    data["nmi"] = data.apply(calc_nmi_row, axis=1)

    return data


def nmi_plot(graph_name):
    data = calc_nmi(graph_name)

    import pdb

    pdb.set_trace()

    # Exclude the unused "name" column.
    data = data.drop("name", axis=1)

    # Sort by seed and max_iter.
    data = data.sort_values(by=["seed", "max_iter"])

    # Average NMI for each max_iter across all seeds.
    stats = data.groupby("max_iter", as_index=False)["nmi"].agg(["mean", "std"])

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
    ax.set_ylabel("NMI")
    ax.set_ylim(0, 1)
    # Since we know in advance max_iter values, show the values on log scale and
    # fix the ticks to max_iter values.
    ax.set_xscale("log")
    ax.set_xticks(stats["max_iter"])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    plt.savefig(f"plots/{graph_name}_nmi.pdf")
    plt.close()


def time_plot(graph_name):
    data = load_results_data(graph_name, only_metadata=True)

    # Exclude the unused "name" column.
    data = data.drop("name", axis=1)

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

    plt.savefig(f"plots/{graph_name}_time-plot.pdf")
    plt.close()


def main(graph_name):
    time_plot(graph_name)

    nmi_plot(graph_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", help="Graph name", required=True)

    args = parser.parse_args()

    main(args.graph_name)
