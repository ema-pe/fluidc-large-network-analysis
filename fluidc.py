from pathlib import Path
import time
import gzip
import argparse

import numpy as np
import networkx as nx


def run(graph_path, seed, ground_truth, max_iter, results_dir=Path("results")):
    """Runs FluidC community detection algorithm on the provided graph and saves
    the detected communities to a file on disk.

    Args:
        graph_path (Path): Path to the graph edge list file.
        seed (int): RNG seed (must be non-negative).
        ground_truth (int): FluidC k argument (number of communities, must be positive).
        max_iter (int): FluidC max iterations (must be positive).
        results_dir (Path): Directory to save the result. Defaults to "results".
    """
    # Enforce Path objects.
    graph_path = Path(graph_path)
    results_dir = Path(results_dir)

    print(f"Loading graph {graph_path.as_posix()!r}...")

    graph = nx.read_edgelist(graph_path)

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Running FluidC...")

    assert seed >= 0, "Seed must be non-negative"
    assert (
        ground_truth > 0
    ), "FluidC k argument must be a positive number (at least one)"
    assert max_iter > 0, "Max iterations must be a positive number (at least one)"

    rng = np.random.default_rng(seed)

    start = time.time()
    communities = nx.community.asyn_fluidc(
        graph, ground_truth, max_iter=max_iter, seed=rng
    )
    end = time.time()

    seconds = round(end - start)
    print(f"Elapsed: {seconds} seconds")
    print("Saving found communities...")

    # Create directory and output file name.
    results_dir.mkdir(exist_ok=True)
    graph_name = graph_path.name.split(".")[
        0
    ]  # Keep only the graph name (eg. com-youtube).
    communities_path = results_dir / Path(
        f"{graph_name}.fluidc.{seed}.{max_iter}.{seconds}.txt.gz"
    )

    # Write the communities to disk as a plain text file: each line is a
    # community with the node IDs.
    with gzip.open(communities_path, "wt") as file:
        for community in list(communities):
            file.write(" ".join(map(str, community)) + "\n")

    print(f"Saved to {communities_path.as_posix()!r}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="RNG seed", required=True)
    parser.add_argument("--graph", type=Path, help="Graph path", required=True)
    parser.add_argument(
        "--ground-truth",
        type=int,
        help="FluidC k argument (ground truth communities)",
        required=True,
    )
    parser.add_argument(
        "--max-iter", type=int, help="FluidC max iterations", required=True
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Directory to save result file",
        required=True,
    )
    args = parser.parse_args()

    run(
        graph_path=args.graph,
        seed=args.seed,
        ground_truth=args.ground_truth,
        max_iter=args.max_iter,
        results_dir=args.result_dir,
    )


if __name__ == "__main__":
    _main()
