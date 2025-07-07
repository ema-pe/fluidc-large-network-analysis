from pathlib import Path
import time
import gzip
import argparse

import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="RNG seed", required=True)
parser.add_argument("--graph", type=Path, help="Graph path", required=True)
parser.add_argument(
    "--ground-truth",
    type=int,
    help="FluidC k argument (ground truth communities)",
    required=True,
)
parser.add_argument("--max-iter", type=int, help="FluidC max iterations", required=True)
args = parser.parse_args()

print(f"Loading graph {args.graph.as_posix()!r}...")

graph = nx.read_edgelist(args.graph)

print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())

print("Running FluidC...")

assert args.seed >= 0, "Seed must be non-negative"
assert (
    args.ground_truth > 0
), "FluidC k argument must be a positive number (at least one)"
assert args.max_iter > 0, "Max iterations must be a positive number (at least one)"

start = time.time()
communities = nx.community.asyn_fluidc(
    graph, args.ground_truth, max_iter=args.max_iter, seed=args.seed
)
end = time.time()

seconds = round(end - start)
print(f"Elapsed: {seconds} seconds")

print("Saving found communities...")

graph_name = args.graph.name.split(".")[
    0
]  # Keep only the graph name (eg. com-youtube).

base_path = Path("results")
base_path.mkdir(exist_ok=True)
communities_path = base_path / Path(
    f"{graph_name}.fluidc.{args.seed}.{args.max_iter}.{seconds}.txt.gz"
)
with gzip.open(communities_path, "wt") as file:
    for community in list(communities):
        file.write(" ".join(map(str, community)) + "\n")

print(f"Saved to {communities_path.as_posix()!r}")
