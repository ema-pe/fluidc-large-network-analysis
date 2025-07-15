"""This module runs several FluidC experiments using predefined graphs, seeds,
and maximum iteration limits. It saves the results to the "Results" directory on
disk. The module can only be executed as a script."""
# pylint: disable=import-error

from pathlib import Path
import itertools

from tqdm import tqdm

import fluidc

output_dir = Path("results")

# (graph_path, ground_truth)
networks = [
    ("dataset/com-amazon.ungraph.txt.gz", 75149),
    ("dataset/com-dblp.ungraph.txt.gz", 13423),
    ("dataset/com-youtube.ungraph.txt.gz", 14870),
]

# RNG seeds.
seeds = [
    773956,
    654571,
    438878,
    433015,
    858597,
    85945,
    697368,
    201469,
    94177,
    526478,
]

# FluidC max iterations parameters.
max_iters = [2, 5, 7, 10, 15]


# Create all possibile experiments with networks, seeds and max_iter arguments.
all_args = [
    (graph, truth, seed, max_iter)
    for (graph, truth), seed, max_iter in itertools.product(networks, seeds, max_iters)
]

# Run all experiments in a sequential mode. Why? Because with some networks the
# RAM can explode.
for i in tqdm(range(len(all_args))):
    graph, truth, seed, max_iter = all_args[i]

    print(
        f"Running experiment: graph={graph}, truth={truth}, seed={seed}, max_iter={max_iter}"
    )
    fluidc.run(graph, seed, truth, max_iter, results_dir=output_dir)
