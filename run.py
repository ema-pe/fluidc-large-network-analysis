from pathlib import Path
import subprocess
import itertools
import multiprocessing

# (graph_path, ground_truth)
networks = [
    ("dataset/com-amazon.ungraph.txt.gz", "75149"),
    ("dataset/com-dblp.ungraph.txt.gz", "13423"),
    ("dataset/com-lj.ungraph.txt.gz", "576120"),
    ("dataset/com-youtube.ungraph.txt.gz", "14870"),
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
    975622,
    735752,
    761139,
    717477,
    786064,
    513226,
    128113,
    839748,
    450385,
    500351,
]

# FluidC max iterations parameters.
max_iters = [10, 50, 100, 200, 500, 1000]


def run_experiment(args):
    graph, truth, seed, max_iter = args
    prefix = f"{Path(graph).name}.fluidc.{seed}.{max_iter}"
    cmd = [
        "python",
        "test.py",
        "--seed",
        str(seed),
        "--graph",
        graph,
        "--ground-truth",
        str(truth),
        "--max-iter",
        str(max_iter),
    ]
    print(f"Running: {' '.join(cmd)}")

    # Start experiment.
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Print the experiment's output line by line.
    for line in process.stdout:
        print(f"{prefix}: {line.rstrip()}")

    # Wait for completion before continue.
    return process.wait()


# Create all possibile experiments with networks, seeds and max_iter arguments.
all_args = [
    (graph, truth, seed, max_iter)
    for (graph, truth), seed, max_iter in itertools.product(networks, seeds, max_iters)
]

# Run all experiments in a sequential mode. Why? Because with some networks the
# RAM can explode.
for i in range(len(all_args)):
    ret = run_experiment(all_args[i])
    print(f"Experiment {i+1}/{len(all_args)} completed with exit code {ret}.")
