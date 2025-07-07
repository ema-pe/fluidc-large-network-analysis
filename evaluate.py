from pathlib import Path
import argparse

import ground_truth


def main(graph_name):
    # Load ground truth communities.
    try:
        ground_truth_path = Path(f"dataset/{graph_name}.all.cmty.txt.gz")
        ground_truth_comm = ground_truth.get_unique_communities(ground_truth_path)
    except FileNotFoundError as e:
        ground_truth_path = Path(f"dataset/{graph_name}.all.dedup.cmty.txt.gz")
        ground_truth_comm = ground_truth.get_unique_communities(ground_truth_path)

    print(len(ground_truth_comm))

    # WIP: load also from results/*
    # Store data in dictionary, then evaluate and generate Pandas DataFrame and
    # save the result to JSON (and also print to stdout).


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", help="Graph name", required=True)

    args = parser.parse_args()

    main(args.graph_name)
