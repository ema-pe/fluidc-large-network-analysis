"""Module for loading unique communities from gzipped files.

Usage as a script:
    python ground_truth.py --communities dataset/graphXX.txt.gz ...
"""

# Copyright (c) 2025 Emanuele Petriglia <inbox@emanuelepetriglia.com>
# All rights reserved. This file is licensed under the MIT license.

from pathlib import Path
import gzip
import argparse


def get_unique_communities(community_path):
    """Returns the set of unique communities for the given gzipped file.

    Args:
        community_path: A Path object pointing to a gzipped file scontaining
          community lists (one per line, space-separated node IDs).

    Returns:
        The set of unique communities (tuple of node IDs).
    """
    communities = set()
    with gzip.open(community_path, "rt") as file:
        for line in file:
            # Each line is a list of node IDs separated by space. Force integer
            # type since the node IDs are integers.
            node_ids = [int(x) for x in line.strip().split()]

            # The ground-truth file may contain duplicated communities. We also
            # need to convert lists to tuples to have a hashable type for the set.
            communities.add(tuple(sorted(node_ids)))

    return frozenset(communities)


def load(graph_name):
    """Returns the ground truth communities for the given graph name."""
    try:
        ground_truth_path = Path(f"dataset/{graph_name}.all.cmty.txt.gz")
        ground_truth_comm = get_unique_communities(ground_truth_path)
    except FileNotFoundError:
        ground_truth_path = Path(f"dataset/{graph_name}.all.dedup.cmty.txt.gz")
        ground_truth_comm = get_unique_communities(ground_truth_path)

    return ground_truth_comm


def _main():
    parser = argparse.ArgumentParser(
        description="Print the number of communities for each given community file"
    )
    parser.add_argument(
        "--communities", type=Path, nargs="+", help="Communities result path(s)"
    )
    args = parser.parse_args()

    for community_path in args.communities:
        communities = get_unique_communities(community_path)
        print(f"{community_path.as_posix()!r}: {len(communities)}")


if __name__ == "__main__":
    _main()
