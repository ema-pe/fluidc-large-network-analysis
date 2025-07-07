#!/usr/bin/env bash

# For each graph there are two files: the graph and the list of ground-truth
# communities.
URLS=(
    # Youtube social network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-Youtube.html
    "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz"

    # LiveJournal social network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-LiveJournal.html
    "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-lj.all.cmty.txt.gz"

    # Friendster social network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-Friendster.html
    # WARNING: Too big to fit in RAM!
    #"https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
    #"https://snap.stanford.edu/data/bigdata/communities/com-friendster.all.cmty.txt.gz"

    # Orkut social network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-Orkut.html
    # WARNING: Too big to fit in RAM!
    #"https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
    #"https://snap.stanford.edu/data/bigdata/communities/com-orkut.all.cmty.txt.gz"

    # DBLP collaboration network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-DBLP.html
    "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz"

    # Amazon product co-purchasing network and ground-truth communities
    # Source: https://snap.stanford.edu/data/com-Amazon.html
    "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.dedup.cmty.txt.gz"
)

# Change to the directory where the script is located.
cd "$(dirname "$0")"

# Download in parallel all graphs and ground-truth communities. With the
# --no-clobber options, the files won't be downloaded twice. Use also the
# original names (--remote-name-all).
curl --location --remote-name-all --no-clobber --parallel --progress-bar "${URLS[@]}"
