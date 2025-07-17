# FluidC on large and real complex networks

This small project evaluates the [FluidC algorithm](doi.org/10.1007/978-3-319-72150-7_19) for community detection in large-scale and real-data complex networks. It includes Python scripts to run experiments, analyze results, and generate plots.

This was developed as part of a computer science doctoral course exam called "Graph Theory and Algorithms" at the University of Milano-Bicocca for the 2024-2025 academic year. A brief report on the experiments and their results is available for download. You can also get the raw results files (plots and CSV files). The project will not be updated after submission, and the code is provided as-is.

The git repository is available online on both [GitLab](https://gitlab.com/ema-pe/fluidc-large-network-analysis) and [GitHub](https://github.com/ema-pe/fluidc-large-network-analysis). However, GitHub is only a mirror of GitLab.

## How to run

1. Prepare the virtual environment (Python 3 is required):

    ```bash
    $ git clone https://gitlab.com/ema-pe/fluidc-large-network-analysis.git
    $ cd fluidc
    $ python3 -m venv .env
    $ source .env/bin/activate
    $ pip install -r requirements.txt
    ```

2. Download the complex networks and the ground-truth communities (from [SNAP](https://snap.stanford.edu/data/index.html)). A Bash script is provided to automatically download them:

    ```bash
    $ chmod u+x dataset/download.sh
    $ ./dataset/download.sh
    ```

3. Get the number of ground-truth communities for each network, using the `ground_truth.py` script. This number is a required parameter (`k`) for the FluidC algorithm. You may need to update the `networks` variable in `run.py` with the correct ground truth values for each network. 

    ```bash
    $ python ground_truth.py --communities dataset/com-amazon.all.dedup.cmty.txt.gz
    # Example output: "dataset/com-amazon.all.dedup.cmty.txt.gz": 75149
    ```

4. Run FluidC algorithm with various configurations. **Warning:** this process runs sequentially and can be very time-consuming, depending on the size of the networks. The communities are saved in the `results/` directory. If you want to run just a single FluidC execution, you can call directly the `fluidc.py` script.

    ```bash
    $ python run.py
    ```

5. Finally, use the plot.py script to analyze the output from the experiments. This script generates metric CSV files and plots and saves them in the `results/` and `results/plots/` directories. The calculated metrics are execution time, **normalized mutual information (NMI)**, **adjusted rand index (ARI)**, and **cluster purity** for each FluidC execution compared to the ground truth.

    ```bash
    $ python plot.py --results-dir results --graph-name com-amazon com-dblp com-youtube
    ```

## License

Copyright (c) 2025 Emanuele Petriglia <inbox@emanuelepetriglia.com>. This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
