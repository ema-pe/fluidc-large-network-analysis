# FluidC on large and real complex networks

This project provides an evaluation of the FluidC algorithm for community detection in large-scale and real-data complex networks. It includes scripts to run experiments, analyze results, and generate plots.

This was developed as part of a doctoral course exam called "Graph Theory and Algorithms". A brief report on the experiments and their results is available here: [Link to Report (coming soon)]. The project will not be updated after submission, and the code is provided as-is.

## How to run

1. Prepare the virtual environment (Python 3 required):

    ```bash
    git clone https://github.com/your_username/fluidc.git
    cd fluidc
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Download the complex networks and the ground-truth communities (from [SNAP](https://snap.stanford.edu/data/index.html)). A Bash script is provided to automatically download them:

    ```bash
    bash dataset/download.sh
    ```

3. Get the number of ground-truth communities for each dataset, using the `ground_truth.py` script. This number is a required parameter (`k`) for the FluidC algorithm. You may need to update the `networks` variable in `run.py` with the correct ground truth values for each network. 

    ```bash
    python ground_truth.py --communities dataset/com-amazon.all.dedup.cmty.txt.gz
    # Example output: "dataset/com-amazon.all.dedup.cmty.txt.gz": 75149
    ```

4. Run FluidC algorithm with various configurations. **Warning:** this process runs sequentially and can be very time-consuming, depending on the size of the networks. The found communities are saved in the `results/` directory. If you want to run just a single FluidC execution, you can call directly the `fluidc.py` script.

    ```bash
    python run.py
    ```

5. Finally, use `plot.py` to analyze the output from the experiments. This script will generate metric CSV files and plots, saving them in the `results/` and `results/plots/` directories.

    ```bash
    python plot.py --results-dir results --graph-name com-amazon com-dblp com-youtube
    ```

## License

Copyright (c) 2025 Emanuele Petriglia <inbox@emanuelepetriglia.com>. This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
