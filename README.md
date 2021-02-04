# OGB Revisited

## Installation
* Tested with Python 3.7, PyTorch 1.7.1., and PyTorch Geometric 1.6.3
* Set up an Anaconda environment: `./setup.sh` 
<br/>(if you install locally w/o CUDA you may need to adapt the torch installation command)
<br/> Aso, you need to have installed [Anaconda](https://www.anaconda.com/products/individual#Downloads). See the [Installation instructions](https://docs.anaconda.com/anaconda/install/).
* Alternatively, install the above and the packages listed in [requirements.txt](requirements.txt)

NOTE I did not test the setup since I have everything installed on my machines. Please let me know if we have to fix this description, add packages or similar.

## Overview

* `/data` 
<br/>The datasets are downloaded into here by default. We also store auxiliary data here (e.g., index files selecting subsets of the training sets to train on those).
* `/ogb-examples` 
<br/>The original "examples" directory from the [Open Graph Benchmark (OGB)](https://github.com/snap-stanford/ogb). We just have it here to easy check them and create our experiments based on those.
* `/papers` 
<br/>Papers about the GNNs we plan to look at in the project.
* `/scripts`
<br/>Scripts for running the experiments.
* `/src` 
<br/> This is the code we plan to write. I created an example experiment for the ogbn-proteins dataset: ogbn_pro.py. 
The corresponding example(s) from OGB is in ogb-examples/nodeproppred/proteins. But they copy a lot of code there. So I suggest to have a single logger/parser etc. file in our code and reuse those across experiments.
