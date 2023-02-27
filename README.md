<div align="center">

# ⚡️ Lightning Equivariant

## Distributed Geometric Deep Learning with Lightning AI

  <img src="assets/lightning-equivariant.png" width="350" />

[![LightningAI](http://img.shields.io/badge/docs-LightningAI-4b44ce.svg)](https://lightning.ai/)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description

Lightning Equivariant provides an interactive kind of literature review, revisiting a number of important methodological developments in the neural network literature but presented under the unifying theoretical lens of [Geometric Deep Learning](https://geometricdeeplearning.com/). This repository demonstrates the discovery utility afforded by the mere act utilizing one infrastructure framework for collating a vast number of architectures trained on various tasks, with associated datasets and benchmarks, along with reusable access to tests and utility functions for distributed model training, all documented under a unifying theoretical lens.

## Directory Structure and Usage

```
.
├── README.md
|
├── experiments                         # Synthetic experiments
│   ├── introduction.ipynb              # A gentle introduction to repository with interactive blocks to demonstrate research
│   ├── incompleteness.ipynb            # Experiment on counterexamples from Pozdnyakov et al.
│   ├── kchains.ipynb                   # Experiment on k-chains
│   └── rotsym.ipynb                    # Experiment on rotationally symmetric structures
| 
└── src                                 # Lightning Equivariant
    ├── blocks                          # Functionality 
    │   ├── irreps_tools.py
    │   ├── radial.py
    │   └── tensorproduct.py        
    │
    ├── data                            # Functionality used for training, plotting, etc.
    │   └── nbody_datamodule.py         # Simulate n-body systems with adjustible numerical accuracy
    │
    └── models                          # 5-Gs of Geometric Deep Learning
        ├── grids                 # Grids and Euclidean Spaces
        │
        ├── groups                # Groups and Homogeneous Spaces
        │   ├── ecnns.py                    # (Cohen & Welling, 2016)
        │   └── harmonic.py                 # (Worrall et al., 2017)
        │
        ├── graphs                # Sets and Graphs
        │   ├── deep_sets.py      # (Zaheer et al., 2017)
        │   ├── gcnn.py           # (Kipf and Welling, 2016) (convolution-based)
        │   ├── gcnn2.py          # (Defferrard et al., 2016) (convolution-based)
        │   ├── sgc.py            # (Wu et al., 2019) (convolution-based)
        │   ├── gat.py            # (Veličković et al., 2018) (attention-based)
        │   ├── monet.py          # (Monti et al., 2017) (attention-based)
        │   └── gaan.py           # (Zhang et al., 2018) (attention-based)
        │
        ├── geodesics             # Geodesics and Manifolds
        │
        │── guages                # Guages and Bundles
        │
        └── geometric_gnn         # Geometric Graphs and Meshes
            ├── cormorant.py      # (Anderson & Hy & Kondor, 2019)
            ├── dimenet.py        # (Klicpera et al., 2020)
            ├── egnn.py           # (Satorras et al., 2021)
            ├── gvpgnn.py         # (Jing et al., 2020) 
            ├── mace.py           # (Batatia, 2022)
            ├── nequip.py         # (Batzner, 2022)
            ├── painn.py          # (Schütt, 2021)
            ├── schnet.py         # (Schütt et al., 2018)
            ├── se3transformer.py # (Fuchs et al., 2020)
            ├── segnn.py          # (Brandsetter et al., 2022)
            ├── spherenet.py      # (Lui et al., 2021)
            ├── steerable3d.py    # (Weiler et al., 2018)
            └── tensorfield.py    # (Thomas et al., 2018)
```

## How to run
First, install dependencies
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```


### Citation (PLEASE READ CAREFULLY)

It should be noted that there does not exist a single architecture presented in this repo that constitute the original work of this author. If any model presented within this repo provides valuable insight in your own work, please refer to the inline comments to ensure proper a citation trail.

```
@article{lightning-equivariant,
  title={Lightning Equivariant: Unifying Methodological Developments with Lightning Infrastrusture},
  author={Clayton Curry},
  journal={},
  year={2023}
}
```
