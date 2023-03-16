# Bayesian Machine Learning Project - MVA Master - Eugène Berta

This repository contains my project on the paper: Variational Refinement for Importance Sampling Using the Forward Kullback-Leibler Divergence

By: Ghassen Jerfel, Serena Wang, Clara Fannjiang, Katherine A. Heller, Yian Ma, Michael I. Jordan

Available at: https://arxiv.org/abs/2106.15980v1

- utils.py contains the FKL and RKL losses described in the paper, and classes to handle multivariate normal distributions and Gaussian Mixture Models in our experiments.
- ImportanceSamplingFKL.ipynb contains all our experiments as described in the report.
- Report.pdf is the project report associated with this repository.
- requirements.txt contains the list of dependencies of the project.

## Running the code

You can install the dependencies using:

```
conda create -n <environment-name>
conda activate <environment-name>
pip3 install -r requirements.txt
```
