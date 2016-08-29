# 2DMeanField

## Overview
This repository contains an implementation of the Variational Mean Field Inference algorithm for image segmentation. This implementation is utilising the contributions of the paper entitled "Efficient Inference in Fully Connected CRFs with
Gaussian Edge Potentials" by Philipp Krahenbuhl et al.

Included is a CPU and GPU implementation of the algorithm utilising direct filtering for the message passing phase. There is also an additional "Permutohedral" branch which utilises the Permutohedral Lattice for high performance bilateral and gaussian filtering. However, at this time it requires some debugging and shall be fixed soon.

## Notice
The code in this repository is distributed under a BSD license, except either where indicated otherwise or where the source does not contain a BSD license header. The Permutohedral branch of this repository contains code sourced from the accompanying materials for the paper "Fast High-Dimensional Filtering Using the Permutohedral Lattice" by Andrew Adams et al. For licensing conditions of this code, please contact the original authors.
