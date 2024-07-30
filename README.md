ST-SCSR

## ST-SCSR: Identifying spatial domains in spatial transcriptomics data via structure correlation and self-representation

### Min Zhang, Wensheng Zhang and  Xiaoke Ma*

### Contributing authors: xkma@xidian.edu.cn;

## Overview

Here, a novel algorithm for spatial domain identification in Spatial Transcriptomics data with Structure Correlation and Self-Representation (ST-SCSR), which integrates  local information, global information and similarity of spatial domains. Specifically, ST-SCSR utilzes matrix tri-factorization to simultaneously decompose expression profiles and spatial network of spots, where expressional and spatial features of spots are fused via the shared factor matrix that interpreted as similarity of spatial domains. Furthermore, ST-SCSR learns affinity graph of spots by manipulating expressional and spatial features, where local preservation and sparse constraints are employed, thereby  enhancing quality of graph. The experimental results demonstrate that ST-SCSR not only outperforms state-of-the-art algorithms in terms of accuracy, but also identifies many potential interesting patterns.

## Prerequisites

### System requirements: 

Machine with 16 GB of RAM. (All datasets tested required less than 16 GB). No non-standard hardware is required.

### Software requirements:

#### Python supprt packages  (Python 3.9.18):

scanpy, igraph, pandas, numpy, scipy, scanpy, anndata, sklearn, seaborn, torch, leidenalg, tqdm.

For more details of the used package., please refer to 'requirements.txt' file

## Tutorial

A jupyter Notebook of the tutorial for 10 $\times$ Visium is accessible from :  

https://github.com/xkmaxidian/stACN/blob/master/tuorial/Tutorial1_10x_Visium%20(DLPFC_dataset).ipynb  

## Compared spatial domain identification algorithms

Algorithms that are compared include: 

* [SCANPY](https://github.com/scverse/scanpy-tutorials)
* [Giotto](https://github.com/drieslab/Giotto)
* [BayesSpace](https://github.com/edward130603/BayesSpace)
* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [SEDR](https://github.com/JinmiaoChenLab/SEDR/)
* [STAGATE](https://github.com/QIFEIDKN/STAGATE)
* [PROST](https://prost-doc.readthedocs.io/en/latest/Installation.html)
* [MNMST](https://github.com/xkmaxidian/MNMST)
* [BANKSY](https://github.com/prabhakarlab/Banksy)
* [DeepST](https://github.com/JiangBioLab/DeepST)

