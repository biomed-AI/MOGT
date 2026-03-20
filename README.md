# MOGT
MOGT is a semi-supervised graph neural network that integrates the multi-omics data of genes and the topology information of biological networks.To construct the multi-omics representation graph, we represented genes as nodes and SNP-SNP interactions as edges. The node features were the multi-omics data of genes, including differential expression (DE), enhancer-promoter interactions (EPI) in patients and controls, and gene expression in five brain regions（Parietal Lobe, Frontal Lobe, Temporal Lobe, Cerebellum, and occipital Lobe）in adolescents and adults.

![image](https://github.com/biomed-AI/MOGT/blob/main/img/Framework.tif)

## Versioned Release and Archival

The exact version of the code used in the manuscript is archived as:

GitHub repository:https://github.com/biomed-AI/MOGT

GitHub release:v1.0.0-paper

Commit hash:1a02176207a6e07c4a5b03bf325e22c28f3ca034

All results reported in the manuscript were generated using this tagged release.

Future updates will not modify this release to ensure long-term reproducibility.

## Documentation
MOGT documentation is available through [Documentation](https://github.com/biomed-AI/MOGT/blob/main/README.md).

## Conda Environment
MOGT was written in Python 3.8, and should run on any OS that support pytorch and pyg. 
We recommend using conda to configure the code runtime environment:
```
conda create -n MOGT python=3.8.20
```
MOGT has the following dependencies:
```
matplotlib==3.7.2
model==0.6.0
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.2
PyYAML==6.0.2
scikit_learn==1.3.2
torch==1.11.0
torch_geometric==2.0.4
transformers==4.46.3
utils==1.0.2
wandb==0.21.3
```
Install commmands of torch_scatter and torch_sparse should be adjusted according to pytorch and cuda version, see [PyG 2.0.3 Installation](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html)

## Installation
We recommend getting MOGT using Git from our Github repository through the following command:

```
git clone https://github.com/biomed-AI/MOGT.git
```
To verify a successful installation, just run:
```
python main.py -cv  # Importing the relevant libraries may take a few minutes.
```

## Tutorial
This tutorial demonstrates how to use MOGT functions with a demo dataset (SCZ as an example). 
Once you are familiar with MOGT’s workflow, please replace the demo data with your own data to begin your analysis. 
[Tutorial notebook](https://github.com/biomed-AI/MOGT/blob/main/Tutorial.ipynb) is available now.

### How to prepare input data

We recommend getting started with MOGT using the provided demo dataset. When you want to apply MOGT to your own multi-omics dataset, please refer to the following tutorials to learn how to prepare input data.

Overall, the input data consists of two parts: the graph, constructed from SNP-SNP interaction and the node feature including DE, EPI, and gene expression in five brain regions（Parietal Lobe, Frontal Lobe, Temporal Lobe, Cerebellum, and Occipital Lobe）in adolescents and adults.

 If you are unfamiliar with MOGT, you may start with our data used in the paper to save your time. For SCZ, the input data as well as the label information are uploaded [here](https://github.com/biomed-AI/MOGT/tree/main/data). For SNP-SNP interaction, download demo data from [Zenodo](). If you start with this data, you can skip the _step 1_ about _How to prepare input data_.
 The following steps from 1.1~1.3 can be found in our source code [data_preprocess_cv.py](https://github.com/biomed-AI/MOGT/blob/main/data_preprocess_cv.py).

>The labels should be collected yourself if you choose analyze your own data.


---

### Hyperparameters

- To reduce the number of parameters and make training feasible within time and resource constraints, the input graphs were sampled using neighbor sampler. The subgraphs included all first and second order neighbors for each node and training was performed on these subgraphs.
- warm-up strategy for learning rate is employed during the initial training phase.
- To avoid overfitting and over-smoothing, we implemented an early stop strategy that would terminate training if the validation performance did not improve for 50 consecutive epochs.
- To improve model robustness and reduce overfitting, the dropout rate was set to 0.4 for the modules.

---

## Questions and Code Issues
If you are having problems with our work, please use the [Github issue page](https://github.com/biomed-AI/MOGT/issues).