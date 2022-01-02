# ProFraGe

#### ETH Zurich, Computer Science Msc, Master Project: Generative Modeling of Protein Fragments

*ProFraGe (**Pro**tein **Fra**gment **Ge**nerator)* is a Python tool developed during my Master Project at ETH Zurich.

The goal of the project was to generate protein fragments via protein-graphs. The work consists of four main phases

* Download and process PDB files to build the main protein dataset
* Extracting, clustering, and validating the protein fragments which will constitute the fragment dataset
* Implementation of deep learning architectures
* Reconstructing the 3D structure of the generated protein graphs

This file provides information on the technical requirements of this tool, as well as instructions and examples on how to build the datasets and run the deep learning models.

## Table of Contents

1. [Tools & Requirements](#tools--requirements)
2. [Building the Dataset](#building-the-dataset)
    * [PDB Dataset](#pdb-dataset)
    * [ProteinNet Dataset](#proteinnet-dataset)
3. [Fragments Extraction](#fragments-extraction)
    * [Mining & Clustering](#mining--clustering)
    * [Tuning via Language Model](#tuning-via-language-model)
4. [Deep Learning Models](#deep-learning-models)

## Tools & Requirements

I used the following external C++ tools

* [Stride](http://webclu.bio.wzw.tum.de/stride/)
* [ConFind](https://grigoryanlab.org/confind/)

I used Python3 coupled with the Anaconda virtual environment throughout my work. Below is the list of requirements.

```
conda==4.10.1
leidenalg==0.8.4
mmtf-python==1.1.2
numpy==1.20.3
ProDy==2.0
scipy==1.6.3
torch==1.7.1
torch-geometric==1.7.1
```

I used the BioPython package with Anaconda.

## Building the Dataset

I experimented with two datasets during my work, but I ultimately settled on ProteinNet. Nevertheless, because the two datasets have their differences, I provide instructions on how to use them both.

### PDB Dataset

Switch to the `profrage/` directory. Below is how to run the PDB pipeline with the default parameters. For more information on the parameters, simply run `python3 pipeline_pdb -h`.

```
python3 pipeline_pdb path/to/store/
```

Note that this step requires downloading PDB files in `.pdb.gz` format. See *pdb/ids/README.md* for details on how to download them.

### ProteinNet Dataset

Switch to the `profrage/` directory. Again, for more information on the parameters simply run `python3 pipeline_proteinnet -h`.

```
python3 pipeline_proteinnet path/to/store/
```

Note that the file *../pdb/casp11/training_30.zip* needs to be unzipped before running the previous command.

For more information please see *pdb/casp11/README.md*.

## Fragments: Mining & Clustering

Switch to the `profrage/` directory. I show an example where I use the ProteinNet dataset

```
python3 pipeline_mine path/to/store/ path/to/cmaps/ path/to/stride path/to/filtered/ path/to/clustered/
```

Note that for ProteinNet, this procedure is to be carried out for all the splits (training/validation/test).

Further note that if contact maps (`CMAP`) are used, then the ConFind tool should be run first on the full proteins.

### Tuning via Language Model

There is no dedicated pipeline for this part. Switch to the `profrage/` directory.
The example below shows an example on how to use the tuning.

```python
import leidenalg
from fragment.tuners import leiden_gridsearch

train_set_dir = 'path/to/training/'
validation_set_dir = 'path/to/validation/'
cmap_train_dir = 'path/to/cmap/training/'
cmap_validation_dir = 'path/to/cmap/validation/'
stride_dir = 'path/to/stride/'

leiden_params = {'partition': [leidenalg.ModularityVertexPartition],
                 'contacts': ['dist', 'cmap'],
                 'bb_strength': [0.4, 0.5, 0.6],
                 'f_thr': [0.1],
                 'dist_thr': [12],
                 'n_iters': [2, 5, 10],
                 'max_size': [30]}
first_lvl_params = {'score_thr': 0.5}
second_lvl_params = {'score_thr': 0.5,
                     'bb_atoms': True}
third_lvl_params = {'rmsd_thr': 2.0,
                    'length_pct': 0.6}
range_params = {'lower': 12}

leiden_gridsearch(train_set_dir, validation_set_dir, cmap_train_dir, cmap_validation_dir, stride_dir, leiden_params, first_lvl_params, second_lvl_params, third_lvl_params, range_params)
```

## Deep Learning Models

Switch to the `profrage/` directory.

```
python3 pipeline_generate full ProRNN <path/to/store>/training/ <path/to/store>/*validation/ <path/to/store>/testing/ path/to/stride/ path/to/dataset_directory/ path/to/model_directory/
```

where `dataset_directory/` and `model_directory/` refer to the directories where features and model weights will be saved respectively.

Note that for this the file `profrage/generate/args.py` is to be configured. The current version of the file holds the default parameters.
