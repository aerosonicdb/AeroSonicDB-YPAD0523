## 2024 UPDATE: 
This branch has been created to preserve the original data presented in the publication listed below. Please note that running the notebooks in this branch may not produce the exact same cross-validation results as published, however they will fall within statistically significant ranges of the original paper.

Unless you have a specific reason to work with this branch, it is recommended to work with the actively maintained main branch https://github.com/aerosonicdb/AeroSonicDB-YPAD0523

# AeroSonicDB-YPAD0523
Download and get started with the AeroSonicDB-YPAD0523 dataset of low-altitude aircraft sounds.
https://zenodo.org/record/8371595

### Publication
The methodology for collecting this dataset is described in; Downward, B., & Nordby, J. (2023). The AeroSonicDB (YPAD-0523) Dataset for Acoustic Detection and Classification of Aircraft. [ArXiv, abs/2311.06368](https://arxiv.org/abs/2311.06368).

## Requirements
- Python 3.11
- Tensorflow 2.14.0 - 2.15.1

## Get started
1. ### Clone repository
Clone this repository and enter it.
```
git clone https://github.com/aerosonicdb/AeroSonicDB-YPAD0523.git
cd AeroSonicDB-YPAD0523
```
2. ### Create virtual environment and activate it (optional)
This step will help to avoid potential dependency conflicts.

3. ### Install
Run the following to install the package and dependencies.
```
pip install -e .

```
4. ### Download the dataset and meta files
The simplest way to download the dataset is to open the "AeroSonicDB_GetStarted" jupyter notebook and run all cells. Similarly, you can use the following lines of code in a python script or notebook.
```
import aerosonicdb.data
aerosonicdb.data.download()
```
Alternatively, straight from the command line with
```
python -m aerosonicdb.data.download
```
*The dataset will be downloaded to the data/raw directory*

