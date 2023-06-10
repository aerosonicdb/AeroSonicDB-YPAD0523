# AeroSonicDB-YPAD0523
Download and get started with the AeroSonicDB-YPAD0523 dataset of low-altitude aircraft sounds.

## Requirements
- Python 3

## Get started
1. ### Clone repository
Clone this repository and enter it.
```
git clone https://github.com/aerosonicdb/AeroSonicDB-YPAD0523.git
cd AeroSonciDB-YPAD0523
```
2. ### Install
Run the following to install the package and dependencies.
```
pip install -e .

python install setup.py
```
3. ### Download the dataset
The simplest way to download the datset is to open the "AeroSonicDB_GetStarted" jupyter notebook and run all cells. Similarly, you can use the following lines of code in a python script or notebook.
```
import aerosonicdb.data
aerosonicdb.data.download()
```
Alternatively, straight from the command line with
```
python -m aerosondicb.data.download
```
*The dataset will be downloaded to the data/raw directory*

