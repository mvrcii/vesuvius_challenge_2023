# Workflow to get started on a GPU

## Setting up the node
### SSH
Create an `authorized_keys` file and add both SSH keys:

SSH Key Micha: `DUMMY`

SSH Key Marcel: `DUMMY`

### Clone Git Repository
```bash
git clone git@github.com:JakeGonder/kaggle1stReimp.git
```

### Setup Virtual Environment (Conda)
Install `requirements.txt`, as well as other required python packages.

## Setting up the training
### Download Data (Fragments)
Download a single Fragment: 
```bash
./util/download.sh
```
Download all Fragments:
```bash
python util/batch_download_frags.py
```
Download all Masks:
```bash
python download_masks.py
```
### Create Dataset
-> dataset_creation_workflow.md