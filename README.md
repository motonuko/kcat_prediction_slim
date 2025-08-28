# kcat_prediction

This project is a simplified version that extracts the XX component from the project
at https://github.com/AlexanderKroll/kcat_prediction.

## Workflow

### Set up environment

```shell
conda env create -f environment.yml
```

The following three dependencies may not be compatible with  GPU environment.
Please adjust them accordingly:

```shell
  - cudatoolkit=11.0.3
  - cuda-version=11.0
  - nccl=2.25.1.1
```

### Download datasets

Following the instruction of [original project](https://github.com/AlexanderKroll/kcat_prediction), 
Download the data folder from zenodo from the following link

https://doi.org/10.5281/zenodo.8367052

### Configure .env

Place `.env` file at the project root and configure paths.

```shell
ORIGINAL_DATA_DIR="<path-to-the-downloaded-and-unzipped-data-dir>/data"
KCAT_PREDICTION_PROJECT_ROOT="<path-to-the-project-root>"
```

### Place embeddings

Place precomputed embeddings under `data_embedding/` dir.

### Run main script

```shell
python kcat_prediction_slim/main.py "<path-to-the-project-root>/data_embedding/kcat_sequence_embeddings_250420_121652.csv" --run-all
```

### Visualize the result

```shell
python  kcat_prediction_slim/visualize_box.py --target-app-ver v2_1_0 --models 250420_121652 --seeds 42 43 44 45 46 47 48 49 50 51 --result-parent-dir data/results/ > build/viz/v2_1_0/visualize_box.log
```
