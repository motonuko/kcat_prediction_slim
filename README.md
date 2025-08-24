# kcat_prediction

This project is a simplified version that extracts the XX component from the project
at https://github.com/AlexanderKroll/kcat_prediction.

## Workflow

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
conda run -n kcat_prediction python kcat_prediction_slim/main.py "<path-to-the-project-root>/data_embedding/kcat_sequence_embeddings_250420_121652.csv" --run-all
```

### Visualize the result

```shell
conda run -n kcat_prediction python  kcat_prediction_slim/visualize_box.py --target-app-ver KCAT_APP_VER  2.1.0 --models 250420_121652 --seeds 42 43 44 45 46 47 48 49 50 51
```
