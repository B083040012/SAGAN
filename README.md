# SAGAN

## About This Repo

* Single Path One-Shot NAS based on [STDN](https://github.com/tangxianfeng/STDN) neural network for predicting bike demand in Bike Sharing System
    * the architecture of `supernet_training.py`, `retrain_network.py`, and `attention.py` is according to [STDN](https://github.com/tangxianfeng/STDN) neural network
* references:
    * [Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction](https://arxiv.org/abs/1803.01254)
    * [Single path one-shot neural architecture search with uniform sampling](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_32)

## Usage

```
git clone https://github.com/B083040012/SAGAN.git
```
* first unzip the region-level dataset in ./data/region
* for differnet external features
    * no poi feature for `region level` now
    * `git checkout no_external_feature` for no external feature (exclude weather & poi)
    * `git checkout poi_only` for poi feature only (exclude weather)
    * `git checkout weather_only` for weather feature only (exclude poi)
    * details for region & station level in different branches

| branches / level | Region | Station |
| ------------------ | ------ | ------- |
| `main` | weather:✔️<br>poi:❌ | weather:✔️<br>poi:✔️ |
| `no external feature` | weather:❌<br>poi:❌ | weather:❌<br>poi:❌ |
| `poi_only` | weather:❌<br>poi:❌ | weather:❌<br>poi:✔️ |
| `weather_only` | weather:✔️<br>poi:❌ | weather:✔️<br>poi:❌ |

### From Command Line

* args for `agnet.py`
    * dataset: `region / station`: different preprocessing dataset
    * project_name: enter project name for each parameter set
    * start_phase: `1~4`: choose the start phase
    * end_phase: `1~4`: choose the end phase
    * running_times: repeat experiment multiple times from `start_phase` to `end_phase` in the same project
    * debug_mode: required for development, with less dataset sampling, epoches, ... for debugging
```
conda activate %conda_env%
python agnet.py
# or
python agent.py --dataset=%dataset% --project_name=%project_name% --start_phase=%start_phase% --end_phase=%end_phase% --running_times=%running_times% --debug_mode=%debug%
```

### From `run.bat`

* simple script for windows user  to set args of `agent.py`
* can modify the `conda_env` in file

### Some Details

* directory `./project_file/%project_name%` will be created for storing *saved model* and *log file*
* if the project has not executed before, `agent.py` will copy `./parameters.yml` to the project folder to ensure that this project will always executed with same parameters in the future
    * so it is recommanded to **create a new project for a new parameter set**
* in order to reduce the complexity, `agent.py` will repeat each phase `running_times` then move to the next phase

## Dataset

* two types of dataset in the experiment that based on [NYCDataProcessing](https://github.com/lynnpepin/NYCDatasetProcessing) used in STDN
* NYC Citi Bike dataset during 2020/07/01 ~ 2020/08/29 (60 days)

### Region-Level

* detailed feature meaning in [NYCDataProcessing](https://github.com/lynnpepin/NYCDatasetProcessing)
* **extending to the region of whole NYC**
* add new external feature
    1. weather

### Station-Level

* predicting demand of **each station** in the BSS instead of each region
* definition of `flow`: num of trip that arriving to / departing from each station
    * not restricted to the station in specific region like the definition in *region level*
    * exclude the maintenance record
* definition of `volume`: $inflow - outflow$ for each station
* add new external feature
    1. weather
    2. point of interest(poi) near each station