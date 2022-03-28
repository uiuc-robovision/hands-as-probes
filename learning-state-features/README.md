# Setup
## Install Dependencies
<!-- ### 1. Detectron2
    cd ./lib/detectron2
    python -m pip install -e detectron2 -->

<!-- ### 2. PYSOT
    cd ./lib/pysot
    python setup.py build_ext --inplace -->
Please note that this library is different from the one used in ACP, so run the following setup before running state-features pretraining and evaluation commands.

    cd ./lib/epic_kitchens_hoa
    python setup.py develop

## Download and Generate Datasets
The `./datasets` directory should have the following structure after downloading and generating all the datasets.
```
...
learning-state-features
├── configs
├── data
├── dataset_generation
├── datasets
│   ├── mit_states
│   │   ├── release_dataset
│   │   │   ├── adj_ants.csv
│   │   │   ├── images/
│   │   │   │   ├── 'adj aluminum'/
│   │   │   │   ├── 'adj animal'/
│   │   │   │   ├── ...
│   │   │   │   ├── 'young iguana'/
│   │   │   │   ├── 'young tiger'/
|   |   |   ├── meta_info/
|   |   |   ├── README.txt
│   ├── EPIC-KITCHENS
│   │   ├── P01/
│   │   |   ├── rgb_frames/
│   │   |   |   ├── P01_01/
│   │   |   |   |   ├── frame_0000000001.jpg
│   │   |   |   |   ├── frame_0000000002.jpg
│   │   |   |   |   ├── ...
│   │   |   |   ├── P01_02/
│   │   |   |   ├── ...
│   │   |   |   ├── P01_18/
│   │   |   |   ├── P01_19/
│   │   ├── ...
│   │   ├── P37/
│   ├── tracks_partitioned
│   │   ├── ioumf/
│   │   │   ├── track_detections_train.pkl
│   │   │   ├── track_detections_validation.pkl
│   │   │   ├── images/
│   │   │   │   ├── P01_01_01_0_0.jpg
│   │   │   │   ├── P01_01_01_100_0.jpg
│   │   │   │   ├── ...
│   │   │   │   ├── P29_04_01_8_0.jpg
│   │   │   │   ├── P29_04_01_9_0.jpg
│   ├── tracks_partitioned_hand
│   │   ├── ioumf/
│   │   │   ├── track_detections_train.pkl
│   │   │   ├── track_detections_validation.pkl
│   │   │   ├── images/
│   │   │   │   ├── P01_01_01_0_0.jpg
│   │   │   │   ├── P01_01_01_100_0.jpg
│   │   │   │   ├── ...
│   │   │   │   ├── P29_04_01_8_0.jpg
│   │   │   │   ├── P29_04_01_9_0.jpg
│   ├── detections_partitioned
│   │   ├── P01/
│   │   │   ├── P01_01.pkl
│   │   │   ├── ...
│   │   │   ├── P01_19.pkl
│   │   ├── P02/
│   │   ├── ...
│   │   ├── P31/
```

### 1. EK-100 RGB Frames
Navigate to [this repository](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and follow the instructions to download **RGB Frames** only. Extract them to `./datasets/EPIC-KITCHENS`. 

### 2. Tracks (ours)
<!-- HOA
    
    asd

Ground Truth 
    
    asd -->

    python dataset_generation/extract_handobj_interaction_tracks.py

### 2. Retrained Detections 
Download from [Box](https://uofi.box.com/s/bwrk54elgefadhiymkq4ohvl82c8mhmr) and extract to `./datasets/detections_partitioned`.

### 3. MIT-States
This is only needed for the MIT-States baseline experiment. Navigate to [the official website](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) and download the dataset. Extract it to `./datasets/mit_states`. Please download and extract the whole folder when using downloaded models.

</br>

# Experiment Results and Commands
GPU can be specified by adding `--gpu <int>` (default is 0). Seed can be specified by supplying `--seed <int>` (default is 0). Note, the results are reported on the EPIC-STATES test set averaged over pretraining seeds 0, 100, and 417.

| Model           | mAP | Command   | Model Download
| --------------- | --     | --        | --       |
| TCN             | 73.4   | <details><summary>Show</summary>`python train_TCN.py --name tcn`</details> |
| SimCLR          | 81.0   | <details><summary>Show</summary>`python train_simCLR.py --name simclr --object_crops`</details> |
| SimCLR + TCN    | 77.4   | <details><summary>Show</summary>`python train_simCLR.py --name simclr_tcn --object_crops --config configs/simclr_tcn.yaml`</details> |
| Action Class.   | 77.9   |
| MIT States      | 81.9   | <details><summary>Show</summary>`python train_mit_states.py --name mit_states --adj`</details> |
|                 |        |           | |
| TSC             | 84.2   | <details><summary>Show</summary>`python train_temporal_simCLR.py --name tsc`</details> | [Box](https://uofi.box.com/s/tu6ukmousnr5ajuhfxfq61x8z88crck4)
| TSC+OHC         | 84.8   | <details><summary>Show</summary>`python train_hand_objects.py --name ohc`</details> | [Box](https://uofi.box.com/s/vschh47kvw4gogkqtnt73be2lh322s9x)
| TSC+OHC [No motion] | N/A | <details><summary>Show</summary>`python train_hand_objects.py --name ohc_appearance --config configs/hand_objects_appearance.yaml`</details> |
| TSC+OHC [No appearance] | N/A | <details><summary>Show</summary>`python train_hand_objects.py --name ohc_motion --config configs/hand_objects_motion.yaml`</details> |



## Evaluation
Please download and extract pretrained models into `./models/ioumf` so that the path looks like `./models/ioumf/<model_dir>/checkpoint_200.pth`, or please run training commands listed in the table above.

To evaluate the pretrained models and generate mAP on EPIC-STATES, please see [the EPIC-STATES README](../evaluation/epic-states).

<!-- | Action Class.   | 77.9   | <details><summary>Show</summary>`python train_action.py --name action --abmil`</details> -->

