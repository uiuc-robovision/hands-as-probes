# Affordances using Context Prediction

## Prerequisites

Install epic-kitchens-hand-object-bboxes library available [here](https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes).

## Directory Structure

`$GITROOT` is the location where you have cloned the repo, typically of the form `.../epic-affordances`.

### 1. Metadata Generation


<strong>Retrained Detections.</strong> Download from [Box](https://uofi.box.com/s/bwrk54elgefadhiymkq4ohvl82c8mhmr) and extract to `PATH_TO_HANDOBJECTDETECTIONS`.

```bash
cd $GIT_ROOT/ACP/scripts/generate_metadata
# For generating metadata where hands are in contact with the object
python generate_metadata.py --contact --det_path PATH_TO_HANDOBJECTDETECTIONS --out_path $GIT_ROOT/ACP/datasets/

# Metadata without using hand contact state information
# python generate_metadata.py --det_path PATH_TO_HANDOBJECTDETECTIONS --out_path $GIT_ROOT/ACP/datasets/
```

### 2. GUN71 data and Hand Tracks
1. <b> GUN-71 crops </b>: We run the hand detector from Shan et al. on GUN71 dataset. The resulting hand crops can be downloaded from [here](https://uofi.box.com/s/sgwg0otegceiq8vf8zbf9kitev1lhj47).

```bash
tar -xf GUN71-crops.tar.gz -C $GIT_ROOT/ACP/datasets/
```

2. <b> EPIC Hand Tracks </b>: To generate the hand tracks, please run the following commands.

```bash
cd $GIT_ROOT/ACP/scripts/hand_tracks
python hoa_tracks.py --out_dir $GIT_ROOT/ACP/datasets/partitioned_tracks
python generate_annotations.py --inp_dir $GIT_ROOT/ACP/datasets/partitioned_tracks_0.8_128
python split_to_frames.py --inp_dir $GIT_ROOT/ACP/datasets/partitioned_tracks_0.8_128 # Will create output directory with by appending "_split" to INP_DIR
```

### 3. Pretrained Models

Download pretrained models [hand-grasp](https://uofi.box.com/s/l8n2mmsartivfwvuoqv11ak1d79f5gum) and [ACP](https://uofi.box.com/s/e927egov5l0pmxzdjz1m7rhh17158hwn) models, and save them to `$GIT_ROOT/ACP/models/GUN71` and `$GIT_ROOT/ACP/models/EPIC-KITCHENS`, respectively.

This should yield the following directory structure.

```
...
$GIT_ROOT/ACP
├── datasets
|   ├── ACP_metadata
|   |    ├── ACP_training_videos_nosupervision.txt
|   |    ├── train_contact_videos.pkl
|   |    ├── validation_contact_videos.pkl
|   ├── GUN71_crops
|   |    ├── data
|   |    ├── annotations_test_extra.pkl
|   |    ├── annotations_test.pkl
|   |    ├── annotations_train_extra.pkl
|   |    ├── annotations_train.pkl
|   |    ├── annotations_val_extra.pkl
|   |    ├── annotations_val.pkl
|   |    ├── annotations.pkl
|   |    ├── metadatagrasps.pkl
|   ├── partitioned_tracks_0.8_128
|   |    ├── annotations_train.pkl
|   |    ├── annotations_validation.pkl
|   |    ├── EPIC_55_annotations.csv
|   |    ├── EPIC_test_s1_object_video_list.csv
|   |    ├── EPIC_test_s2_object_video_list.csv
|   |    ├── train
|   |    |    ├── hand 
|   |    |    |    ├── P05_01_1023_1045_l.jpg
|   |    |    |    ├── P05_01_13203_13310_l.jpg
|   |    |    |    ├── ...
|   |    ├── validation
|   |    |    ├── hand 
|   |    |    |    ├── P05_07_10437_10474_r.jpg
|   |    |    |    ├── P05_07_1047_1158_l.jpg
|   |    |    |    ├── ...
|   ├── partitioned_tracks_0.8_128_split
|   |    ├── annotations_train.pkl
|   |    ├── annotations_validation.pkl
|   |    ├── EPIC_55_annotations.csv
|   |    ├── EPIC_test_s1_object_video_list.csv
|   |    ├── EPIC_test_s2_object_video_list.csv
|   |    ├── train
|   |    |    ├── hand 
|   |    |    |    ├── P05_01_1023_1045_l
|   |    |    |    ├── P05_01_13203_13310_l
|   |    |    |    ├── ...
|   |    ├── validation
|   |    |    ├── hand 
|   |    |    |    ├── P05_07_10437_10474_r
|   |    |    |    ├── P05_07_1047_1158_l
|   |    |    |    ├── ...
├── demo-gun71
├── hand-grasp
├── ...
├── models
|   ├── EPIC_KITCHENS
|   |    ├── SegNet_hands_seed0_checkpoint_350.pth
|   |    ├── SegNet_hands_seed0_checkpoint_75.pth
|   ├── GUN71
|   |    ├── GUN71_tsc_seed0_checkpoint_27.pth
|   |    ├── resGUN71_seed0_best.pth

```

### 1. \[Optional\] Hand Model

If you have downloaded the pretrained hand grasp model, you can skip and move to step 2.

You can use the hand model to generate hand grasp predictions by following these [instructions](./demo-gun71/README.md).

### Training Command
Update `ACP/hand-grasp/configs/base_tsc.yaml` as follows:
```python
data:
  # ...
  dir: "{GIT_ROOT}/ACP/datasets/GUN71_crops"
  
  EPIC:
    # ...
    EPIC_dir: "{GIT_ROOT}/ACP/datasets/GUN71_crops/partitioned_tracks_0.8_128_split"

```

Then, train the hand_model with temporal simclr (TSC), use the following command
```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=0 \
python hand-grasp/train.py \
    --tsc \
    --config ./hand-grasp/configs/base_tsc.yaml \
    --name GUN71 \
    --seed 0
```

<em> The generated snapshots can be used to train ACP models as well, however selecting snapshots will require validation on the GAO benchmark. Typically snapshots after training 24-30 epochs yields good validation mAP on GAO benchmark. </em>

### 2. Training ACP (Affordances via Context Prediction) model
To train ACP model with just the ROI head (ACP no hand),

Update `ACP/src/configs/nohands.yaml` as follows:
```python
# ...
data:
  # ...
  data_dir: "{PATH_TO_EPIC_DATASET}"
  annot_dir: "{GIT_ROOT}/ACP/datasets/ACP_metadata"

```

Then train using,

```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=GPU_ID \
python src/train.py \
    --config src/configs/nohands.yaml \
    --name SegNet \
    --seed 0
```

To train ACP model with both grasp prediction and ROI heads,

Update `ACP/src/configs/hands.yaml` as follows:
```python
# ...
data:
  # ...
  data_dir: "{PATH_TO_EPIC_DATASET}"
  annot_dir: "{GIT_ROOT}/ACP/datasets/ACP_metadata"

training:
  # ...
  hand_ckpt: "{GIT_ROOT}/ACP/models/GUN71/GUN71_tsc_seed0_checkpoint_27.pth"

```

```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=GPU_ID \
python src/train_hands.py \
    --config src/configs/hands.yaml \
    --name SegNet_hands \
    --seed 0
```

### 3. Inference using ACP (Affordances via Context Prediction) model

To <em>infer</em> using the learned ACP models,
1. Inferring region of intearctions using ACP (no hands) i.e. with just one head

```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=$GPU \
python src/inference_ROI.py \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --hand_cond True \
    --split $SPLIT \
    --ckpt 350 \
    --out_dir $OUT_DIR \
    --two_heads False
```

2. Inferring regions of interaction using ACP with two heads

```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=$GPU \
python src/inference_ROI.py \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --hand_cond True \
    --split $SPLIT \
    --ckpt 350 \
    --out_dir $OUT_DIR \
    --two_heads True
```

#### Other Ablation

| Model           | Result | Command   |
| --------------- | --     | --        |
| ACP             | 61.4   | <details><summary>Show</summary>Config: `src/configs/hands.yaml`</details>
|                 |        |           |
| ACP (no hand hiding)          | 60.8   | <details><summary>Show</summary>Config: `src/configs/hands_nomask.yaml`</details>
| ACP (no contact filtering)    | 59.9   | <details><summary>Show</summary>Config: `src/configs/hands_nocontact.yaml`</details>
| ACP (symmetric context)   | 60.2   |
| ACP (noobject)      | 53.6   | <details><summary>Show</summary>Config: `src/configs/hands_center.yaml`</details>
|                 |        |           |
| ACP (hand segmentation masks as opposed to box-masks)             | 60.8   | <details><summary>Show</summary>Config: `src/configs/hands_nomask.yaml`, Requires Hand Seg Masks!</details>
|                 |        |           |
| ACP (2s x 2s output, loss everywhere)         | 61.5   | <details><summary>Show</summary>Config: `src/configs/hands_symm_nolossmask.yaml`</details>
| ACP (2s x 2s output, loss on bottom center) | 61.7 | <details><summary>Show</summary>Config: `src/configs/hands_symm_wlossmask.yaml`</details>
|                 |        |           |
| ACP (no L_temporal)         | NA   | <details><summary>Show</summary>Config: `src/configs/hands_resGUN71.yaml`</details>


Note: We also provide a 2x faster implementation in `inference_ROI_fast.py` to infer regions of interaction that yield similar performance on EPIC-ROI benchmark.
 
3. Inferring hand predictions (used for evaluation on GAO Benchmark)

```bash
cd $GIT_ROOT/ACP
CUDA_VISIBLE_DEVICES=$GPU \
python src/inference_YCB.py \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --split $SPLIT \
    --ckpt 75 \
    --wsize 80 \
    --out_dir $OUT_DIR \
```
Example: `$MODEL_DIR` -> `./models/EPIC-KITCHENS`; `$MODEL_NAME` -> `SegNet_hands_seed0`; `$SPLIT` -> `val/test`.

### Inferring using a pretrained model

To infer affordances (spatial hand grasp predictions masked with regions of interaction) on EPIC-ROI dataset, run

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
python inference_affordances.py \
    --model_dir ./models/EPIC-KITCHENS \
    --model_name SegNet_hands \
    --ckpt_ROI 350 \
    --ckpt_YCB 75 \
    --two_heads True \
    --out_dir ./inferred_IHdata/epicaff_benchmark \
    --split SPLIT \
    --hand_cond True
```

To infer affordances (spatial hand grasp predictions masked with regions of interaction) on custom image directory (containing `1080x1920 images`), run

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
python inference_affordances.py \
    --model_dir ./models/EPIC-KITCHENS \
    --model_name SegNet_hands \
    --ckpt_ROI 350 \
    --ckpt_YCB 75 \
    --two_heads True \
    --inp_dir $INP_DIR \
    --out_dir ./inferred_IHdata/ACP_demo
```

Note: To visualize the affordances (or extract local patches with maximum confidence scores for each hand grasp), we recommend using the notebooks [here](./visualizations/).