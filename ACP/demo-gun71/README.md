# Demo for obtaining hand grasp given hand detections

Download pretrained models [hand-grasp](https://uofi.box.com/s/l8n2mmsartivfwvuoqv11ak1d79f5gum) and and save them to `$GIT_ROOT/ACP/models/GUN71`. For more info on directory structure refer to [README.md](../README.md).

<em>Note: `$GITROOT` is the location where you have cloned the repo, typically of the form `.../epic-affordances`.</em>

```bash
conda env create --f environment.yml
conda activate humanhands
```

```bash
CUDA_VISIBLE_DEVICES=GPU_ID python demo_handgrasp.py
```

You should get the following output
```
Hand grasp for left hand: Precision Sphere
Hand grasp for right hand: Medium Wrap
```