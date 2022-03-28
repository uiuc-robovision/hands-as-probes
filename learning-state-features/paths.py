import os
import yaml
from pathlib import Path

root = Path(__file__).parent.absolute() / 'datasets'

# Change me
paths = {
    'dataset_dirs': {
        'epic_kitchens_rgb_frames': root / "EPIC-KITCHENS",
        'epic_kitchens_shan_detections': root / "detections_partitioned",
        'mit_states_dataset': root / "mit_states",
        'tracks': root / "tracks_partitioned",
    }
}

# Do not modify code below this line
DIR_RGB_FRAMES = Path(paths['dataset_dirs'].get('epic_kitchens_rgb_frames', ''))
DIR_SHAN_DETECTIONS = Path(paths['dataset_dirs'].get('epic_kitchens_shan_detections', ''))
DIR_MIT_STATES = Path(paths['dataset_dirs'].get('mit_states_dataset', ''))
DIR_TRACKS = Path(paths['dataset_dirs'].get('tracks', ''))
DIR_TRACKS_HAND = Path(f'{DIR_TRACKS}_hand')

DIR_ANNOTATIONS = Path(__file__).absolute().parent / 'lib' / 'epic_kitchens_annotations'
