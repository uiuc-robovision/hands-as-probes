import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from html4vision import Col, imagetable

sys.path.append(str(Path(__file__).absolute().parent.parent))
from data.data_utils import get_splits


# dirs = ["/data01/smodi9/datasets/EPIC-KITCHENS/tracks/srpn_maskrcnn/images/", "/data01/smodi9/datasets/EPIC-KITCHENS/tracks/srpn/images/"]
# names = ["SRPN BG MaskRCNN", "SRPN FG"]

dirs = ["/data01/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned/ioumf/images/", "/data01/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/"]
names = ["object tracks", "hand tracks"]

def save_grid(track_path, output_path):
    if output_path.exists():
        return
    width = 128
    height = 128
    track = np.array(Image.open(track_path))
    images = [Image.fromarray(track[:, i*128:(i+1)*128]) for i in range(track.shape[1]//width)]

    # output_path = Path(output_path)
    # if len(images) < 10 and output_path.stem[-2:] != '_2':
    #     new_out = output_path.parent / 'pdfs' / f'{output_path.stem}.pdf'
    #     new_out.parent.mkdir(exist_ok=True)
    #     Image.fromarray(track).save(new_out)

    rows = cols = int(np.ceil(np.sqrt(len(images))))
    new_im = Image.new(images[0].mode, (width*cols, height*rows))
    for imnum, im in enumerate(images):
            i, j = imnum % cols, imnum // cols
            new_im.paste(im, (i*width, j*height))
    new_im.save(output_path)
    

# np.random.seed(0)
files = glob.glob(dirs[0] + "*.jpg", recursive=True)
files = [f for f in files if Path(f).stem.split("_")[0] in get_splits("validation")]
sample = [Path(x) for x in np.random.choice(files, replace=False, size=(100,))]
if '_hand' in dirs[1]:
    sample2 = sample
else:
    files2 = glob.glob(dirs[1] + "*.jpg", recursive=True)
    files2 = [f for f in files2 if Path(f).stem.split("_")[0] in get_splits("validation")]
    sample2 = [Path(x) for x in np.random.choice(files2, replace=False, size=(100,))]

output_dir = Path(__file__).parent
image_cache = output_dir / "cache"
image_cache.mkdir(exist_ok=True)

paths = []
for object_path, object_path2 in tqdm(zip(sample, sample2), total=len(sample)):
    path1 = image_cache / f"{object_path.stem}_1.jpg"
    path2 = image_cache / f"{object_path.stem}_2.jpg"
    save_grid(object_path, path1)
    if '_hand' in dirs[1]:
        save_grid(str(object_path).replace("/tracks_partitioned/", "/tracks_partitioned_hand/"), path2)
    else:
        save_grid(object_path2, path2)
    paths.append((path1, path2))

cols = [Col('id1', 'ID')]
cols.append(Col("img", names[0], [o.relative_to(output_dir) for o,h in paths]))
cols.append(Col("img", names[1], [h.relative_to(output_dir) for o,h in paths]))
imagetable(cols, output_dir / 'vis_tracks.html', f'{names[0]} v. {names[1]}',
    # imscale=1.0,                # scale images to 0.4 of the original size
    imsize=(1024, 1024),
    # sortcol=0,                  #
    sortable=True,              # enable interactive sorting
    sticky_header=True,         # keep the header on the top
    sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
    zebra=True,                 # use zebra-striped table
)