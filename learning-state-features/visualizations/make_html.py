from collections import defaultdict
from pathlib import Path
import os
from PIL import Image

from html4vision import Col, imagetable

d = "/data01/smodi9/VOS/vis"
os.chdir(d)

for dir in os.listdir("./"):
    if ".py" in dir:
        continue
    dir = Path(dir)
    sample_names = os.listdir(dir)
    data = defaultdict(list)
    for s in sample_names:
        if '.html' in s:
            continue
        for exp in os.listdir(dir / s):
            if exp == 'query.jpg':
                continue
            data['qname'].append(s)
            data['query'].append(dir / s / "query.jpg")
            data['name'].append(Path(exp).stem)
            data['nn'].append(dir / s / exp)
    
    cols = []
    cols.append(Col("text", f"Query Name", data['qname']))
    cols.append(Col("text", f"Model Name", data['name']))
    cols.append(Col("img", f"Query Image", [str(p.relative_to(dir)) for p in data['query']]))
    cols.append(Col("img", f"Nearest Neighbors", [str(p.relative_to(dir)) for p in data['nn']]))

    imagetable(cols, dir / f'compare.html', f'{dir} Compare Samples',
                # imscale=1.0,                # scale images to 0.4 of the original size
                # imsize=(128, 128),
                # sortcol=0,                  #
                sortable=True,              # enable interactive sorting
                sticky_header=True,         # keep the header on the top
                sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
                zebra=True,                 # use zebra-striped table
    )