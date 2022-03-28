from html4vision import Col, imagetable
from PIL import Image
import pickle as pkl
import numpy as np
import argparse, sys, os
import glob
from pathlib import Path


def make_html(args):
    path = f"./{args.res_dir}/{args.name}/{args.name}_{args.split}.pkl"
    with open(path, "rb") as f:
        data = pkl.load(f)

    # Use only validation participants
    color_images = [f"./images/{p}" for p in data['fname']]

    prob_scores = {}
    for i in [0, 2, 10, 11, 21, 25, 27]:
        prob_scores[i] = []
        for j in data['pred_tax']:
            prob_scores[i].append(j[i])

    gt_tax = [','.join(str(x) for x in sorted(tax)) for tax in data['gt_tax']]

    cols = [
        Col('id1', 'ID'),
        Col('img', 'images', color_images),
    ]
    for i, j in prob_scores.items():
        cols.append(Col('text', str(i+1), j))
    cols.append(Col('text', 'GT', gt_tax))

    imagetable(cols, f'{args.res_dir}/{args.name}/vis_{args.split}.html', 'Hand Grasp Vis',
               imscale=1.0,  # scale images to 0.4 of the original size
               sortcol=2,  # initially sort based on column 2 (class average performance)
               sortable=True,  # enable interactive sorting
               sticky_header=True,  # keep the header on the top
               sort_style='materialize',  # use the theme "materialize" from jquery.tablesorter
               )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='html creation arguments')
    parser.add_argument('--name', dest='name', type=str, default="SegNet_hands_nosupckpt27_seed0_75")
    parser.add_argument('--res_dir', dest='res_dir', type=str, default="results_html")
    parser.add_argument('--split', dest='split', type=str, default="val")
    args = parser.parse_args()
    make_html(args)

