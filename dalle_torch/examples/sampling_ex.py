# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
import clip
import numpy as np
from timeit import default_timer as timer
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_candidates', type=int, default=32)
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean')
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cuda'
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)
start = timer()
#model loading
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.to(device=device)
# read coco captions
prompts= []
with open("examples/data/negated_coco_objects_100.txt","r") as f:
    for line in f.readlines():
        prompts.append(line.strip())
# Sampling
for prompt in prompts:
    images = model.sampling(prompt=prompt,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        softmax_temperature=args.softmax_temperature,
                        num_candidates=args.num_candidates,
                        device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking

    rank = clip_score(prompt=prompt,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    device=device)

    # Save images
    images = images[rank]
    print(rank, images.shape)
    if not os.path.exists('./negated_coco_figures'):
        os.makedirs('./negated_coco_figures')
    for i in range(min(5, args.num_candidates)):
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        im.save(f'./negated_coco_figures/{prompt}_{i}.png')


end = timer()
print("Total time {}".format(end-start))
