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
from dalle.utils.utils import set_seed, clip_score_batched
from gen_dataset import CocoDataset,get_loader


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_candidates', type=int, default=32)
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean')
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--caption_json_valid',type=str,default="coco/annotations/captions_val2017.json")

args = parser.parse_args()

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cpu'
model,_ = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

#load the dataset in dataloader
valid_loader = get_loader(args.caption_json_valid,
                                model.tokenizer,
                                batch_size=4,
                                shuffle=False,
                                num_workers=4)

start = timer()
#model loading
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.to(device=device)
# read coco captions
# prompts= []
# with open("examples/data/negated_coco_objects_100.txt","r") as f:
#     for line in f.readlines():
#         prompts.append(line.strip())
# Sampling
for captions,tokens in valid_loader:
    images = model.sampling_batched(tokens,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        softmax_temperature=args.softmax_temperature,
                        num_candidates=args.num_candidates,
                        device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking

    rank = clip_score_batched(prompt=captions,
                    images=images,
                    model_clip=model_clip,
                    preprocess_clip=preprocess_clip,
                    num_candidates=args.num_candidates,
                    device=device)

    # Save images
    print(rank, images.shape)
    images = images.reshape(-1,args.num_candidates,256,256)
    if not os.path.exists('./generated_coco_figures'):
        os.makedirs('./generated_coco_figures')
    for i in range(rank.shape[0]):
        best_image_index = rank[i,0]
        im = Image.fromarray((images[i,best_image_index]*255).astype(np.uint8))
        im.save(f'./generated_coco_figures/{captions[i]}.png')


end = timer()
print("Total time {}".format(end-start))
