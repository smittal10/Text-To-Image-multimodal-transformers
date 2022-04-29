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
from perturb_dataset import CocoDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_candidates', type=int, default=1)
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean')
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--caption_json_valid',type=str,default="coco/annotations/captions_val2017.json")
parser.add_argument('--caption_json_valid_bt',type=str,default="coco/annotations/captions_val2017_bt.json")
parser.add_argument('--caption_original_filtered',type=str,default="examples/data/original_captions_filtered.txt")
parser.add_argument('--caption_negated_filtered',type=str,default="examples/data/negated_coco_filtered.txt")


args = parser.parse_args()

def gen_save(captions,tokens,args,model,path_to_save,map_fp,indx):
    B = tokens.shape[0]
    images = model.sampling_batched(tokens,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        softmax_temperature=args.softmax_temperature,
                        num_candidates=args.num_candidates,
                        device=device).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking

    # rank = clip_score_batched(prompt=captions,
    #                 images=images,
    #                 model_clip=model_clip,
    #                 preprocess_clip=preprocess_clip,
    #                 num_candidates=args.num_candidates,
    #                 device=device)

    # Save images
    # print(rank, images.shape)
    # images = images.reshape(B,256,256,3)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    for i in range(images.shape[0]):
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        # fname = str(indx+i)
        fname = ''.join(filter(str.isalnum, captions[i]))
        im.save(f'./{path_to_save}/{fname}.png')
        map_fp.write(fname + '\t' + captions[i] + '\n')

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cuda'
model = Dalle.from_pretrained('1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

#load the dataset in dataloader
coco = CocoDataset(args.caption_json_valid,args.caption_json_valid_bt,model.tokenizer)
valid_loader = DataLoader(dataset=coco,batch_size=150,shuffle=False,num_workers=4)

start = timer()
#model loading
# model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
# model_clip.to(device=device)

#model dir and map file path
root_dir = "./perturbations_fullfinetune_25layers_5e-5_no_reranking"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
model_dir_orig = os.path.join(root_dir,'orig')
model_dir_bt = os.path.join(root_dir,'bt')
fp_orig = open(os.path.join(root_dir,'orig_map.tsv'),"w")
fp_bt = open(os.path.join(root_dir,'bt_map.tsv'),"w")

# Sampling
for global_i, (captions1,captions2,tokens1,tokens2) in enumerate(valid_loader):
    gen_save(captions1,tokens1,args,model,model_dir_orig,fp_orig,global_i)
    gen_save(captions2,tokens2,args,model,model_dir_bt,fp_bt,global_i)
    if global_i%5 == 0:
        fp_orig.flush()
        fp_bt.flush()
fp_orig.close()
fp_bt.close()
end = timer()
print("Total time {}".format(end-start))
