import os
import sys

import argparse
import clip
import numpy as np
from PIL import Image
import torch
import pandas as pd
from pathlib import Path

threshold = 0.9

parser = argparse.ArgumentParser()
parser.add_argument('--male_img_dir', type=str)
parser.add_argument('--female_img_dir', type=str)
parser.add_argument('--output_file_name', type=str, default='gender_eval_result')


args = parser.parse_args()

def gender_class(im_path):
    image = preprocess_clip(Image.open(im_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a man", "a woman"]).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(text)

        logits_per_image, logits_per_text = model_clip(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs


image_paths = sorted(list(Path(args.male_img_dir).iterdir()), key=lambda x: os.path.getmtime(x))
male_results = []
for im_path in image_paths:
    probs = gender_class(im_path)
    male_results.append((im_path,probs))
        
image_paths = sorted(list(Path(args.female_img_dir).iterdir()), key=lambda x: os.path.getmtime(x))
female_results = []
for im_path in image_paths:
    probs = gender_class(im_path)
    female_results.append((im_path,probs))


male_labels = []
for row in male_results:
    name = row[0]
    logits = row[1][0]
    
    if logits[0] >=threshold:
        label = 'male'
    elif logits[1] >= threshold:
        label= 'female'
    else:
        label = 'unknown'
    male_labels.append((name,label))
    
female_labels = []
for row in female_results:
    name = row[0]
    logits = row[1][0]
    
    if logits[0] >=threshold:
        label = 'male'
    elif logits[1] >= threshold:
        label= 'female'
    else:
        label = 'unknown'
    female_labels.append((name,label))
    

male_df = pd.DataFrame(male_labels,columns=['img_name','label'])
female_df = pd.DataFrame(female_labels,columns=['img_name','label'])

print('male_df', male_df.groupby('label').count())
male_df.to_csv(f"{args.output_file_name}_male.csv",index=False)

print("female_df",female_df.groupby('label').count())
female_df.to_csv(f"{args.output_file_name}_female.csv",index=False)
