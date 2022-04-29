import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import random
import json
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, orig, negations, tokenizer):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.orig = []
        self.ng = []
        with open(orig,'r') as fp:
            for line in fp.readlines():
                self.orig.append(line.strip())
        with open(negations,'r') as fp:
            for line in fp.readlines():
                self.ng.append(line.strip())
        # self.ids = list(self.coco.imgToAnns.keys())
        print("Total number of captions is {}".format(len(self.orig)))
        print("Total number of captions is {}".format(len(self.ng)))
        #shuffle and select top 5000 ids
        # random.shuffle(self.ids)
        # self.ids = self.ids[:5000]
        # print("Total number of captions is {}".format(len(self.ids)))
        # with open('ids.json', 'w') as f:
        #     json.dump(self.ids, f)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption1 = self.orig[index]
        caption2 = self.ng[index]

        # Convert caption (string) to BPE tokens.
        tokens1 = self.tokenizer.encode(caption1)
        tokens1 = torch.LongTensor(tokens1.ids)

        tokens2 = self.tokenizer.encode(caption2)
        tokens2 = torch.LongTensor(tokens2.ids)
        
        return caption1,caption2,tokens1,tokens2

    def __len__(self):
        return len(self.orig)

# def collate_fn(data):
#     """Creates mini-batch tensors from the list of tuples (image, caption).
    
#     We should build custom collate_fn rather than using default collate_fn, 
#     because merging caption (including padding) is not supported in default.

#     Args:
#         data: list of tuple (image, caption). 
#             - image: torch tensor of shape (3, 256, 256).
#             - caption: torch tensor of shape (?); variable length.

#     Returns:
#         images: torch tensor of shape (batch_size, 3, 256, 256).
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions = zip(*data)

#     # Merge images (from tuple of 3D tensor to 4D tensor).
#     images = torch.stack(images, 0)

#     # Merge captions (from tuple of 1D tensor to 2D tensor).
#     lengths = [len(cap) for cap in captions]
#     targets = torch.zeros(len(captions), max(lengths)).long()
#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]        
#     return images, targets, lengths