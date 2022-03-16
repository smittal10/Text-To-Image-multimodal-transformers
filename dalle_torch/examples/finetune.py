# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
from typing import Optional
from datetime import datetime

import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, Callback
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.distributed import rank_zero_only

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import CocoDataset

parser = argparse.ArgumentParser()

# parser.add_argument('-d', '--config-downstream', type=str, default=None, required=True)
# parser.add_argument('-u', '--path-upstream', type=str, default=None, required=True)
# parser.add_argument('-r', '--result-path', type=str, default=None, required=True)
# parser.add_argument('--imagenet-path', type=str, default=None, required=True)

parser.add_argument('--n-gpus', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rank',type=int,default=0)
parser.add_argument('--world_size',type=int,default=4)
parser.add_argument('--image_root_train',type=str,default="coco/images/train2017")
parser.add_argument('--caption_json_train',type=str,default="coco/annotations/captions_train2017.json")
parser.add_argument('--image_root_valid',type=str,default="coco/images/val2017")
parser.add_argument('--caption_json_valid',type=str,default="coco/annotations/captions_val2017.json")
parser.add_argument('--image_root_test',type=str,default="'coco/images/test2017")
parser.add_argument('--caption_json_test',type=str,default="coco/annotations/image_info_test2017.json")
parser.add_argument('--ckpt_save_path',type=str,default='saved_models/GPU')


args = parser.parse_args()

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, root, json, batch_size, shuffle, tokenizer,pin_memory=False, num_workers=8):
    coco = CocoDataset(root,
                       json,
                       tokenizer)
    sampler = DistributedSampler(coco, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(coco, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler)
    
    return dataloader

def train_one_epoch(train_loader,model,criterion,optimizer,grad_accm_steps):
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for i, (image,captions) in enumerate(train_loader):
        # image shape is [B,C,H,W] and shape of captions [B,64]
        image, captions = image.to(device),captions.to(device)
        logits_img, logits_txt, codes =  model(image,captions)
        loss_txt = criterion(logits_txt.view(-1, logits_txt.shape[-1]),captions[:,1:].reshape(-1))
        loss_img = criterion(logits_img.view(-1, logits_img.shape[-1]),codes.view(-1))
        loss = loss_txt/8 + 7*loss_img/8
        loss = loss / grad_accm_steps
        loss.backward()
        train_loss += loss.item()
        if (i+1) % grad_accm_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=model.config.optimizer.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
    return train_loss / (len(train_loader))

def validate(valid_loader,model,criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():

        for image,captions in valid_loader:
            image, captions = image.to(device),captions.to(device)
            # image shape is [B,C,H,W] and shape of captions [B,64]
            logits_img, logits_txt, codes =  model(image,captions)
            loss_txt = criterion(logits_txt.view(-1, logits_txt.shape[-1]),captions[:,1:].reshape(-1))
            loss_img = criterion(logits_img.view(-1, logits_img.shape[-1]),codes.view(-1))
            loss = loss_txt/8 + 7*loss_img/8
            val_loss += loss.item()
    return val_loss/ len(valid_dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    #DDP
    # setup the process groups
    #rank = 0/1/2/3, world_size = 4

    # setup(args.rank, args.world_size)
    #Build dalle and load from checkpoint
    model = Dalle.from_pretrained('1.3B')  # This will automatically download the pretrained model.
    print("Loaded model in memory")
    print(model)
    # prepare the dataloader
    train_dataloader = prepare(args.rank, args.world_size, args.image_root_train, args.caption_json_train, model.config.experiment.local_batch_size, True, model.tokenizer,pin_memory=False, num_workers=16)
    valid_dataloader = prepare(args.rank, args.world_size, args.image_root_valid, args.caption_json_valid, model.config.experiment.valid_batch_size, False, model.tokenizer,pin_memory=False, num_workers=16)


    model.to(device)

    # model = DDP(model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.token_to_id('[PAD]'))
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=model.config.optimizer.base_lr,
                                betas=model.config.optimizer.betas,
                                weight_decay=model.config.optimizer.weight_decay)
    
    # Calculate how many batches are accumulated
    assert model.config.experiment.total_batch_size % (model.config.experiment.local_batch_size * args.world_size) == 0
    grad_accm_steps = model.config.experiment.total_batch_size // (model.config.experiment.local_batch_size * args.world_size)
    # config.optimizer.max_steps = len(dataset.trainset) // model.config.experiment.total_batch_size * model.config.experiment.epochs
    best_valid_loss = 10000000
    for epoch in range(model.config.experiment.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(train_dataloader,model,criterion,optimizer,grad_accm_steps)
        valid_loss = validate(valid_dataloader,model,criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save the model if valid_loss is better
        if valid_loss < best_valid_loss and args.rank == 0:
            best_valid_loss = valid_loss
            torch.save( model.state_dict(), f'{args.ckpt_save_path}/coco_gpu.ckpt')


        print(f"Epoch: {epoch+1:02} "
                f"| Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} ")
        print(f"\t Val. Loss: {valid_loss:.3f} ")
        

