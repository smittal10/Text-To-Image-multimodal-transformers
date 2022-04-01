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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from dataset import CocoDataset
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

parser.add_argument('--n_cores', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--image_root_train',type=str,default="coco/images/train2017")
parser.add_argument('--caption_json_train',type=str,default="coco/annotations/captions_train2017.json")
parser.add_argument('--image_root_valid',type=str,default="coco/images/val2017")
parser.add_argument('--caption_json_valid',type=str,default="coco/annotations/captions_val2017.json")
parser.add_argument('--result_path',type=str,default='saved_models/')
parser.add_argument('--freeze_embeddings',type=bool,default=True)
parser.add_argument('--freeze_layers', type=bool, default=True)


args = parser.parse_args()

class ImageLogger(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module.stage1(images)
            images = images.cpu()
            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=8) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=8) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="test")


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_train,
                 json_train,
                 root_valid,
                 json_valid,
                 tokenizer,
                 image_resolution: int = 256,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8):
        super().__init__()

        self.root_train = root_train
        self.json_train = json_train
        self.root_valid = root_valid
        self.json_valid = json_valid
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.valid_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.CenterCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def setup(self, stage=None):
        self.trainset = CocoDataset(self.root_train,self.json_train,self.tokenizer,self.train_transform)
        self.validset =CocoDataset(self.root_valid,self.json_valid,self.tokenizer,self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)


def setup_callbacks(config):
    # Setup callbacks
    # now = datetime.now().strftime('%d%m%Y_%H%M%S')
    # result_path = os.path.join(args.result_path,
                            #    now)
    # ckpt_path = os.path.join(args.result_path, 'ckpt')
    log_path = os.path.join(args.result_path, 'log')
    ckpt_path = args.result_path

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="dalle-mscoco-tune-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=True,
        save_last=True
    )
    logger = TensorBoardLogger(log_path, name="dalle")
    # logger_img = ImageLogger()
    return checkpoint_callback, logger


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    # Build iGPT
    model, config = Dalle.from_pretrained('1.3B')
    print(model)

    layer_list = model.stage2.blocks
    embed_list = [model.stage2.tok_emb_img, model.stage2.tok_emb_txt, model.stage2.pos_emb_img, model.stage2.pos_emb_txt]

    if args.freeze_embeddings:
        for param in embed_list:
            param.requires_grad = False
            print ("Froze Embedding Layer")
    
    if args.freeze_layers:
        layer_indexes = list(range(0,28))
        for layer_idx in layer_indexes:
            for param in list(layer_list[layer_idx].parameters()):
                param.requires_grad = False
            print ("Froze Layer: ", layer_idx)
    # sys.exit()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("The total number of parameters are")
    print(pytorch_total_params)

    # Setup callbacks
    ckpt_callback, logger = setup_callbacks(config)

    # Build data modules
    dataset = ImageNetDataModule(root_train=args.image_root_train,
                                 json_train=args.caption_json_train,
                                 root_valid=args.image_root_valid,
                                 json_valid=args.caption_json_valid,
                                 tokenizer=model.tokenizer,
                                 image_resolution=config.dataset.image_resolution,
                                 train_batch_size=config.experiment.local_batch_size,
                                 valid_batch_size=config.experiment.valid_batch_size,
                                 num_workers=8)
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.valid_dataloader()
    print(f"len(train_dataset) = {len(dataset.trainset)}")
    print(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * args.n_cores) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * args.n_cores)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs
    print("training batch size: {}".format(config.experiment.local_batch_size))

    # Build trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.experiment.use_amp else 32,
                         gpus=None,
                         callbacks=[ckpt_callback],
                        #  accelerator="gpu",
                        #  devices=args.n_gpus,
                         tpu_cores=8,
                        #  strategy="ddp",
                         logger=logger)
    trainer.fit(model, train_dataloader, valid_dataloader)
        

