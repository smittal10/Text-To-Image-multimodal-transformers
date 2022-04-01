# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from .stage1.vqgan import VQGAN
from .stage2.transformer import Transformer1d, iGPT
from .. import utils
from ..utils.config import get_base_config
from ..utils.sampling import sampling, sampling_igpt
from .tokenizer import build_tokenizer
import clip
import torch_xla.core.xla_model as xm
from torchvision import transforms
_MODELS = {
    'minDALL-E/1.3B': 'https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz'
}
import torch_xla.debug.metrics as met


class Dalle(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.tokenizer = None
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = Transformer1d(vocab_size_txt=config.stage2.vocab_size_txt,
                                    vocab_size_img=config.stage2.vocab_size_img,
                                    hparams=config.stage2.hparams)
        self.config_stage1 = config.stage1
        self.config_stage2 = config.stage2
        self.config_dataset = config.dataset
        self.config = config
        for p in self.stage1.parameters():
            p.requires_grad = False

        # self.resize = transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None)
        self.preproc_image = torch.nn.Sequential(
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None),
            transforms.CenterCrop((224,224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            )
        
        # self.center_crop = transforms.CenterCrop((224,224))
        self.clip_model, _preprocess = clip.load("ViT-B/32")
        self.cosine_loss = nn.CosineEmbeddingLoss()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # self.target = torch.ones(1)
        self.register_buffer("target", torch.ones(1))

    @classmethod
    def from_pretrained(cls,
                        path: str) -> nn.Module:
        path = _MODELS[path] if path in _MODELS else path
        path = utils.realpath_url_or_path(path, root=os.path.expanduser("~/.cache/minDALL-E"))

        config_base = get_base_config(False)
        config_new = OmegaConf.load(os.path.join(path, 'config.yaml'))
        config_update = OmegaConf.merge(config_base, config_new)

        model = cls(config_update)
        model.tokenizer = build_tokenizer(os.path.join(path, 'tokenizer'),
                                          context_length=model.config_dataset.context_length,
                                          lowercase=True,
                                          dropout=None)
        model.stage1.from_ckpt(os.path.join(path, 'stage1_last.ckpt'))
        model.stage2.from_ckpt(os.path.join(path, 'stage2_last.ckpt'))
        return model, config_update

    @torch.no_grad()
    def sampling(self,
                 prompt: str,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        tokens = self.tokenizer.encode(prompt)
        tokens = torch.LongTensor(tokens.ids)
        tokens = torch.repeat_interleave(tokens.unsqueeze(0), num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels
    
    #batched sampling
    @torch.no_grad()
    def sampling_batched(self,
                 tokens,
                 top_k: int = 256,
                 top_p: Optional[float] = None,
                 softmax_temperature: float = 1.0,
                 num_candidates: int = 96,
                 device: str = 'cuda:0',
                 use_fp16: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()
        tokens = torch.repeat_interleave(tokens, num_candidates, dim=0)

        # Check if the encoding works as intended
        # print(self.tokenizer.decode_batch(tokens.tolist(), skip_special_tokens=True)[0])

        tokens = tokens.to(device)
        codes = sampling(self.stage2,
                         tokens,
                         top_k=top_k,
                         top_p=top_p,
                         softmax_temperature=softmax_temperature,
                         use_fp16=use_fp16)
        B = tokens.shape[0]*num_candidates
        codes = codes.view(B, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    #########################
    #insert for dalle training
    def forward(self,
                images: torch.FloatTensor,
                texts: torch.FloatTensor) -> torch.FloatTensor:
        B,N = texts.shape
        device = texts.device
        self.stage1.eval()
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
        _,M = codes.shape
        pos_encoding_txt = torch.arange(N, device=device).repeat((B, 1))
        pos_encoding_img = torch.arange(M, device=device).repeat((B, 1))
        logits_img, logists_txt = self.stage2(codes, texts, pos_encoding_img, pos_encoding_txt )
        return logits_img, logists_txt, codes
    def training_step(self, batch, batch_idx):
        images, captions = batch
        logits_img, logits_txt, codes = self(images, captions)
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]),captions[:,1:].reshape(-1),ignore_index=self.tokenizer.token_to_id('[PAD]'))
        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]),codes.view(-1),ignore_index=self.tokenizer.token_to_id('[PAD]'))
        loss = loss_txt/8 + 7*loss_img/8

        #generate image
        codes = codes.view(-1, 16, 16)
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)
        pixels = self.preproc_image(pixels)
        texts  = self.tokenizer.decode_batch(captions.tolist())
        text_tokens = clip.tokenize(texts).to(self.device)
        # print(f"pixels shape: {pixels.shape}")

        image_features = self.clip_model.encode_image(pixels)
        text_features = self.clip_model.encode_text(text_tokens)
        # target = torch.ones(1)

        cosine_sim_loss = self.cosine_loss(image_features, text_features, self.target)

        loss = loss + cosine_sim_loss

        # print(f'texts: {len(texts)},{texts}')
        # print(f"images: {images.shape}")
        # print(f"captions: {captions.shape}")
        # print(f"codes: {codes.shape}")
        # print(met.metrics_report())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        logits_img, logits_txt, codes = self(images, captions)
        loss_txt = F.cross_entropy(logits_txt.view(-1, logits_txt.shape[-1]),captions[:,1:].reshape(-1))
        loss_img = F.cross_entropy(logits_img.view(-1, logits_img.shape[-1]),codes.view(-1))
        loss = loss_txt/8 + 7*loss_img/8

        #generate image
        codes = codes.reshape(-1, 16, 16)
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)
        # xm.master_print('before preproc')
        # xm.master_print(f"pixels shape: {pixels.shape}")
        # xm.master_print(f"pixels type: {pixels.dtype}")
        # xm.master_print(f"pixels device: {pixels.get_device()}")
        # xm.master_print(f"pixels: {pixels.reshape(-1)}")
        # xm.master_print('before preproc')
        # pixels = self.resize(pixels)
        # xm.master_print('after resize-before crop')
        # pixels = self.center_crop(pixels)
        # xm.master_print('after resize')

        pixels = self.preproc_image(pixels)
        print(f"Is the error after pixels preproc")

        texts  = self.tokenizer.decode_batch(captions.tolist())
        print(f"texts")
        

        text_tokens = clip.tokenize(texts).to(self.device)

        print(f"Is the error after creating text tokens?")

        # xm.master_print(f"pixels shape: {pixels.shape}")
        # xm.master_print(f'texts: {len(texts)}')

        image_features = self.clip_model.encode_image(pixels)
        text_features = self.clip_model.encode_text(text_tokens)
        print(f"Is the error after creating text feature?")

        cosine_sim_loss = self.cosine_loss(image_features, text_features, self.target)

        loss = loss + cosine_sim_loss
        # xm.master_print(met.metrics_report())
        # exit()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        sched = CosineAnnealingLR(opt,
                                  T_max=self.config.optimizer.max_steps,
                                  eta_min=self.config.optimizer.min_lr)
        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
    #                    on_tpu=True, using_native_amp=False, using_lbfgs=False):
    #     optimizer.step(closure=optimizer_closure)
    #     self.lr_schedulers().step()
    #     self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()

class ImageGPT(pl.LightningModule):
    def __init__(self,
                 config: OmegaConf) -> None:
        super().__init__()
        self.stage1 = VQGAN(n_embed=config.stage1.n_embed,
                            embed_dim=config.stage1.embed_dim,
                            hparams=config.stage1.hparams)
        self.stage2 = iGPT(vocab_size_img=config.stage2.vocab_size_img,
                           use_cls_cond=config.stage2.use_cls_cond,
                           hparams=config.stage2.hparams)
        self.config = config
        self.use_cls_cond = config.stage2.use_cls_cond

        # make the parameters in stage 1 not trainable
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls,
                        path_upstream: str,
                        path_downstream: str) -> Tuple[nn.Module, OmegaConf]:
        config_base = get_base_config(use_default=False)
        config_down = OmegaConf.load(path_downstream)
        config_down = OmegaConf.merge(config_base, config_down)

        model = cls(config_down)
        model.stage1.from_ckpt(os.path.join(path_upstream, 'stage1_last.ckpt'), strict=True)
        model.stage2.from_ckpt(os.path.join(path_upstream, 'stage2_last.ckpt'), strict=False)
        return model, config_down

    def sample(self,
               cls_idx: Optional[int] = None,
               top_k: int = 256,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               num_candidates: int = 16,
               device: str = 'cuda:0',
               use_fp16: bool = True,
               is_tqdm: bool = True) -> torch.FloatTensor:
        self.stage1.eval()
        self.stage2.eval()

        if cls_idx is None:
            sos = self.stage2.sos.repeat(num_candidates, 1, 1)
        else:
            sos = torch.LongTensor([cls_idx]).to(device=device)
            sos = sos.repeat(num_candidates)
            sos = self.stage2.sos(sos).unsqueeze(1)

        codes = sampling_igpt(self.stage2,
                              sos=sos,
                              top_k=top_k,
                              top_p=top_p,
                              softmax_temperature=softmax_temperature,
                              use_fp16=use_fp16,
                              is_tqdm=is_tqdm)
        codes = codes.view(num_candidates, 16, 16)  # [B, 16, 16]
        pixels = torch.clamp(self.stage1.decode_code(codes) * 0.5 + 0.5, 0, 1)  # [B, 256, 256]
        return pixels

    def forward(self,
                images: torch.FloatTensor,
                labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        B, C, H, W = images.shape
        with torch.no_grad():
            with autocast(enabled=False):
                codes = self.stage1.get_codes(images).detach()
        logits = self.stage2(codes, labels)
        return logits, codes

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, codes = self(images, labels=labels if self.use_cls_cond else None)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), codes.view(-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        assert self.config.optimizer.opt_type == 'adamW'
        assert self.config.optimizer.sched_type == 'cosine'

        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.config.optimizer.base_lr,
                                betas=self.config.optimizer.betas,
                                weight_decay=self.config.optimizer.weight_decay)
        sched = CosineAnnealingLR(opt,
                                  T_max=self.config.optimizer.max_steps,
                                  eta_min=self.config.optimizer.min_lr)
        sched = {
            'scheduler': sched,
            'name': 'cosine'
        }
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_start(self):
        self.stage1.eval()
