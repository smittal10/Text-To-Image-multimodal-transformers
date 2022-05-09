import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, tokenizer, transform):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.tokenizer = tokenizer
        self.pad_id=tokenizer.token_to_id('[PAD]')

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to BPE tokens.
        tokens = self.tokenizer.encode(caption)
        tokens = torch.LongTensor(tokens.ids)
        return image, tokens

    def __len__(self):
        return len(self.ids)
    
def collate_fn_double_input(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    
    images, captions = zip(*data)
    batch_size_half = int(len(images)/2)
    input_batch = {}
    other_batch = {}
    input_batch['tgt'], input_batch['src'] = torch.stack(images[:batch_size_half]), torch.stack(captions[:batch_size_half])
    other_batch['tgt'], other_batch['src'] = torch.stack(images[batch_size_half:]), torch.stack(captions[batch_size_half:])

    B, _, _, N  = input_batch['tgt'].shape
    input_batch['src_paddings'] = torch.where(input_batch['src']==0,1.,0.)
    input_batch['tgt_paddings'] = torch.zeros((B,N), dtype=torch.float32)
    other_batch['src_paddings'] = torch.where(other_batch['src']==0,1.,0.)
    other_batch['tgt_paddings'] = torch.zeros((B,N), dtype=torch.float32)

    #obtain source masks for lambdas
    input_batch['source_mask'] = _CreateSourceLambdas(input_batch['src_paddings'])
    
    # return input_batch, other_batch
    return input_batch['src'],input_batch['tgt'],other_batch['src'],other_batch['tgt'], input_batch['source_mask'], input_batch['src_paddings'],input_batch['tgt_paddings'],other_batch['src_paddings'],other_batch['tgt_paddings']

def _CreateSourceLambdas(source_paddings,source_mask_ratio=0.5):
    """Generate a 0/1 tensor where 1 takes up a percentage for each row."""
    # p = self.params
    # if source_mask_ratio > 0.:
    #     ratio = torch.Tensor(source_mask_ratio, dtype=torch.float32)
    ratio = source_mask_ratio
    # else:
    #   source_mask_ratio_beta = [
    #       float(a) for a in p.source_mask_ratio_beta.split(',')
    #   ]

    #   beta_dist = tf.distributions.Beta(source_mask_ratio_beta[0],
    #                                      source_mask_ratio_beta[1])
    #   ratio = beta_dist.sample(source_paddings.shape[0])
    #   ratio = torch.minimum(ratio, 1.0 - ratio)
    source_mask = _SelectMaskPositions(source_paddings, ratio)
    return source_mask

def _SelectMaskPositions(paddings, ratio=None, sampled_num=None):
    """Sample ratio * len(sentences) or sampled_num positions from sentences.
    Args:
    paddings: a paddings tensor of shape [batch, time].
    ratio: a sampling ratio of a float or a tensor of shape [batch].
    sampled_num: a tensor of shape [batch] and will be used when ratio=None.
    Returns:
    mask: a mask tensor with 1 as selected positions and 0 as non-selected
        positions.
    """
    shape = paddings.shape
    z = -torch.log(-torch.log(torch.rand(shape)))
    input_length = torch.sum(1.0 - paddings, 1)
    input_mask = _sequence_mask((input_length - 1).to(dtype=torch.int32), shape[1], dtype=torch.float32)
    z = z * input_mask + (1.0 - input_mask) * (-1e9)
    topk = torch.max(input_length - 1)

    if sampled_num is not None:
        topk = torch.maximum(sampled_num, 1)
        topk = torch.max(topk)
    elif ratio is not None and isinstance(ratio, float):
        # topk = torch.Tensor(topk * ratio, dtype=torch.int32)
        topk = topk * ratio
        topk = torch.maximum(topk, torch.Tensor([1.]))
        sampled_num = (input_length - 1) * ratio
    elif ratio is not None and isinstance(ratio, torch.Tensor):
        topk = torch.Tensor(topk * ratio, dtype=torch.int32)
        topk = torch.maximum(topk, 1)
        topk = torch.max(topk)
        sampled_num = (input_length - 1) * ratio

    sampled_num = torch.maximum(sampled_num, torch.Tensor([1.]))
    topk = topk.to(dtype=torch.int32)
    _, indices = torch.topk(z, topk.item(),sorted=False)

    seq_mask = _sequence_mask(sampled_num.to(dtype=torch.int32), topk.item(), dtype=torch.int32)

    indices = (indices + 1) * seq_mask
    indices = torch.reshape(indices, [-1])
    top_id = torch.div(torch.arange(shape[0] * topk.item()), topk.item(), rounding_mode='floor')
    indices = torch.stack([top_id, indices], axis=0)

    ######################################
    # mask = tf.sparse_to_dense(
    #     indices, (shape[0], shape[1] + 1), 1., 0., validate_indices=False)

    #make indices into corrdinate frame format--- indices.shape torch.Size([28, 2])---shape torch.Size([4, 64])
    mask = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], dtype=torch.float32), (shape[0], shape[1] + 1))
    mask = mask.to_dense()
    #####################################
    mask = mask[:, 1:]
    # mask = torch.Tensor(mask, dtype=torch.float32)
    mask = mask * input_mask
    return mask

def _sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask = mask.to(dtype=dtype)
    return mask

# def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, val=False):
#     """Returns torch.utils.data.DataLoader for custom coco dataset."""
#     # COCO caption dataset
#     coco = CocoDataset(root=root,
#                     json=json,
#                     vocab=vocab,
#                     transform=transform)

# # Data loader for COCO dataset
# # This will return (images, captions, lengths) for each iteration.
# # images: a tensor of shape (batch_size, 3, 224, 224).
# # captions: a tensor of shape (batch_size, padded_length).
# # lengths: a list indicating valid length for each caption. length is (batch_size).
# if val:
#     data_loader = torch.utils.data.DataLoader(dataset=coco, 
#                                           batch_size=batch_size,
#                                           shuffle=shuffle,
#                                           num_workers=num_workers)
# else:
#     data_loader = torch.utils.data.DataLoader(dataset=coco, 
#                                           batch_size=batch_size,
#                                           shuffle=shuffle,
#                                           num_workers=num_workers,
#                                           collate_fn=coco.collate_fn)

# return data_loader