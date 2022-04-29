import numpy as np
import torch
from torch import nn
import clip
import yaml
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.coco import CocoCaptions

single_caption = True # choose if evalating only using the first caption
model_name = "ViT-B/32" #"RN50" #"RN50x4" #"RN101" #

def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=False):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        np.save('ranks_bt.npy',ranks,allow_pickle=True)
        np.save('top1_bt.npy',top1,allow_pickle=True)
        return report_dict, (ranks, top1)
    else:
        return report_dict

class MyDataset(Dataset):
    def __init__(self, path_to_img, path_to_orig, image_dir,transform=None):
        self.image_dir = image_dir
        self.original = None
        self.lines = None
        self.transform = transform
        with open(path_to_img,'r') as fp:
            self.lines = fp.readlines()
        with open(path_to_orig,'r') as fp:
            self.original = fp.readlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index) :
        img_path = self.lines[index].strip().split('\t')[0]
        image = Image.open(os.path.join(self.image_dir, img_path +'.png')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        caption = self.original[index].strip().split('\t')[1]
        return image, caption


print(model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load(model_name, device=device)

data_root = "./perturbations_fullfinetune_25layers_5e-5_no_reranking"
valid_dir = os.path.join(data_root, 'orig')
imgs_path = os.path.join(data_root, 'orig_map.tsv')
captions_path = os.path.join(data_root,'orig_map.tsv')
valid_dataset = MyDataset(imgs_path, captions_path, valid_dir, transform = preprocess)
valid_dataloader = DataLoader(valid_dataset, batch_size = 80, shuffle=False, num_workers=4)

# fwd all samples
image_features = []
text_features = []
for batch_idx, batch in enumerate(valid_dataloader):
    print('Evaluating batch {}/{}'.format(batch_idx, len(valid_dataloader)), end = "\r")
    images, texts = batch

    texts = clip.tokenize(texts).to(device=device) #tokenize
    text_emb = model.encode_text(texts) #embed with text encoder
    # if not single_caption:
    #     text_emb = text_emb.unsqueeze(0)

    image_emb = model.encode_image(images.to(device=device)) #embed with image encoder
    

    text_features.append(text_emb.detach().cpu())
    image_features.append(image_emb.detach().cpu())

image_features = torch.cat(image_features, 0)
text_features = torch.cat(text_features, 0)
print('Done forward')

# normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

if not single_caption:
    for cap_idx in range(text_features.shape[1]):
        similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
        i2t_dict = compute_retrieval(similarity_scores.numpy())
        t2i_dict = compute_retrieval(similarity_scores.t().numpy())
        print(cap_idx, 'i2t', i2t_dict)
        print(cap_idx, 't2i', t2i_dict)
else:
    similarity_scores = compute_similarity(image_features, text_features)
    i2t_dict = compute_retrieval(similarity_scores.numpy())
    t2i_dict = compute_retrieval(similarity_scores.t().numpy())
    print('i2t', i2t_dict)
    print('t2i', t2i_dict)

