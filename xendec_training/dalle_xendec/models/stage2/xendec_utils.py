import torch
import torch.nn.functional as F

def _CreateTargetLambdas(atten_probs,
                           source_lambdas_pair,
                           source_paddings_pair,
                           target_paddings_pair,
                           smooth=0):
    """Compute target interpolation ratios.
    Args:
      atten_probs: A list containing two attention matrics.
      source_lambdas_pair: A list containing two source interpolation ratios.
      source_paddings_pair: A list containing two source paddings.
      target_paddings_pair: A list containing two target paddings
      smooth: A real value to smooth target interpolation ratios before
        normalization.
    Returns:
      source_lambdas_pair: Source interpolation ratios.
      input_lambdas: Interpolation ratios for target input embeddings.
      label_lambdas: Interpolation ratios for target labels.
    """
    atten_probs_0 = atten_probs[0].detach().item()
    atten_probs_1 = atten_probs[1].detach().item()

    source_lambdas = source_lambdas_pair[0]
    other_source_lambdas = source_lambdas_pair[1]
    lambdas_0 = atten_probs_0 * torch.unsqueeze(
        source_lambdas * (1.0 - source_paddings_pair[0]), 1)

    lambdas_0 = torch.sum(lambdas_0, -1)
    lambdas_0 = (lambdas_0 + smooth) * (1.0 - target_paddings_pair[0])
    lambdas_1 = atten_probs_1 * torch.unsqueeze(
        other_source_lambdas * (1.0 - source_paddings_pair[1]), 1)
    lambdas_1 = torch.sum(lambdas_1, -1)
    lambdas_1 = (lambdas_1 + smooth) * (1.0 - target_paddings_pair[1])
    label_lambdas_0 = lambdas_0 / (lambdas_0 + lambdas_1 + 1e-9)

    label_lambdas = [label_lambdas_0, (1.0 - label_lambdas_0)]
    input_lambdas_0 = F.pad(
        label_lambdas_0, (1, 0, 0, 0), value=1.)[:, :-1]
    input_lambdas = [
        input_lambdas_0 * (1. - target_paddings_pair[0]),
        (1.0 - input_lambdas_0) * (1. - target_paddings_pair[1])
    ]

    return source_lambdas_pair, input_lambdas, label_lambdas

# def _sequence_mask(lengths, maxlen=None, dtype=torch.bool):
#         if maxlen is None:
#             maxlen = lengths.max()
#         row_vector = torch.arange(0, maxlen, 1)
#         matrix = torch.unsqueeze(lengths, dim=-1)
#         mask = row_vector < matrix

#         mask.type(dtype)
#         return mask
# def _SelectMaskPositions(paddings, ratio=None, sampled_num=None):
#     """Sample ratio * len(sentences) or sampled_num positions from sentences.
#     Args:
#       paddings: a paddings tensor of shape [batch, time].
#       ratio: a sampling ratio of a float or a tensor of shape [batch].
#       sampled_num: a tensor of shape [batch] and will be used when ratio=None.
#     Returns:
#       mask: a mask tensor with 1 as selected positions and 0 as non-selected
#           positions.
#     """
#     shape = paddings.shape
#     z = -torch.math.log(-torch.math.log(torch.random.uniform(shape, 0., 1.)))
#     input_length = torch.sum(1.0 - paddings, 1)
#     input_mask = _sequence_mask(
#         torch.Tensor(input_length - 1, dtype=torch.int32), shape[1], dtype=torch.float32)
#     z = z * input_mask + (1.0 - input_mask) * (-1e9)
#     topk = torch.max(input_length - 1)

#     if sampled_num is not None:
#       topk = torch.maximum(sampled_num, 1)
#       topk = torch.max(topk)
#     elif ratio is not None and isinstance(ratio, float):
#       topk = torch.Tensor(topk * ratio, dtype=torch.int32)
#       topk = torch.maximum(topk, 1)
#       sampled_num = (input_length - 1) * ratio
#     elif ratio is not None and isinstance(ratio, torch.Tensor):
#       topk = torch.Tensor(topk * ratio, dtype=torch.int32)
#       topk = torch.maximum(topk, 1)
#       topk = torch.max(topk)
#       sampled_num = (input_length - 1) * ratio

#     sampled_num = torch.maximum(sampled_num, 1)
#     topk = torch.Tensor(topk, dtype=torch.int32)
#     _, indices = torch.top_k(z, topk)

#     seq_mask = _sequence_mask(
#         torch.Tensor(sampled_num, dtype=torch.int32), topk, dtype=torch.int32)

#     indices = (indices + 1) * seq_mask
#     indices = torch.reshape(indices, [-1])
#     top_id = torch.arange(shape[0] * topk) // topk
#     indices = torch.stack([top_id, indices], axis=1)

#     ######################################
#     # mask = tf.sparse_to_dense(
#     #     indices, (shape[0], shape[1] + 1), 1., 0., validate_indices=False)

#     #make indices into corrdinate frame format
#     mask = torch.sparse_coo_tensor(indices, 1., (shape[0], shape[1] + 1))
#     mask.to_dense()
#     #####################################
#     mask = mask[:, 1:]
#     mask = torch.Tensor(mask, dtype=torch.float32)
#     mask = mask * input_mask
#     return mask