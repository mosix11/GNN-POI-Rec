import torch
import torch.nn as nn

class MRR(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:

        predicted = logits.softmax(dim=-1)
        top_k = predicted.topk(logits.size(-1), dim=-1)[1]
        ranks = (top_k == targets.unsqueeze(-1)).nonzero()[:, -1].float() + 1
        reciprocal_ranks = 1.0 / ranks
        if padding_mask is not None:
            reciprocal_ranks *= padding_mask.float()
            # Avoid division by zero by counting non-zero elements in the mask
            mrr = reciprocal_ranks.sum() / padding_mask.float().sum()
        else:
            mrr = reciprocal_ranks.mean()
        return mrr