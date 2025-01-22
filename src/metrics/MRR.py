import torch
import torch.nn as nn


class MRR(nn.Module):
    """
    Mean Reciprocal Rank (MRR):
    For each query, the reciprocal rank is the inverse of the rank of the first correct answer.
    MRR = (1/N_test) * Σ (1/rank_i), where rank_i is the rank of the true label in the sorted predictions.

    inputs:
        - `logits` a tensor of shape [N_test, num_classes], containing model scores for each class.
        - `targets` a tensor of shape [N_test], containing the ground truth integer labels.
    outputs:
        - Mean Reciprocal Rank (MRR) as a scalar tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(targets.shape) == 2:
            targets = targets.squeeze(1)

        # get the rank of each class for all queries, sorted by descending logits
        ranked_indices = logits.argsort(dim=-1, descending=True)
        # get the rank position of the correct labels for each query
        target_ranks = (ranked_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[
            1
        ] + 1

        reciprocal_ranks = 1.0 / target_ranks.float()
        mrr = reciprocal_ranks.mean()
        return mrr
