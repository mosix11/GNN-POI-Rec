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
        target_ranks = (ranked_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        
        reciprocal_ranks = 1.0 / target_ranks.float()
        mrr = reciprocal_ranks.mean()
        return mrr
    
    
    
# def mean_reciprocal_rank(logits: torch.Tensor, targets: torch.Tensor) -> float:
#     """
#     Compute the Mean Reciprocal Rank (MRR) given logits and ground-truth targets.
    
#     Parameters:
#     -----------
#     logits  : torch.Tensor 
#         A tensor of shape (batch_size, num_classes) representing model outputs
#         (unnormalized scores) for each example.
#     targets : torch.Tensor 
#         A tensor of shape (batch_size,) containing the integer class index 
#         that is correct/relevant for each example in the batch.

#     Returns:
#     --------
#     float
#         The mean reciprocal rank across all samples in the batch.
#     """
#     with torch.no_grad():
#         # Sort each row of logits in descending order and get the indices
#         # shape: (batch_size, num_classes)
#         sorted_indices = torch.argsort(logits, dim=1, descending=True)

#         # Create a mask where the correct label is True in the sorted indices
#         # shape: (batch_size, num_classes)
#         correct_mask = (sorted_indices == targets.unsqueeze(1))

#         # Find the (batch_idx, rank) where the correct label appears
#         # shape: (batch_size, 2) if every target appears exactly once
#         found_indices = correct_mask.nonzero(as_tuple=False)
        
#         # Sort found_indices by batch_idx so that rank alignment is consistent
#         # found_indices[:, 0] is the batch_idx
#         # found_indices[:, 1] is the rank position for that batch_idx
#         found_indices = found_indices[found_indices[:, 0].argsort()]

#         # Extract the rank positions; rank is zero-based, so add 1 later
#         ranks = found_indices[:, 1]

#         # Compute reciprocal rank: 1 / (rank+1) for each sample
#         rr = 1.0 / (ranks + 1).float()

#         # Return the mean reciprocal rank
#         return rr.mean().item()