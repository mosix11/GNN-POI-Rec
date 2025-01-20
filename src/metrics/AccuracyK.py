import torch
import torch.nn as nn

class AccuracyK(nn.Module):
    """
        Acc@K = (#hits within top K)/N_test
        
        inputs: 
            - `logits` a list of logits of shape [N_test, num_classes]
            - `targets` a list of ground truth integer labels of shape [N_test]
        ouputs:
            - Acc@K
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(targets.shape) == 2:
            targets.squeeze(1)
        predicted=logits.softmax(dim=-1)
        
        # Get the indices of the top-K predictions
        _, top_k=predicted.topk(self.k)
        top_k += 1 # the true labels start from 1
        hits = top_k.eq(targets.unsqueeze(1)).any(dim=-1)
        acc_at_k = hits.float().mean()
        return acc_at_k
