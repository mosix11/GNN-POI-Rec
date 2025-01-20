import torch
import torch.nn as nn

class AccuracyK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        predicted=logits.softmax(dim=-1)
        top_k=predicted.topk(self.k, dim=-1)[1]
        correct=(top_k==targets.unsqueeze(-1)).any(dim=-1).float()
        if padding_mask is not None:
            correct *= padding_mask.float()
            # Avoid division by zero by counting non-zero elements in the mask
            accuracy = correct.sum() / padding_mask.float().sum()
        else:
            accuracy = correct.mean()

        return accuracy


class Accuracy1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        predicted = logits.argmax(dim=-1)
        correct = (predicted == targets).float()
        accuracy = correct.mean()
        return accuracy