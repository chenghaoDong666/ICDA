import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self):
        super(PrototypeContrastiveLoss, self).__init__()

    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != 255)
        labels = labels[mask]
        feat = feat[mask]
        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)
        # (N,A) x (A,N) = (N,N)
        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / 50.0
        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(logits, labels)
        
        return loss
