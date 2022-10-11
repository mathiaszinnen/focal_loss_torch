import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(
            self,
            alpha=1,
            gamma=2,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def _process_target(
            self, target: Tensor, num_classes: int
            ) -> Tensor:
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        mask = target == self.ignore_index
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = (focal ** self.gamma) * nll
        return self._reduce(loss)

    def _reduce(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
