# focal_loss_torch
Simple pytorch implementation of focal loss introduced by Lin et al [[1]](#1).

## Usage
Install the package using pip
```bash
pip install focal_loss_torch
```

Focal loss is now accessible in your pytorch environment:
```python
from focal_loss.focal_loss import FocalLoss

# Withoout class weights
criterion = FocalLoss(gamma=0.7)

# with weights 
# The weights parameter is similar to the alpha value mentioned in the paper
weights = torch.FloatTensor([2, 3.2, 0.7])
criterion = FocalLoss(gamma=0.7, weights=weights)

# to ignore index 
criterion = FocalLoss(gamma=0.7, ignore_index=0)
```

## Contributions
Contributions, criticism or corrections are always welcome. 
Just send me a pull request!

## References 
<a id="1">[1]</a> 
Lin, T. Y., et al.
"Focal loss for dense object detection."
arXiv 2017." arXiv preprint arXiv:1708.02002 (2002).
