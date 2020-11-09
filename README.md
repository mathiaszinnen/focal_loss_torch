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

...
criterion = FocalLoss(alpha=2, gamma=5)
...
```

## Contributions
Contributions, criticism or corrections are always welcome. 
Just send me a pull request!

## References 
<a id="1">[1]</a> 
Lin, T. Y., et al.
"Focal loss for dense object detection."
arXiv 2017." arXiv preprint arXiv:1708.02002 (2002).