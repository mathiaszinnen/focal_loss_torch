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

# To make it behaves as CrossEntropy loss
criterion = FocalLoss(gamma=0)
```

### Examples
For Binary-classification
```python
batch_size = 10
m = torch.nn.Sigmoid()
logits = torch.randn(batch_size)
target = torch.randint(0, 2, size=(batch_size,))
loss = criterion(m(logits), target)
```

For Multi-Class classification
```python
batch_size = 10
n_class = 5
m = torch.nn.Softmax(dim=-1)
logits = torch.randn(batch_size, n_class)
target = torch.randint(0, n_class, size=(batch_size,))
criterion(m(logits), target)
```

For Multi-Class Sequence classification
```python
batch_size = 10
max_length = 20
n_class = 5
m = torch.nn.Softmax(dim=-1)
logits = torch.randn(batch_size, max_length, n_class)
target = torch.randint(0, n_class, size=(batch_size, max_length))
criterion(m(logits), target)
```


## Contributions
Contributions, criticism or corrections are always welcome. 
Just send me a pull request!

## References 
<a id="1">[1]</a> 
Lin, T. Y., et al.
"Focal loss for dense object detection."
arXiv 2017." arXiv preprint arXiv:1708.02002 (2002).
