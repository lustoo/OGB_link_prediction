# ogbl-ppa-MLP

We combine node labels with scores drew from local similarity measures and use a simple MLP to gain a good performance on the link prediction task.

## Requirements

Python>=3.6

Pytorch>=1.4

torch-geometric>=1.6.0

ogb>=1.3.1

## Generate Feature

```python
python generate_feature.py
```

## Train and Predict

```pytho
python train.py --sim all
```

## Results

We conduct the experiments for 10 times with the random seed 0~9 and results are below:

|    Model     | Test Hits@20  |  Val Hits@20  |
| :----------: | :-----------: | :-----------: |
| MLP+RA&CN&AA | 0.5062±0.0035 | 0.4906±0.0029 |
|    MLP+CN    |               |               |
|    MLP+AA    |               |               |
|    MLP+RA    |               |               |