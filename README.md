# ogbl-ppa-MLP

We combine node labels with scores given by local similarity measures and use a simple MLP to gain a good performance on the link prediction task.

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
|    MLP+CN    | 0.3064±0.0116 | 0.3161±0.0070 |
|    MLP+RA    | 0.4896±0.0048 | 0.4794±0.0029 |
|    MLP+AA    | 0.3459±0.0033 | 0.3454±0.0029 |