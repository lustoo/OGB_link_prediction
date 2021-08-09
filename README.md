# ogbl-ppa

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

|    Model     | Test Hits@100 | Val Hits@100  |
| :----------: | :-----------: | :-----------: |
| MLP+RA&CN&AA | 0.5062±0.0035 | 0.4906±0.0029 |
|    MLP+CN    | 0.3064±0.0116 | 0.3161±0.0070 |
|    MLP+RA    | 0.4896±0.0048 | 0.4794±0.0029 |
|    MLP+AA    | 0.3459±0.0033 | 0.3454±0.0029 |

# ogbl-ddi



We introduce an additional anchor sampling strategy and modify the aggregation stage of GraphSAGE. Results show that our strategy and modification surpass existing methods by a large extent.

## Train and Predict

```python
python link_pred_ddi_graphsage_edge.py --node_emb 512 --hidden_channels 512 --num_samples 3
```

## Results

|       Model       | Test Hits@20  |  Val Hits@20  |
| :---------------: | :-----------: | :-----------: |
|     baseline      | 0.7985±0.0494 | 0.8152±0.0310 |
| DE with edge(k=1) | 0.8633±0.0313 | 0.7916±0.0324 |
| DE with edge(k=2) |               |               |
| DE with edge(k=3) | 0.8781±0.0474 | 0.8044±0.0404 |
| DE with edge(k=4) |               |               |
| DE with edge(k=5) | 0.8527±0.0247 | 0.7839±0.0278 |