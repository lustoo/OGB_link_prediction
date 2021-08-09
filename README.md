# ogbl-ppa

We combine node labels with scores calculated by local similarity measures and use a simple MLP for the link prediction task which can obtain good performance.

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

We conduct the experiments for 10 times with the random seed 0~9 and results are listed below:

|      Model       |   Test Hits@100   |   Val Hits@100    |
| :--------------: | :---------------: | :---------------: |
|      MLP+CN      |   0.3064±0.0116   |   0.3161±0.0070   |
|      MLP+RA      |   0.4896±0.0048   |   0.4794±0.0029   |
|      MLP+AA      |   0.3459±0.0033   |   0.3454±0.0029   |
| **MLP+RA&CN&AA** | **0.5062±0.0035** | **0.4906±0.0029** |

# ogbl-ddi



We use multiple anchor sets selected from random sampling to encode distance information for edges on graph. We also modify the aggregation stage of GraphSAGE to incorporate edge information.

## Train and Predict

To get the best performance, run:

```python
python link_pred_ddi_graphsage_edge.py --node_emb 512 --hidden_channels 512 --num_samples 3
```

## Results

|            Model             |   Test Hits@20    |    Val Hits@20    |
| :--------------------------: | :---------------: | :---------------: |
|   GraphSAGE+Edge Attr(k=1)   |   0.8633±0.0313   |   0.7916±0.0324   |
| **GraphSAGE+Edge Attr(k=3)** | **0.8781±0.0474** | **0.8044±0.0404** |
|   GraphSAGE+Edge Attr(k=5)   |   0.8527±0.0247   |   0.7839±0.0278   |