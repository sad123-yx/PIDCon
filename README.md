# Introduction

Connectivity Relationship Recognition in Piping and Instrumentation Diagrams Using Graph Neural Networks

# Contribution

1. We introduce a novel connectivity recognition method based on Graph Neural Networks that overcomes the limitations of conventional position-based approaches, significantly improving the accuracy of connectivity recognition in complex P&ID scenarios.
2. We have constructed PIDCon, a P&ID dataset annotated for multiple tasks, including connectivity recognition, component detection, line segment detection, line segment logic analysis, and digital twin construction.

# Install requirements

```
pip3 install -r requirements.txt
```

# PIDCon dataset

Our proposed dataset PIDCon consists of 600 P&ID images, encompassing 82 unique component categories and a total of 7,212 component connection pairs.

The anotation info include：

* Box:			(xywh,class)
* Line:			(xyxy,id)
* Connection:	[obj_1, obj_2]
* Path:			{line_1, ..., line_n}

Upon acceptance of this paper, we will release the complete dataset; at present, the publicly available version includes annotations for components boxes, line endpoints, and connection relationships. This dataset aims to facilitate further research and validation in the field of P&ID digitization.

The PIDCon dataset can download at [Baidu Disk](https://pan.baidu.com/s/1LObGLWcrH06_r_7FGyOjpw?pwd=8888)

# Training

* python3 train_prune.py
  
  During model training，you can change the GCNConv layer num or change the distance calculation method (midpoint or box).  We suggest using midpoint method during training and using box method during testing.

# Testing

* python3 test.py
  
  The final output includes six metrics: Accuracy, Precision, Recall, F1-Score, Graph Edit Distance (GED), and Node Connectivity Accuracy(NCA)
  
  Give an example as follows：
  
  | Method| Accuracy| Precision|Recall|F1-Score|GED|NCA|
  | --- | --- | --- | --- | --- | --- | --- |
  |Linear Assigment| 0.345 |0.648  |0.345|0.451|0.069|0.748
  |Position_Based  |  0.886|  **0.985**|0.886|0.933|0.024|0.973
  |**GNN (ours)**|**0.989**|0.938|**0.989**|**0.963**|0.017|**0.975**
  
