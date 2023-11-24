## Temporary anonymous repository for reviewers

- The repo is a temporary anonymous place, providing code and supplementary experiments for rebuttal.
- Complete repository would be made public if the paper is accepted.
- Related experimental codes are placed in [code](https://github.com/AnonymousForICASSP2024/TemporaryAnonymousRepForReviewers/tree/master/code).
- The paper with additional experiments are placed in [paper](https://github.com/AnonymousForICASSP2024/TemporaryAnonymousRepForReviewers/blob/master/docs/paper.pdf).


## Responses to the reviewers' comments

- In this section, we would focus on the concerns of the reviewers.


### Experimental results using larger and realistic datasets.

- table: Experimental results for Clothing1M, (mini) WebVision, and ILSVRC12 datasets.
Clothing1M-I: Training with all 1 million samples of Clothing1M. 
Clothing1M-II: Training with randomly selected 5000 samples.
The test accuracy (\%) is evaluated on Clothing1M validation set, WebVision validation set, and ILSVRC12 validation set respectively.

| **Methods**         | **Clothing1M-I** | **Clothing1M-II** | **WebVision** | **ILSVRC12** |
|----------------------|------------------|-------------------|---------------|--------------|
| CE (Standard)        | 69.55            | 45.11             | -             | -            |
| T-Revision           | 74.18           | 40.32             | -             | -            |
| PTD                  | 71.67           | 25.33             | -             | -            |
| ELR+                 | 74.81      | *60.67*           | 77.78    | 70.29   |
| DivideMix            | 74.76       | 56.57             | 77.32   | 75.20  |
| ProMix               | 72.85            | 55.39             | 75.73         | 74.96        |
| SOP                  | 73.50      | 48.78             | 76.60     | 69.10    |
| ClusterMix (ours)   | 74.84            | 61.98        | 78.19     | 75.54    |

We evaluate the proposed method on three larger datasets, i.e. Clothing1M with 1 million samples, WebVision with 2.4 million images, and ILSVRC12 with approximately 66 thousand images.
Experimental results illustrates the effectiveness of the proposed method on larger datasets.

### Experimental results compared with SOTA self-supervised method.

- Table: Experiments on CIFAR-10 dataset (50000 training samples) with symmetric label noise. 

| ACC(%) |     T-Revision    |     DivideMix    |     ClusterMix    |     ClusterMix (Given clean data)    |     SPICE (self-supervised)   |
|:---:|:---:|:---:|:---:|:---:|:---:|
|     20% Noise    |     88.10    |     95.97    |     96.35    |     96.52    |     92.26    |
|     40% Noise    |     84.11    |     94.82    |     95.50    |     95.83    |     92.26    |
| 60% Noise | 71.18 | 93.07 | 94.34 | 94.71 |     92.26    |
|     80% Noise    |     60.32    |     91.15    |     91.54    |     91.97    |     92.26    |


Experimental results illustrates the effectiveness of the proposed method, except the extremely high noise rate.


### Experimental results with different network structures.

To be updated.


### Experimental results with different number of clusters and clustering methods.

To be updated.