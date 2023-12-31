# Triple-I Net (TINet)
Official code for "Illumination-guided RGBT Object Detection with Inter- and Intra-modality Fusion"

![The structure of TINet](2_new_overall.jpg)

## Installation
Please refer to <https://github.com/open-mmlab/mmdetection/tree/2.x>

## Results
Below is the ablation study for our TINet on FLIR-aligned (the training and testing splits follow the official splits). 
Note that in our paper the results are given by a different train/test data distribution. 
If you intend to include our results, please make sure that the data distribution is aligned.

|     IGFW    |     Inter-MA    |     Intra-MA    |     AP50     |     mAP      |
|-------------|-----------------|-----------------|--------------|--------------|
|             |                 |                 |     75.19    |     35.88    |
|     √       |                 |                 |     74.94    |     36.41    |
|             |     √           |                 |     75.00    |     36.21    |
|             |                 |     √           |     74.96    |     36.07    |
|     √       |     √           |                 |     75.27    |     36.70    |
|     √       |                 |     √           |     75.42    |     36.61    |
|             |     √           |     √           |     75.32    |     36.06    |
|     √       |     √           |     √           |     76.07    |     36.54    |

## Citation
````
@ARTICLE{tinet,
  author={Zhang, Yan and Yu, Huai and He, Yujie and Wang, Xinya and Yang, Wen},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Illumination-Guided RGBT Object Detection With Inter- and Intra-Modality Fusion}, 
  year={2023},
  volume={72},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2023.3251414}}
````
