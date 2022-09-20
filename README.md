# NeuralBKI
Welcome! This repository contains all software used to create the Bayesian Kernel Inference Neural Network.
<p align="center">
 <img width="600" src="https://user-images.githubusercontent.com/91337470/191110565-cc98d66e-43a9-4b8e-8b63-e657fd899a1f.gif">
</p>


## Table of Contents
 - [Network](#network)
 - [Use NeuralBKI](#use-neuralbki)
   - [Dependencies](#dependencies)
   - [Datasets](#datasets)
 - [Results](#results)
   - [KittiOdometry](#kittiodom)
   - [SemanticKITTI](#semantickitti)
 - [Acknowledgement](#acknowledgement)


## Network: **NeuralBKI**
<p align="center">
 <img width="674" alt="Diagram" src="https://user-images.githubusercontent.com/91337470/191112305-26045690-65a1-47ae-a769-9d65e5877cd1.png">
</p>

## Use NeuralBKI
### Dependencies
* [Pytorch](https://pytorch.org/get-started/locally/) - we tested on PyTorch 1.10 and 1.8.2
* [ROS](http://wiki.ros.org/noetic) - we used ros noetic for map visualization  

We also provide an environment.yml which you can use to create a conda environment
```
conda env create -f environment.yml
conda activate NeuralBKI
```

### Datasets
* KittiOdometry 
  * We preprocessed the Kitti Odometry following [Yang et al.](https://github.com/shichaoy/semantic_3d_mapping/tree/master/preprocess_data#readme). We use [ELAS](https://www.cvlibs.net/software/libelas/) to generated depth images from Kitti Odometry dataset's stereo images. The semantic segmentation of the images are from a [Dilation Network](https://github.com/fyu/dilation). Then using the depth images and semantic segmentation, 3D point clouds can be generated from the image projections.
  * You can download the preprocessed data here.
* SemanticKitti
  * You can download the SemanticKitti [ground turth](http://www.semantic-kitti.org/dataset.html#download) and the semantic segmentation output from [darknet53 with KNN](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-knn.tar.gz)

## Results

#### KittiOdometry
| Method      | Building | Road | Vege. | Sidewalk | Car | Sign | Fence | Pole | Average |
|-------------------|------|------|------|------|------|------|------|------|------|
| Segmentation      | 92.1 | 93.9 | 90.7 | 81.9 | 94.6 | 19.8 | 78.9 | 49.3 | 75.1 |
| Yang et al.       | 95.6 | 90.4 | 92.8 | 70.0 | 94.4 | 0.1  | 84.5 | 49.5 | 72.2 |
| BGKOctoMap-CRF    | 94.7 | 93.8 | 90.2 | 81.1 | 92.9 | 0.0  | 78.0 | 49.7 | 72.5 |
| S-CSM             | 94.4 | 95.4 | 90.7 | 84.5 | 95.0 | 22.2 | 79.3 | 51.6 | 76.6 |
| S-BKI             | 94.6 | 95.4 | 90.4 | 84.2 | 95.1 | 27.1 | 79.3 | 51.3 | 77.2 |
| ConvBKI Single    | 92.7 | 94.8 | 90.9 | 84.7 | 95.1 | 22.1 | 80.2 | 52.1 | 76.6 |
| ConvBKI Per Class | 94.0 | 95.5 | 91.0 | 87.0 | 95.1 | 22.8 | 81.8 | 52.9 | 77.5 |
| ConvBKI Compound  | 94.0 | 95.6 | 91.0 | 87.2 | 95.1 | 22.8 | 81.9 | 54.3 | 77.7 |

#### SemanticKitti
| Data Split | Method | Car | Bicycle | Motocycle | Truck | Other Veh. | Person | Bicylist | Motorcyclist | Road | Parking | Sidewalk | Other Gr.  | Building | Fence | Vegetation | Trunk | Terrain | Pole | Sign | Average |
|---|---------|----|------|------|------|------|------|------|-----|-----|------|------|------|------|------|------|------|------|----|----|------|
| Val | Da.-kNN      | 91.0 | 25.0 | 47.1 | 40.7 | 25.5 | 45.2 | 62.9 | 0.0 | 93.8 | 46.5 | 81.9 | 0.2  | 85.8 | 54.2 | 84.2 | 52.9 | 72.7 | 53.2 | 40.0 | 52.8 |
|   | S-CSM        | 92.6 | 32.5 | 54.9 | 43.4 | 26.2 | 51.3 | 69.2 | 0.0 | 94.6 | 49.2 | 84.0 | 0.1  | 87.9 | 58.4 | 85.8 | 59.9 | 73.3 | 61.7 | 43.0 | 56.2 |
|   | S-BKI        | 93.5 | 33.5 | 57.3 | 44.5 | 27.2 | 52.9 | 72.1 | 0.0 | 94.4 | 49.6 | 84.0 | 0.0  | 88.7 | 59.6 | 86.9 | 62.5 | 75.3 | 63.6 | 45.1 | 57.4 |
|   | ConvBKI Sin. | 92.0 | 29.8 | 57.4 | 44.4 | 25.2 | 53.1 | 72.1 | 0.0 | 93.1 | 45.8 | 80.9 | 0.1  | 88.2 | 57.8 | 86.1 | 61.2 | 74.0 | 59.7 | 44.4 | 56.1 |
|   | ConvBKI PC   | 92.6 | 34.5 | 59.2 | 34.6 | 39.4 | 58.6 | 73.5 | 0.0 | 93.0 | 47.2 | 80.9 | 0.1  | 88.4 | 58.3 | 86.4 | 61.7 | 74.2 | 58.4 | 47.4 | 57.3 |
|   | ConvBKI Com. | 94.0 | 37.5 | 60.0 | 33.3 | 40.5 | 59.4 | 74.4 | 0.0 | 93.3 | 49.0 | 81.2 | 0.1  | 88.5 | 59.5 | 86.8 | 62.2 | 75.0 | 59.9 | 46.5 | 58.0 |
|---|---------|----|------|------|------|------|------|------|-----|-----|------|------|------|------|------|------|------|------|----|----|------|
| Test | Da.-kNN      | 82.4 | 26.0 | 34.6 | 21.6 | 18.3 | 6.7  | 2.7  | 0.5 | 91.8 | 65.0 | 75.1 | 27.7 | 87.4 | 58.6 | 80.5 | 55.1 | 64.8 | 47.9 | 55.9 | 47.5 |
|   | S-BKI        | 83.8 | 30.6 | 43.0 | 26.0 | 19.6 | 8.5  | 3.4  | 0.0 | 92.6 | 65.3 | 77.4 | 30.1 | 89.7 | 63.7 | 83.4 | 64.3 | 67.4 | 58.6 | 67.1 | 51.3 |
|   | ConvBKI Com. | 83.8 | 32.2 | 43.8 | 29.8 | 23.2 | 8.3  | 3.1  | 0.0 | 91.4 | 62.6 | 75.2 | 27.5 | 89.1 | 61.6 | 81.6 | 62.5 | 65.2 | 53.9 | 63.0 | 50.4 |

#### Qualitative 
* Example map produced by ConvBKI Compound on the validation set of Semantic KITTI. It can be seen that filtering out voxels with high variance improves the quality of the robotic map. 
<p align="center">
 <img width="600" alt="Diagram" src="https://user-images.githubusercontent.com/91337470/191287167-dd1e67c5-2f0c-4bba-bd2a-d7f47e2aedaa.png">
</p>

* Illustration of kernels learned by ConvBKI on the road and pole semantic classes
<p align="center">
 <img width="600" alt="Diagram" src="https://user-images.githubusercontent.com/91337470/191287635-10ff78ce-c8ae-4044-a241-c1db252801d8.png">
</p>

## Acknowledgement
We utilize data and code from: 
- [1] [Kitti Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [2] [SemanticKITTI](http://www.semantic-kitti.org/)
- [3] [ELAS](https://www.cvlibs.net/software/libelas/)
- [4] [Dilation Network](https://github.com/fyu/dilation)
- [5] [darknet53 with KNN](https://github.com/PRBonn/lidar-bonnetal)

## Reference
If you find our work useful in your research work, consider citing [our paper]
```
```
Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping ([PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954837))
```
@ARTICLE{gan2019bayesian,
author={L. {Gan} and R. {Zhang} and J. W. {Grizzle} and R. M. {Eustice} and M. {Ghaffari}},
journal={IEEE Robotics and Automation Letters},
title={Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping},
year={2020},
volume={5},
number={2},
pages={790-797},
keywords={Mapping;semantic scene understanding;range sensing;RGB-D perception},
doi={10.1109/LRA.2020.2965390},
ISSN={2377-3774},
month={April},}

```





