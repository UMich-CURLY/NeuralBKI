# NeuralBKI
Welcome! This repository contains all software used to create the Bayesian Kernel Inference Neural Network.
<p align="center">
 <img width="800" src="https://user-images.githubusercontent.com/91337470/191110565-cc98d66e-43a9-4b8e-8b63-e657fd899a1f.gif">
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


