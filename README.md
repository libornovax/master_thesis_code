Vehicle Detection and Pose Estimation for Autonomous Driving
------------------------------------------------------------
**Libor Novak**, May 2017

This repository contains source code for my Master's thesis, which describes a deep leanining approach to 2D and 3D bounding box detectioin of cars from monocular images with an end-to-end neural network. The network was created by combining the ideas from DenseBox, SSD, and MS-CNN. It can perform multi-scale detection of 2D or 3D bounding boxes in a single pass and can run in 10fps on 0.5MPx images (images from the KITTI dataset) on a GeForce GTX Titan X GPU.

For details about the method see [PDF with the Master's thesis](master_thesis.pdf).


### 2D and 3D Bounding Box Detection Video
I created a video showing the output of a trained `r_2_x2_to_x16_s2` DNN on unseen data - sequences from the KITTI dataset, which you can find on [YouTube](https://youtu.be/O9OMIL0NwYk).

[![YouTube video with detections](https://img.youtube.com/vi/O9OMIL0NwYk/0.jpg)](https://youtu.be/O9OMIL0NwYk)
