Vehicle Detection and Pose Estimation for Autonomous Driving
------------------------------------------------------------
**Libor Novak**, May 2017

This repository contains source code for my Master's thesis, which describes a deep leanining approach to 2D and 3D bounding box detectioin of cars from monocular images with an end-to-end neural network. The network was created by combining the ideas from DenseBox, SSD, and MS-CNN. It can perform multi-scale detection of 2D or 3D bounding boxes in a single pass and can run in 10fps on 0.5MPx images (images from the KITTI dataset) on a GeForce GTX Titan X GPU.

For details about the method see [PDF with the Master's thesis](master_thesis.pdf).


### 2D and 3D Bounding Box Detection Video
I created a video showing the output of a trained `r_2_x2_to_x16_s2` DNN on unseen data - sequences from the KITTI dataset, which you can find on [YouTube](https://youtu.be/O9OMIL0NwYk) ([https://youtu.be/O9OMIL0NwYk](https://youtu.be/O9OMIL0NwYk)).

[![YouTube video with detections](mockup.png)](https://youtu.be/O9OMIL0NwYk)

### Network Models
The final 2D and 3D detection network architectures can be found in [caffe/models](caffe/models). There are 2 networks with the same structure:
  * `macc_0.3_r2_x2_to_x16_s2` - 2D bounding box detection network
  * `macc3d_0.3_r2_x2_to_x16_s2` - 3D bounding box detection network

### Testing
There are several executables for examination of the network testing output under [caffe/examples/ln](caffe/examples/ln). The fact that their names contain 'pyramid' is a bit misleading as now the image pyramid has only one scale and the detectors perform multiscale detection by themseslves.
  * `macc_pyramid_test` - running 2D detector
  * `macc3d_pyramid_test` - running 3D detector
  * `detect_pyramid` - displays response maps of a 2D or 3D detector
