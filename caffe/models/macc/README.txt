Multiscale Accumulator Network
==============================
These are networks for generating multiscale accumulators.

The scales of the accumulators are given by pooling, thus they may only take 1/1, 1/2, 1/4, 1/8, 1/16, ... of the input image dimensions as the pooling layers always shrink the image 2x.


Accumulator Size Computation
----------------------------
We have to choose a size of the circle with which we will be representing each bounding box in the accumulator. As in the DenseBox paper, we chose the diameter of the circle to be 0.3*max(w,h), where w and h are the width and height of the bounding box.

However, in each accumulator we can only create circles of a certain radius (diameter). This radius r will be the same for all bounding boxes in the accumulator. Therefore the aforementioned formula tells us into which accumulator (which scale) should the circle from this bounding box be added.

By choosing r, we get the maximum bounding box size to be included in the accumulator of a given scale. Because in OpenCV, if we plot a circle with r=0, we get a single pixel, with r=1 we get diameter d=3, thus always one pixel larger than we would want.

This table shows circle diameters in accumulators of different scales:
     |    r =
     | 0   1   2
-----|------------
1/1  | 1   3   5
1/2  | 2   6   10
1/4  | 4   12  20
1/8  | 8   24  40
1/16 | 16  48  80
1/32 | 32  96  160
1/64 | 64  192 320

This means that accumulator of scale 1/32, if we use r=1 can represent objects (bounding boxes) up to 320x320 pixels. But if we chose to use r=2 it could represent objects all the way up to 533x533 pixels.


Network Creation
----------------
Inspired by DenseBox, we say that we want to detect objects (bounding boxes) with a convolutional kernel, which is double the size of the bounding box (largest side of the bounding box). This means that an object in a bounding box 30x50 we want to detect by a kernel with field of view (FOV) 100x100. It is because we want to allow the detector to learn some context around the objects.

Taken from the previous example, if we then want to detect objects in 1/32 accumulator with r=1 (i.e. bounding boxes up to 320x320 pixels) we need to create this accumulator with a convolutional kernel with FOV at least 640x640.

The FOV of a convolution can be broadened by using a larger convolutional kernel, multiple convolutional layers, or multiple layers with 'atrous' (dilated) convolution, which increases the FOV the most. We conveniently use 'atrous' convolution because it removes the need for upsampling.


Choosing Layers
---------------
The whole network was designed to follow these rules. The users can then select which accumulators they want to use in their network by considering the minimal and maximal size of objects they want to detect.

The networks are names using the following scheme (the 0.3 refers to the size of circle with respect to the object bounding box, which was taken from DenseBox):

  * macc_0.3_r<accumulator_circle_radius>