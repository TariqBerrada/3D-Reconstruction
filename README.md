# 3D-Reconstruction
An implementation of 3D reconstruction methods given in openCV.

## Image Calibration

Getting rid of tangential and radial distortions for a certain camera output.

**before:**

![](chessboard.jpeg)

**after:**

![](calibresult.png)

## Get a 3D axes 

Calculate the 3D axes associated wih the predominant object in the camera's perspective, making use of epipolar geometry to calculate the required angles.

![](axes.jpg)
![](cube.jpg)

## Depth Estimation by calculating disparity
![](left.png)
![](right.png)
**Result**
![](heatmap.jpg)
![](heatmap_3D.gif)
