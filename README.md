# Structure-from-Motion

The aim was to construct 3D sparse reconstruction of a scene using a set of unordered images

### Algorithm
* Extract features between two images using fast features
* Estimate the relative pose of 2nd frame wrt 1.
* Triangulate common features found between two images to obtain 3D point.
* Append the point in a .ply file
* Repeat the above 4 steps for the series of images.



