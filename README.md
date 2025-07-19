# 3D-Data-Processing

Each homework tackles a different popular task concerning 3D Data, the first three were implemented using C++ meanwhile the last one consists in a Jupyter Notebook. In both cases, the OpenCV library has been used:

-In Homework 1 a SGM Stereo Matching has been implemented. It was tested on several images, collecting the relative errors

-In Homework 2 a complete classic SfM has been implemented. Also the feature extractor and matcher has been implemented from scratch. The implementation was tested on several images, providing different Point Clouds

-In Homework 3 an implementation of the classic ICP registration has been implemented. The initial transformation between the Point Clouds was not given, so in order to obtain good initial matches a descriptor based approach was implemented. The descriptor used in this case was the FPFH.

-In Homework 4 the PointNet architecture has been modified in order to retrieve a low dimensional descriptor of a 3D Point Cloud, which is then used in order to classify it.
