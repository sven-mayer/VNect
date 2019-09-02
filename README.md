# VNect

A tensorflow implementation of [VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/VNect/).

For the **caffe model** required in the repository: please contact [the author of the paper](http://gvv.mpi-inf.mpg.de/projects/VNect/).



<p align="center">
    <img src="./pic/test_pic_show.png" height="260">
</p>
<p align="center">
    <img src="./pic/test_video_show.gif" height="300">
</p>


## Environments

- Python 3.5
  - opencv-python 3.4.4.19
  - tensorflow-gpu 1.12.0
  - [pycaffe](https://github.com/BVLC/caffe/tree/windows)
  - matplotlib 3.0.0 or 3.0.2 (this module shuts down occasionally for unknown reason)
  - ……

## Setup
<details><summary>Fedora 29</summary>
<p>

#### Install python dependencies:
```
pip3 install -r requirements.txt --user
```
#### Install caffe dependencies
```
sudo dnf install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel glog-devel gflags-devel lmdb-devel atlas-devel python-lxml boost-python3-devel
```
#### Setup Caffe
```
git clone https://github.com/BVLC/caffe.git
cd caffe
```

#### Configure Makefile.config (Include python3 and fix path)

#### Build Caffe
```
sudo make all
sudo make runtest
sudo make pycaffe
sudo make distribute
sudo cp .build_release/lib/ /usr/lib64
sudo cp -a distribute/python/caffe/ /usr/lib/python3.7/site-packages/
```
</p>
</details>


## Usage

### Preparation

1. Drop the caffe model into `models/caffe_model`.
2. Run `init_weights.py` to generate tensorflow model.

### Application

1. `benchmark.py` is a class implementation containing all the elements needed to run the model.

2. `run_estimator.py` is a script for running with **video stream**.

3. `run_estimator_ps.py` is a multiprocessing version script. When 3d plotting function shuts down in `run_estimator.py` mentioned above, you can try this one.

4. `run_estimator_robot.py` provides ROS and serial connection for communication in robot controlling besides the functions in `run_estimator.py`.

5. **[NOTE]** To run the video stream based scripts mentioned above:

   i ) click the left mouse button to confirm a simple static bounding box generated by HOG method;

   ii) trigger any keyboard input to exit while the network running.

6. `run_pic.py` is a script for running with **one single picture**: the outputs are 4×21 heatmaps and 2D results.



## Notes

1. I don't know why in some cases the 3d plotting function shuts down in the script. It may result from the variety of programming environments. Anyone capable to fix this and pull a request would be gratefully appreciated.
2. The input image in this implementation is in BGR color format (cv2.imread()) and the pixel value is regulated into a range of [-0.4, 0.6).
3. The joint-parent map (detailed information in `joint_index.xlsx`):

<p align="center">
    <img src="./pic/joint_index.png" height="300">
</p>

3. The joint positions (index numbers as above):

<p align="center">
    <img src="./pic/joint_pos.jpg" height="300">
</p>


4. Every input image is assumed to contain 21 joints to be found, which means it is easy to fit wrong results when a joint is actually not in the input.
5. In some cases the estimation results are not so good as the results shown in the paper author's promotional video.
6. UPDATE: the running speed is now faster thanks to some coordinate extraction optimization!



## TODO

1. Optimize the structure of the codes.
2. Implement a better bounding box strategy.
4. **Implement the training script.**



## About Training Data

Refer to [MPI-INF-3DHP Dataset](https://github.com/XinArkh/mpi_inf_3dhp)



## Reference Repositories

- original MATLAB implementation provided by the model author
- [timctho/VNect-tensorflow](https://github.com/timctho/VNect-tensorflow)
- [EJShim/vnect_estimator](https://github.com/EJShim/vnect_estimator)
