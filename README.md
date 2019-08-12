## Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds
Bo Yang, Jianan Wang, Ronald Clark, Qingyong Hu, Sen Wang, Andrew Markham, Niki Trigoni. [arXiv:1906.01140](https://arxiv.org/abs/1906.01140), 2019.
### (1) Setup
ubuntu 16.04 + cuda 8.0

python 2.7 or 3.6

tensorflow 1.2 or 1.4

scipy 1.3

h5py 2.9

open3d-python 0.3.0

### (2) Data
[https://drive.google.com/open?id=1hOsoOqOWKSZIgAZLu2JmOb_U8zdR04v0](https://drive.google.com/open?id=1hOsoOqOWKSZIgAZLu2JmOb_U8zdR04v0)

Acknowledgement: we use the same data released by [https://github.com/pqhieu/jsis3d](JSIS3D).

### (3) Train/test
python main_train.py

python main_eval.py

### (4) Quantitative Results on ScanNet
![Arch Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_res_scannet.png)
### (5) Qualitative Results on ScanNet
![Arch Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_ins_scannet.png)
### (6) Qualitative Results on S3DIS
![Teaser Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_bb_s3dis.png)
### (7) Training Curves on S3DIS
![Teaser Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_traincurv_s3dis.png)
