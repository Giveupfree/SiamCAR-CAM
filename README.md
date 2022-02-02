# [SiamCAR](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html)
# [Group-CAM](https://arxiv.org/pdf/2103.13859v4.pdf)


## 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.7, Pytorch 1.7.1, CUDA 11.1.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
cd toolkit/utils/
python setup.py build_ext --inplace
```

## 2. model
- [x] [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR)

## 3. classfication heatmap
```bash 
cd /path/to/SiamCAR-CAM
conda activate SiamCAR
export PYTHONPATH=./:$PYTHONPATH
python tools/CAM-demo.py                                \
	--dataset_dir  /path/to/dataset/root            \ # dataset path
	--dataset UAV123                                \ # dataset name(OTB100, GOT10k, LaSOT, UAV123)
	--snapshot snapshot/general_model.pth           \ # tracker_name
	--format bmp                                    \ # save fomat (pdf,png,jpg,bmp)   
	--save_dir /path/to/save                        \ # save dir
	--config ./experiments/siamcar_r50/config.yaml  \ # config file
	--register_layer softmax                        \ # module register name
```
