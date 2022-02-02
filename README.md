# [SiamCAR](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html)
# [Group-CAM](https://arxiv.org/pdf/2103.13859v4.pdf)


## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1/1.2.0, CUDA 9.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. model
- [x] [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR)
Download the pretrained model:  
[general_model](https://pan.baidu.com/s/1ZW61I7tCe2KTaTwWzaxy0w) code: lw7w  
[got10k_model](https://pan.baidu.com/s/1KSVgaz5KYP2Ar2DptnfyGQ) code: p4zx  
[LaSOT_model](https://pan.baidu.com/s/1g15wGSq-LoZUBxYQwXCP6w) code: 6wer  
(The model in [google Driver](https://drive.google.com/drive/folders/1ud0iF4Vm96TfxOddUHV1LoY-soF2zk8b?usp=sharing))
 and put them into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. Jsons of commonly used datasets can be downloaded from [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F) or [Google driver](https://drive.google.com/drive/folders/1TC8obz4TvlbvTRWbS4Cn4VwwJ8tXs2sv?usp=sharing). If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```bash 
python test.py                                \
	--dataset UAV123                      \ # dataset_name
	--snapshot snapshot/general_model.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.
