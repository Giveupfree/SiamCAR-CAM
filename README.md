# [SiamCAR](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html)

## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1/1.2.0, CUDA 9.0.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test
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

## 3. Train

### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg) (code: v7s6)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)
* [LaSOT](https://cis.temple.edu/lasot/)

**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.

### Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1IfZoxZNynPdY2UJ_--ZG2w) (code: 7n7d) and put them into `pretrained_models` directory.

### Train a model
To train the SiamCAR model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

## 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1z3bwLcxIsg8u_ZAZPlbEVw) (code: 8c7b) (results in [google driver](https://drive.google.com/drive/folders/1i-Ar4NbJauOo7SIqPVhDsLxJSc3s3-zB?usp=sharing) )of GOT10K, LaSOT, OTB and UAV. If you want to evaluate the tracker, please put those results into  `results` directory.

```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV123                  \ # dataset_name
	--tracker_prefix 'general_model'   # tracker_name
```

## 5. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.


## 6. Cite
If you use SiamCAR in your work please cite our papers:
> @InProceedings{Guo_2020_CVPR,  
   author = {Guo, Dongyan and Wang, Jun and Cui, Ying and Wang, Zhenhua and Chen, Shengyong},  
   title = {SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking},  
   booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
   month = {June},  
   year = {2020}  
}

> @InProceedings{Guo_2021_CVPR,  
  author = {Guo, Dongyan and Shao, Yanyan and Cui, Ying and Wang, Zhenhua and Zhang, Liyan and Shen, Chunhua},  
  title = {Graph Attention Tracking},  
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  month = {June},  
  year = {2021}  
}
