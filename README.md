# Deep learning model for extensive smartphone-based diagnosis and triage of cataracts and multiple corneal diseases

## About This repository

We provide source codes of our deep learning model proposed in [our journal paper published in the British Journal of Ophthalmology](https://bjo.bmj.com/content/early/2024/01/18/bjo-2023-324488).
Most parts of the codes are same as [YOLOv5](https://github.com/ultralytics/yolov5).

## How to use code

### Training

Please put your dataset for training in data_train directory in yaml format.  
Run training:
```python
python train.py --data data_train/yourdataset.yaml --cfg yolov5s.yaml --batch-size 16 --epochs 200
```

Details of the running options can be found in the documentation provided by [YOLOv5](https://github.com/ultralytics/yolov5).

### Inference

Please put image files for inference in data_inference directory.  
Run inference:
```python
python detect.py --source data_inference --weights weights/yourweightfile.pt
```
Results will be saved in runs/detect/exp/ directory.  
_result_allbox.txt file provides estimation results of image files.  
Each line in the txt file means:
```bash
filename, x0 coordinate of bounding box (BB), y0 of BB, width of BB, height of BB, class likelihood, class number
```
Correspondence between class numbers and its meanings:  
0: infection  
1: normal  
2: non-infection  
3: scar  
4: tumor  
5: deposit  
6: APAC  
7: lens opacity  
8: bullous  
