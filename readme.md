## AEPL: Automated and Editable Prompt Learning for Brain Tumor Segmentation
This repository provides the code for "AEPL: Automated and Editable Prompt Learning for Brain Tumor Segmentation". 





![AEPL](./pictures/AEPL_v6.png)
Fig. 1. Structure of AEPL.



### Requirementss
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* Python == 3.7 
* Some basic python packages such as Numpy.
* nnUNetv1
* Follow official guidance to install [Pytorch][torch_link]. 
* Follow official guidance to install [nnUNetv1][nnUNetv1_link].

[torch_link]:https://pytorch.org/
[nnUNetv1_link]:https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1?tab=readme-ov-file#installation

## Dataset
You can download BraTS 2018 via
https://www.med.upenn.edu/sbia/brats2018/data.html
## Usages
### Preprocess
```
python ./preprocess/BraTS.py
python ./preprocess/generate_json.py
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity
```

### Train
```
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2_AEPL 501 0
```

### Test
```
CUDA_VISIBLE_DEVICES=0 nnUNet_predict -i /nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_BraTS/imagesTs/ -o /nnUNet/nnUNet_output/Task501_BraTS/nnUNetTrainerV2_AEPL/fold_0 -t 501 -m 3d_fullres -f 0 -tr nnUNetTrainerV2_AEPL
```

2. Our experimental results are shown in the table:
![refinement](./pictures/img.png)



