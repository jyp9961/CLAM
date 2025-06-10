# Method
We implement CLAss-dependent Multiplicative-weights (CLAM) method to reduce class-dependent effects of data augmentation in classification tasks.

# Instructions
We assume that you have your conda environment and pytorch installed.

Train a classifier with CLAM on CIFAR-10 with crop lower bound equal to 1. 
```
python train_CV.py --CLAM_loss true --crop_lower_bound 1.0 --task cifar10
```

Train a classifier with CLAM on CIFAR-100 with crop lower bound equal to 1. 
```
python train_CV.py --CLAM_loss true --crop_lower_bound 1.0 --task cifar100
```

Train a classifier with CLAM on Fashion-Mnist with crop lower bound equal to 1. 
```
python train_CV.py --CLAM_loss true --crop_lower_bound 1.0 --task fmnist
```

Train a classifier with CLAM on Mini-Imagenet with crop lower bound equal to 1. 
```
python train_CV.py --CLAM_loss true --crop_lower_bound 1.0 --task miniImagenet
```

For all examples above, "--crop_lower_bound" can be set as any float number between 0 (not included) and 1 (included).

We assume that you have downloaded Imagenet dataset and convert the training and validation dataset to "train.beton" and "test.beton".
If this is not the case, please refer to [ffcv](https://github.com/libffcv/ffcv).

Train a classifier with CLAM on Imagenet with crop lower bound equal to 1. In this example, "--training.crop_scal" can be set as any number between 0 (not included) and 100 (included).
```
python train_imagenet.py --training.loss_type CLAM --training.crop_scale=100 --data.train_dataset train.beton --data.val_dataset test.beton
```

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@misc{jiang2025understandingreducingclassdependenteffects,
      title={Understanding and Reducing the Class-Dependent Effects of Data Augmentation with A Two-Player Game Approach}, 
      author={Yunpeng Jiang and Paul Weng and Yutong Ban},
      year={2025},
      eprint={2407.03146},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2407.03146}, 
}
```
