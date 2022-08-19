# MobileNetV2-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381v4.pdf)
.

## Table of contents

- [MobileNetV2-PyTorch](#mobilenetv2-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](#mobilenetv2-inverted-residuals-and-linear-bottlenecks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `mobilenet_v2`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `mobilenet_v2`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change
  to `./results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `mobilenet_v2`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/mobilenet_v2-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1801.04381v4.pdf](https://arxiv.org/pdf/1801.04381v4.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|    Model     |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------:|:-----------:|:-----------------:|:-----------------:|
| mobilenet_v2 | ImageNet_1K | 28.0%(**28.0%**)  |    -(**9.4%**)    |

```bash
# Download `MobileNetV2-ImageNet_1K-86ab0476.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `mobilenet_v2` model successfully.
Load `mobilenet_v2` model weights `/MobileNetV2-PyTorch/results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar` successfully.
tench, Tinca tinca                                                          (24.90%)
barracouta, snoek                                                           (7.63%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (1.00%)
soccer ball                                                                 (0.71%)
reel                                                                        (0.66%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### MobileNetV2: Inverted Residuals and Linear Bottlenecks

*Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen*

##### Abstract

In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of
mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe
efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally,
we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile
DeepLabv3.
The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block
are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an
MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.
Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain
representational power. We demonstrate that this improves performance and provide an intuition that led to this design.
Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which
provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object
detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by
multiply-adds (MAdd), as well as the number of parameters

[[Paper]](https://arxiv.org/pdf/1801.04381v4.pdf)

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```