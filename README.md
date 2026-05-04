# Enhancing SAM2 for Industrial Defect Detection via Dual-Adapter Fine-Tuning

## Requirements

This project uses the SAM2 code in `./sam2/` and does **not** require installing SAM2 as a separate package.
If you already have a working environment for SAM2, you can reuse it. Otherwise, you may create a new conda environment:

```shell
conda create -n SDDNet python=3.10
conda activate SDDNet
pip install -r requirements.txt
```

## Training

### Download Weights

Download the pretrained weights file `sam2_hiera_large.pt`. The download link is wrapped below:

[Download Weights](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

### Run Training

Use the `train.sh` script to start training. Make sure the weights file is placed in the correct path (e.g., `sam2_hiera_large.pt` in the current directory).

Example `train.sh` content:

```bash
#!/bin/bash
# Training script example
CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "path/to/hiera_checkpoint.pt" \
--train_image_path "path/to/train/images" \
--train_mask_path "path/to/train/masks" \
--save_path "path/to/save/checkpoints" \
--epoch 400 \
--lr 0.001 \
--batch_size 8
```

## Test

Use the `test.sh` script to run testing.

### Run Testing

Example `test.sh` content:

```bash
#!/bin/bash
# Testing script example
CUDA_VISIBLE_DEVICES="" \
python test.py \
--checkpoint "path/to/checkpoint.pth" \
--test_image_path "path/to/test/images" \
--test_gt_path "path/to/test/ground_truth" \
--save_path "path/to/save/predictions" \
```

## Eval

Use the `eval.sh` script to run testing.

### Run Eval

Example `eval.sh` content:

```bash
#!/bin/bash
# Eval script example
python eval-sod.py \
--dataset_name "dataset_name" \
--pred_path "path/to/predictions" \
--gt_path "path/to/ground_truth" \
```


## 📥 预测图下载

预测结果已上传至百度网盘（提取码：`m396`），点击下面链接下载：  
链接: https://pan.baidu.com/s/12bkGTBcI1kqNXj1XKZVsrg?pwd=m396

## 📂 数据集下载

### SD-Saliency-900 分割数据集

- **百度网盘链接**：  
  链接: https://pan.baidu.com/s/1WDahh0tT-_pMhGdkWSoMjQ?pwd=m396
- **提取码**：`m396`

The code will be available here soon, after the paper is accepted.
