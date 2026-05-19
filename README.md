# PneumonMamba

A novel Mamba-based Network for Computer-Aided Diagnosis of Pneumonia with Omnidirectional Scanning and Multi-Scale Convolution.

## Overview

PneumonMamba is a deep learning framework based on the State Space Model (SSM) for automated pneumonia classification from chest X-ray images. The model introduces three key innovations:

1. **Omnidirectional State Space Model (OSSM)**: An 8-directional scanning mechanism that captures spatial dependencies from all orientations, overcoming the unidirectional limitation of vanilla Mamba.

2. **Multi-scale Dilated Residual Module (MDR)**: Employs asymmetric convolution decomposition with varying dilation rates to extract multi-scale contextual features efficiently.

3. **Dual Attention Module (DAM)**: Combines channel attention with multi-scale spatial attention using depthwise separable convolutions at different kernel sizes.

## Project Structure

```
PneumonMamba/
├── checkpoints/          # Saved model weights
├── configs/              # Training configurations
│   └── default.py
├── data/                 # Dataset directory (not included)
├── docs/                 # Documentation
├── models/               # Model architecture
│   ├── __init__.py
│   ├── pneumonmamba.py   # Main model (VSSM)
│   └── csms6s_plus.py    # Cross-scan operations
├── results/              # Training results and logs
├── scripts/              # Training and evaluation scripts
│   └── train.py
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── dual_attention.py # Dual Attention Module (DAM)
│   ├── multi_scale_conv.py # Multi-scale Dilated Residual (MDR)
│   └── wtconv2d.py       # Wavelet Transform Convolution
├── README.md
└── requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.13.0
- CUDA >= 11.7

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The model is designed for chest X-ray classification. Organize your dataset as follows:

```
data/
├── train/
│   ├── COVID/
│   ├── Lung_Opacity/
│   ├── Normal/
│   └── Viral_Pneumonia/
├── val/
│   ├── COVID/
│   ├── ...
└── test/
    ├── COVID/
    └── ...
```

## Training

```bash
cd scripts
python train.py \
    --data_dir ../data \
    --num_classes 4 \
    --batch_size 128 \
    --epochs 500 \
    --lr 0.0001 \
    --model_size tiny \
    --save_path ../checkpoints/best_model.pth
```

### Model Variants

| Variant | Depths | Dims | Parameters |
|---------|--------|------|------------|
| Tiny    | [2,2,4,2] | [96,192,384,768] | ~25M |
| Small   | [2,2,8,2] | [96,192,384,768] | ~44M |
| Base    | [2,2,12,2] | [128,256,512,1024] | ~88M |

## Architecture

The PneumonMamba architecture follows a hierarchical encoder design:

1. **Patch Embedding**: Converts input images to patch tokens
2. **VSS Layers**: Each layer contains SS_Conv_SSM blocks that split features into two branches:
   - **Convolutional branch**: Standard 3x3 convolutions for local feature extraction
   - **SSM branch**: GroupMambaLayer with 8-directional scanning for global context
3. **Channel Shuffle**: Mixes information between the two branches
4. **Patch Merging**: Downsamples between stages
5. **Classification Head**: Global average pooling + linear classifier

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{chen2026pneumomamba,
  title={PneumoMamba: A Novel Mamba and CNN Dual-Path Network for Computer-Aided Diagnosis of Pneumonia Using Omnidirectional Feature Extraction and Multi-Scale Asymmetric Convolution},
  author={Chen, Meng and Gu, Yu and Wang, ManSheng and Yang, Lidong and Zhang, Baohua and Li, Jianjun and Liu, Xin and Hao, Juan and Ma, Hao and Zhang, Wei and Tang, Siyuan and He, Qun},
  journal={International Journal of Imaging Systems and Technology},
  year={2026},
  publisher={Wiley},
  doi={10.1002/ima.70367}
}
```

**Paper link**: [https://doi.org/10.1002/ima.70367](https://doi.org/10.1002/ima.70367)

## Acknowledgements

This work builds upon:
- [Mamba](https://github.com/state-spaces/mamba) - Selective State Space Model
- [VMamba](https://github.com/MzeroMiko/VMamba) - Visual State Space Model
- [WTConv2d](https://github.com/BGU-CS-VIL/WTConv2d) - Wavelet Transform Convolution
