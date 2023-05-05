# FBI-Learning
Implementation of "Foreground/Background-Masked Interaction Learning for Spatio-temporal Action Detection"
## Installation
- Python>=3.6
- PyTorch=1.8.1 and corresponding torchvision=0.9.0 (if use other versions, note the differences in the torch.nn.MultiheadAttention parameter list)
- Pillow>=8.1.1
- PyYAML>=5.4
- easydict
- numpy
- tensorboardX>=2.0
## Preparation
### Data Preparation
Please refer to [here](https://github.com/open-mmlab/mmaction2/tree/0.x/tools/data/ava) for AVA dataset preparation.
### Annotation Preparation
- Download the official annotation from [here](https://research.google.com/ava/download.html#ava_actions_download)
- Download the AVA [train annotation](https://drive.google.com/file/d/1CsCUVxdxVyZ5vUM2eGzzV42wzKxPa7bK/view) and [val annotation](https://drive.google.com/file/d/1uTlgYtR_zt85JCx-HoqNNXWwCilUTd9w/view) for FBIL.
- Put the annotation files into annotations folder.
### Pre-trained Model
Download the [SlowFast model pre-trained](https://drive.google.com/file/d/1qDdAntE5Onh7btniftOL8MrbsD7OIqj4/view) on K-700 datasets and put it into pretrained folder
## Train
```
python main.py --config .\configs\SLOWFAST_R101_FBIL.yaml --nproc_per_node 2
```
## Val
```
python main.py --config .\configs\EVAL_SLOWFAST_R101_FBIL.yaml
```
