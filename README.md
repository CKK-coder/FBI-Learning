# FOR CHINESE VERSION SEE README_CHINESE.md
# Foreground/Background-Masked Interaction Learning for Spatio-temporal Action Detection
## INTRODUCTION
Aiming at the problem of effectively learning multi granularity spatio-temporal representations of data in the context of multi event detection in video media, a fusion mechanism combining front/background multi granularity action representations is proposed. The front/background masking strategy is adopted to narrow the search range of the attention mechanism while using random masks to alleviate the overfitting problem caused by high correlation semantics between queries and key values. Based on the spatiotemporal representation of event subjects, the front/background multi granularity features are fused to promote the effective learning of multi granularity spatiotemporal representations in media data by the model.
## REQUIREMENT
- Python>=3.6
- PyTorch=1.8.1 and corresponding torchvision=0.9.0 (if use other versions, note the differences in the torch.nn.MultiheadAttention parameter list)
- Pillow>=8.1.1
- PyYAML>=5.4
- easydict
- numpy
- tensorboardX>=2.0
## PREPARATION
### DATASET
From [HERE](https://github.com/open-mmlab/mmaction2/tree/0.x/tools/data/ava) to prepare AVA dataset。
### ANNOTATIONS
- From [Here](https://research.google.com/ava/download.html#ava_actions_download) download annotations。
- Download AVA [train annotations](https://drive.google.com/file/d/1CsCUVxdxVyZ5vUM2eGzzV42wzKxPa7bK/view) and [val annotation](https://drive.google.com/file/d/1uTlgYtR_zt85JCx-HoqNNXWwCilUTd9w/view) 。
- Place these annotation files in the 'annotations' folder
### PRETRAINED CHECKPOINTS
From [SlowFast model pre-trained](https://drive.google.com/file/d/1qDdAntE5Onh7btniftOL8MrbsD7OIqj4/view) download the pretrained weights on the K-700 dataset and place them in the 'pre_trained' folder.
## TRAIN
```
python main.py --config .\configs\SLOWFAST_R101_FBIL.yaml --nproc_per_node 2
```
## VAL
```
python main.py --config .\configs\EVAL_SLOWFAST_R101_FBIL.yaml
```
