# 基于前/背景多粒度时空表征的事件检测框架
## 算法描述
针对视频媒体的多事件检测背景下，如何有效学习数据的多粒度时空表征问题，提出了一种结合前/背景多粒度的动作表征的融合机制，采用前/背景掩码策略缩小注意力机制的搜索范围同时利用随机掩码缓解由于查询与键值之间存在的高相关语义带来的区域过拟合问题，并且基于事件主体的时空表征对前/背景多粒度特征进行融合，促进模型有效学习媒体数据中的多粒度时空表征。
## 环境依赖
- Python>=3.6
- PyTorch=1.8.1 and corresponding torchvision=0.9.0 (if use other versions, note the differences in the torch.nn.MultiheadAttention parameter list)
- Pillow>=8.1.1
- PyYAML>=5.4
- easydict
- numpy
- tensorboardX>=2.0
## 准备工作
### 数据准备
可以参考 [这里](https://github.com/open-mmlab/mmaction2/tree/0.x/tools/data/ava) 准备AVA数据集。
### 标注准备
- 从 [这里](https://research.google.com/ava/download.html#ava_actions_download) 下载官方标注文件。
- 下载 AVA [训练标注](https://drive.google.com/file/d/1CsCUVxdxVyZ5vUM2eGzzV42wzKxPa7bK/view) and [验证标注](https://drive.google.com/file/d/1uTlgYtR_zt85JCx-HoqNNXWwCilUTd9w/view) 。
- 将这些标注文件放置''annotations''文件夹下.
### 预训练权重
从 [SlowFast model pre-trained](https://drive.google.com/file/d/1qDdAntE5Onh7btniftOL8MrbsD7OIqj4/view) 下载在K-700 数据集上的预训练权重并放置''pretrained''文件夹下。
## 训练
```
python main.py --config .\configs\SLOWFAST_R101_FBIL.yaml --nproc_per_node 2
```
## 验证
```
python main.py --config .\configs\EVAL_SLOWFAST_R101_FBIL.yaml
```
