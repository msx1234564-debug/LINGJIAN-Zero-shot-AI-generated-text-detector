# 灵鉴
**灵鉴检测器的开源代码**

## 数据
以下文件夹为实验数据集：
* ./exp_Open_source_model -> 开源生成模型实验.
* ./exp_API-based_model -> 闭源生成模型实验.

## 代理模型加载
huggingface上下载
* BART-base: https://huggingface.co/facebook/bart-base
* OPT-350m: https://huggingface.co/facebook/opt-350m


## 环境
* Python3.8
* PyTorch2.1.0

GPU: NVIDIA 3090 GPU with 24GB memory

## 快速开始
Please run following commands for a demo:
请运行以下指令以开始复现实验
```
python eval.py
```



