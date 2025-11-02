# 手动搭建Transformer并进行文本摘要任务

 本项目实现了基于字符级Transformer的文本摘要模型，通过对比不同参数配置的性能，验证模型在小样本场景下的有效性，并通过消融实验分析核心组件的作用。  ## 1. 项目简介 - **任务**：基于字符级输入的文本摘要生成（抽取式+生成式结合） - **模型**：简化版Transformer（编码器-解码器结构） - **核心实验**：参数敏感性分析、消融实验、性能评估（ROUGE分数、困惑度等）  ## 2. 环境配置 ### 2.1 硬件要求 - 推荐GPU：NVIDIA GTX 1080Ti及以上（支持CUDA 11.3） - 内存：≥16GB（处理中等规模数据集） ### 2.2 软件依赖 - 操作系统：Linux（Ubuntu 20.04+）/ Windows 10+ - 依赖安装：  ```bash  # 克隆仓库后，在项目根目录执行  pip install -r requirements.txt

## 简介

基于手动搭建的Transformer模型进行文本摘要训练，对比注意力头数、初始学习率和前馈神经网络维度对效果的影响，通过消融实验分析位置编码、多头注意力和残差链接及归一化对模型效果的影响。

## 硬件要求

NVIDIA GTX 4090

显存24GB

## 运行环境

+ 创建并激活虚拟环境

  ```python
  conda create -n transformer python=3.9.7 -y
  conda activate transformer
  ```

* 安装依赖

  ```python
  pip install -r requirements.txt
  ```

+ 运行脚本

  ````
  # 先将data中的数据集解压！
  
  # 使用自定义配置训练
  sh run.sh --config configs/config.yaml
  
  # 使用自定义配置测试
  sh run.sh --config configs/config.yaml --test
  
  # 命令行参数覆盖配置
  sh run.sh --config configs/config.yaml --lr 5e-4
  ````
  
  