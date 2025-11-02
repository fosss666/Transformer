# 手动搭建Transformer并进行文本摘要任务

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
  
  