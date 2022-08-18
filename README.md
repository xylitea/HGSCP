# 基于混合图表示的软件变更预测方法
该存储仓库包含我们题为 “基于混合图表示的软件变更预测方法” 的论文中的代码和数据。在本文中，我们提出了一种新颖的基于混合图的变更预测方法HGSCP，我们首先结合抽象语法树、控制流和数据流等信息对代码的语法结构和语义信息进行建模；然后，利用图神经网络学习出强有力的特征表示用于预测代码的变更倾向性。

### 版本
+ python 3.8<br>
+ pandas 1.2.0<br>
+ scikit-learn 0.24.0<br>
+ pytorch 1.7.1<br>
+ pytorch-geometric 1.7.2<br>
+ pycparser 2.20<br>
+ javalang 0.11.0<br>
+ RAM 16GB or more<br>
+ GPU with CUDA support<br>

### 安装
通过pip安装依赖包:

	$ pip install pandas==1.2.0 scikit-learn==0.24.0 pycparser==2.20 javalang==0.11.0

请根据您的环境安装 pytorch，参考 https://pytorch.org/ <br>
请根据您的环境安装 pytorch-geometric，参考 https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

### 数据集
我们的数据集在“data”文件夹中，包含了所有项目的数据； 它是一个“pkl”文件，使用 pandas 库很容易进行处理。 每个文件包含三列，即文件id、代码和标签。 <br>

### 变更预测

1. 在train.py中设置projects参数，选择用于实验的项目
2. 运行 train.py 得到实验结果
3. 运行compute_metrics.py 计算指标

### 最后感谢 https://github.com/jacobwwh/graphmatch_clone 仓库提供的参考和灵感。
